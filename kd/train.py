import os
# prevent any accidental DeepSpeed path during unwrap/saving
os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")

import argparse, time, signal, pathlib
import torch
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import get_cosine_schedule_with_warmup

from kd.models import load_student
from kd.kd_rb import response_kd_loss
from kd.kd_fb import feature_kd_loss, LinearProjector
from kd.kd_relb import relation_kd_loss
from kd.datasets import (
    RBTopKIterableDataset, FBDataset, RelBDataset,
    collate_rb, collate_pad
)

# ------------------------- Arg parsing -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kd.mode', dest="kd_mode", choices=['rb', 'fb', 'relb'], required=True)
    ap.add_argument('--student', type=str, required=True)
    ap.add_argument('--data', type=str, required=True, help="Parquet path glob")
    ap.add_argument('--seq_len', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=1)

    # accept both spellings; some launchers pass --bash_size
    ap.add_argument('--batch_size', '--bash_size', dest='batch_size', type=int, default=2)

    ap.add_argument('--warmup_steps', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=1000)

    # RB
    ap.add_argument('--rb.topk', dest='rb_topk', type=int, default=16)
    ap.add_argument('--rb.temperature', dest='rb_temperature', type=float, default=2.0)

    # FB
    ap.add_argument('--fb.teacher_layer', dest='fb_teacher_layer', type=int, default=22)
    ap.add_argument('--fb.student_layer', dest='fb_student_layer', type=int, default=12)
    ap.add_argument('--fb.token_subset_ratio', dest='fb_token_subset_ratio', type=float, default=0.25)

    # RELB
    ap.add_argument('--relb.lambda_dist',  dest='relb_lambda_dist',  type=float, default=1.0)
    ap.add_argument('--relb.lambda_angle', dest='relb_lambda_angle', type=float, default=0.5)

    # LoRA
    ap.add_argument('--lora.r',     dest='lora_r',     type=int, default=16)
    ap.add_argument('--lora.alpha', dest='lora_alpha', type=int, default=32)

    # Checkpointing
    ap.add_argument('--save-dir',   dest='save_dir',   type=str, required=True, help='Root directory to run + checkpoints')
    ap.add_argument('--save_every', type=int, default=0, help='Steps between checkpoints (0=off)')
    ap.add_argument('--resume',     type=str, default='auto', choices=['auto','none','path'])
    ap.add_argument('--resume_path', type=str, default='')
    return ap.parse_args()

# ------------------------- Checkpoint utils -------------------------
def _latest_ckpt(root: str):
    p = pathlib.Path(root)
    if not p.exists(): return None
    cks = sorted(p.glob("ckpt_step*"), key=lambda x: x.name)
    return str(cks[-1]) if cks else None

def _unwrap_for_save(model: torch.nn.Module) -> torch.nn.Module:
    # Works for DDP without importing deepspeed/accelerate unwrap
    return model.module if hasattr(model, "module") else model

def _save_ckpt(step, model, tok, optimizer, scheduler, save_dir):
    ck = pathlib.Path(save_dir) / f"ckpt_step{step:07d}"
    ck.mkdir(parents=True, exist_ok=True)

    base = _unwrap_for_save(model)
    if hasattr(base, "save_pretrained"):
        base.save_pretrained(ck.as_posix())
    else:
        torch.save(base.state_dict(), ck / "pytorch_model.bin")

    try:
        tok.save_pretrained(ck.as_posix())
    except Exception:
        pass

    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, ck / "trainer_state.pt")

def _load_ckpt(path, model, tok, optimizer, scheduler):
    from transformers import AutoTokenizer
    # copy pad token only if missing
    tok_init = AutoTokenizer.from_pretrained(path, use_fast=True)
    if getattr(tok, "pad_token", None) is None and getattr(tok_init, "pad_token", None) is not None:
        tok.pad_token = tok_init.pad_token

    # try adapters first if this is a PEFT model
    base = _unwrap_for_save(model)
    if hasattr(base, "load_adapter"):
        try:
            base.load_adapter(path, "default", is_trainable=True)
        except Exception:
            base.load_adapter(path, is_trainable=True)
    else:
        try:
            loaded = base.__class__.from_pretrained(path)
            base.load_state_dict(loaded.state_dict(), strict=False)
        except Exception:
            pass

    st = torch.load(pathlib.Path(path) / "trainer_state.pt", map_location="cpu")
    optimizer.load_state_dict(st["optimizer"])
    scheduler.load_state_dict(st["scheduler"])
    return int(st.get("step", 0))

def _zero_touch_all_params(model: torch.nn.Module) -> torch.Tensor:
    z = None
    for p in model.parameters():
        t = p.view(-1)[0] * 0.0         # 0-weighted read, keeps graph connected
        z = t if z is None else (z + t)
    if z is None:
        z = torch.zeros((), device=next(model.parameters()).device)
    return z

# ------------------------- GC helpers (avoid re-entrant backward with LoRA) -------------------------
def _maybe_disable_use_cache(m):
    try:
        if hasattr(m, "config") and getattr(m.config, "use_cache", None) is True:
            m.config.use_cache = False
    except Exception:
        pass

def _enable_gc_nonreentrant(model) -> bool:
    """
    Try to enable gradient checkpointing with use_reentrant=False.
    If unsupported, disable GC to avoid DDP 'mark ready twice'.
    """
    _maybe_disable_use_cache(model)
    # Try directly
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return True
    except Exception:
        pass
    # Try common base attributes
    for attr in ("base_model", "model", "module"):
        base = getattr(model, attr, None)
        if base is None:
            continue
        try:
            _maybe_disable_use_cache(base)
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            return True
        except Exception:
            continue
    # Disable GC if nothing worked
    try:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
    except Exception:
        pass
    return False

# ------------------------- Iterable sharder -------------------------
class _ShardIter(IterableDataset):
    def __init__(self, ds, rank: int, world: int):
        self.ds = ds
        self.rank = rank
        self.world = world
    def __iter__(self):
        for i, item in enumerate(self.ds):
            if (i % self.world) == self.rank:
                yield item

# ------------------------- Main -------------------------
def main():
    args = parse_args()

    # DDP config — we explicitly touch all buckets, so no unused-param search
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        static_graph=False,
        gradient_as_bucket_view=True
    )
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    device = accelerator.device
    rank  = accelerator.process_index
    world = accelerator.num_processes

    # ----- build model BEFORE prepare so we can set GC safely
    model, tok = load_student(args.student, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    # Make GC non-reentrant (or disable). This avoids re-entrant backward + LoRA + DDP crash
    if not _enable_gc_nonreentrant(model):
        print("[GC] Non-reentrant checkpointing unavailable -> gradient checkpointing disabled for stability.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    # prepare model/opt/sched only (do NOT prepare the DataLoader for iterable ds)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # ---- [Checkpoint] resume detection ----
    save_dir = args.save_dir
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    def handle_sigusr1(signum, frame):
        try:
            if accelerator.is_main_process:
                _save_ckpt(step, model, tok, optimizer, scheduler, save_dir)
                print(f"[SIGNAL] Saved checkpoint at step={step} due to SIGUSR1")
        finally:
            pass
    signal.signal(signal.SIGUSR1, handle_sigusr1)

    if args.resume == 'auto':
        lp = _latest_ckpt(save_dir)
        if lp:
            step = _load_ckpt(lp, model, tok, optimizer, scheduler)
            if accelerator.is_main_process:
                print(f"[RESUME] Resumed from {lp} at step={step}")
    elif args.resume == 'path' and args.resume_path:
        step = _load_ckpt(args.resume_path, model, tok, optimizer, scheduler)
        if accelerator.is_main_process:
            print(f"[RESUME] Resumed from {args.resume_path} at step={step}")
    else:
        if accelerator.is_main_process:
            print("[RESUME] Starting fresh")

    # ---- Dataset + collate ----
    if args.kd_mode == 'rb':
        dataset = RBTopKIterableDataset(args.data)
        collate = collate_rb
    elif args.kd_mode == 'fb':
        dataset = FBDataset(args.data, teacher_layer=args.fb_teacher_layer)
        collate = collate_pad
    else:  # relb
        dataset = RelBDataset(args.data)
        collate = collate_pad

    # Shard per-rank to avoid Accelerate concatenation of uneven iterable batches
    dataset = _ShardIter(dataset, rank, world)

    # Important: do NOT wrap with accelerator.prepare for IterableDataset
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,   # per-rank batch size
        collate_fn=collate,
        drop_last=True,               # keep ranks aligned
        num_workers=4,
        pin_memory=True
    )

    model.train()
    projector = None
    t0 = time.time()
    total_tokens = 0

    for epoch in range(args.epochs):
        for batch in loader:
            if step >= args.max_steps:
                break

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attn_mask = batch['attn_mask'].to(device, non_blocking=True)

            if args.kd_mode == 'rb':
                with accelerator.autocast():
                    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                    s_logits = out.logits[:, :-1, :]
                    min_len = min(s_logits.size(1), batch['topk_ids'].size(1))
                    kd = response_kd_loss(
                        s_logits[:, :min_len, :],
                        batch['topk_ids'][:, :min_len, :].to(device, non_blocking=True),
                        batch['topk_logprobs'][:, :min_len, :].to(device, non_blocking=True),
                        T=args.rb_temperature
                    )
                    loss = kd
                token_this = (attn_mask.sum() - input_ids.size(0)).item()

            elif args.kd_mode == 'fb':
                with accelerator.autocast():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                                use_cache=False, output_hidden_states=True)

                    s_hid = out.hidden_states[args.fb_student_layer]                 # [B,T,Hs]
                    t_feats = batch['teacher_feats'].to(device, non_blocking=True)   # [B,T,Ht]

                    if projector is None:
                        projector = LinearProjector(s_hid.size(-1), t_feats.size(-1)).to(device)
                        projector = accelerator.prepare(projector)
                        # ensure projector is optimized
                        optimizer.add_param_group({"params": projector.parameters()})

                    s_proj = projector(s_hid)                                        # [B,T,Ht]

                    # dtype alignment
                    if t_feats.dtype != s_proj.dtype:
                        t_feats = t_feats.to(s_proj.dtype)

                    loss = feature_kd_loss(s_proj, t_feats, token_mask=attn_mask)

                    # DDP safety: ensure ALL params/buckets are “seen”
                    loss = loss + out.logits.mean() * 0.0         # keep forward graph tied
                    loss = loss + _zero_touch_all_params(model)   # touch every param/bucket

                token_this = attn_mask.sum().item()

            else:  # relb
                with accelerator.autocast():
                    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False, output_hidden_states=True)
                    last = out.hidden_states[-1]                # [B, T, H]
                    mask = attn_mask.unsqueeze(-1)              # [B, T, 1]
                    pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # [B, H]
                    t_emb = batch['teacher_embed'].to(device, non_blocking=True)      # [B, H]
                    loss = relation_kd_loss(
                        pooled, t_emb,
                        lambda_dist=args.relb_lambda_dist,
                        lambda_angle=args.relb_lambda_angle
                    )
                token_this = attn_mask.sum().item()

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            if accelerator.is_main_process and args.save_every > 0 and step > 0 and (step % args.save_every == 0):
                _save_ckpt(step, model, tok, optimizer, scheduler, save_dir)
                print(f"[ckpt] Saved {save_dir}/ckpt_step{step:07d}")

            total_tokens += token_this
            step += 1
            if accelerator.is_main_process and step % 10 == 0:
                dt = time.time() - t0
                tps = total_tokens / max(dt, 1e-6)
                print(f"[step {step}] loss={loss.item():.4f} tokens={int(total_tokens)} tok/s={tps:.1f}")

        if step >= args.max_steps:
            break

    if accelerator.is_main_process:
        pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        base = _unwrap_for_save(model)
        if hasattr(base, "save_pretrained"):
            base.save_pretrained(args.save_dir)
        else:
            torch.save(base.state_dict(), pathlib.Path(args.save_dir) / "pytorch_model.bin")
        try:
            tok.save_pretrained(args.save_dir)
        except Exception:
            pass
        print(f"Saved to {args.save_dir}")

if __name__ == '__main__':
    main()
