import os
# prevent any accidental DeepSpeed path during unwrap/saving
os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")

import argparse, time, signal, pathlib, glob, random
import numpy as np

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

    # lowered default LR for stability
    ap.add_argument('--lr', type=float, default=5e-5)

    ap.add_argument('--epochs', type=int, default=1)

    # accept both spellings; some launchers pass --bash_size
    ap.add_argument('--batch_size', '--bash_size', dest='batch_size', type=int, default=2)

    ap.add_argument('--warmup_steps', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=1000)  # <=0 means "no early stop"

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

    # Extra checkpoint knobs (optional)
    ap.add_argument('--files_per_ckpt', type=int, default=0,
                    help='Also checkpoint every N processed files (0=off)')
    return ap.parse_args()

# ------------------------- Checkpoint utils -------------------------
def _latest_ckpt(root: str):
    p = pathlib.Path(root)
    if not p.exists(): return None
    cks = sorted(p.glob("ckpt_step*"), key=lambda x: x.name)
    return str(cks[-1]) if cks else None

def _unwrap_for_save(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model

def _rng_pack():
    return {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
        "cuda": [t.tolist() for t in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [],
    }

def _rng_unpack(st):
    try:
        random.setstate(st["py_random"])
        np.random.set_state(tuple(st["np_random"]))
        torch.set_rng_state(torch.tensor(st["torch"], dtype=torch.uint8))
        if torch.cuda.is_available() and st.get("cuda"):
            for d, s in enumerate(st["cuda"]):
                torch.cuda.set_rng_state(torch.tensor(s, dtype=torch.uint8), device=d)
    except Exception:
        pass

def _save_ckpt(step, model, tok, optimizer, scheduler, save_dir, dataset=None, projector=None):
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

    if projector is not None:
        torch.save(projector.state_dict(), ck / "projector.pt")

    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "dataset": getattr(dataset, "state", lambda: {})(),
        "rng": _rng_pack(),
    }
    torch.save(state, ck / "trainer_state.pt")

    # cheap "last" symlink for convenience
    try:
        tgt = pathlib.Path(save_dir) / "last"
        if tgt.exists() or tgt.is_symlink():
            tgt.unlink()
        tgt.symlink_to(ck.name, target_is_directory=True)
    except Exception:
        pass

def _load_ckpt(path, model, tok, optimizer, scheduler, dataset=None, projector=None):
    from transformers import AutoTokenizer

    tok_init = AutoTokenizer.from_pretrained(path, use_fast=True)
    if getattr(tok, "pad_token", None) is None and getattr(tok_init, "pad_token", None) is not None:
        tok.pad_token = tok_init.pad_token

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

    st_path = pathlib.Path(path) / "trainer_state.pt"
    st = {}
    if st_path.exists():
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        if "optimizer" in st:
            optimizer.load_state_dict(st["optimizer"])
        if "scheduler" in st:
            scheduler.load_state_dict(st["scheduler"])

    if dataset is not None and "dataset" in st and hasattr(dataset, "load_state"):
        dataset.load_state(st["dataset"])

    if "rng" in st:
        _rng_unpack(st["rng"])

    proj_sd = None
    proj_path = pathlib.Path(path) / "projector.pt"
    if proj_path.exists():
        if projector is not None:
            projector.load_state_dict(torch.load(proj_path, map_location="cpu"))
        else:
            proj_sd = torch.load(proj_path, map_location="cpu")

    return int(st.get("step", 0)), proj_sd

def _zero_touch_all_params(model: torch.nn.Module) -> torch.Tensor:
    params = [p for p in model.parameters()]
    if not params:
        return torch.zeros((), device='cpu')
    z = params[0].view(-1)[0] * 0.0
    for p in params[1:]:
        z = z + p.view(-1)[0] * 0.0
    return z

# ------------------------- GC helpers -------------------------
def _maybe_disable_use_cache(m):
    try:
        if hasattr(m, "config") and getattr(m.config, "use_cache", None) is True:
            m.config.use_cache = False
    except Exception:
        pass

def _enable_gc_nonreentrant(model) -> bool:
    _maybe_disable_use_cache(model)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return True
    except Exception:
        pass
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
    try:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
    except Exception:
        pass
    return False

# ------------------------- Iterable sharder -------------------------
class _ShardIter(IterableDataset):
    """
    Wraps an IterableDataset and shards it by global index % world == rank.
    Keeps an index counter _i so resume can approximately continue.
    """
    def __init__(self, ds, rank: int, world: int):
        super().__init__()
        self.ds, self.rank, self.world = ds, rank, world
        self._i = 0

    def state(self):
        base = getattr(self.ds, "state", lambda: {})()
        base.update({"_i": self._i, "rank": self.rank, "world": self.world})
        return base

    def load_state(self, st):
        if hasattr(self.ds, "load_state"):
            self.ds.load_state(st)
        self._i = int(st.get("_i", 0))

    def __iter__(self):
        for item in self.ds:
            if (self._i % self.world) == self.rank:
                yield item
            self._i += 1

# ------------------------- Deterministic workers -------------------------
def _worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)

# ------------------------- Main -------------------------
def main():
    args = parse_args()
    print(f"[ARGS] {args}")

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        static_graph=False,
        gradient_as_bucket_view=True
    )
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    device = accelerator.device
    rank  = accelerator.process_index
    world = accelerator.num_processes

    print(f"[INFO] mixed_precision={accelerator.mixed_precision} world={world} rank={rank}")
    print(f"[INFO] batch_size={args.batch_size} seq_len={args.seq_len}")

    model, tok = load_student(args.student, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    if not _enable_gc_nonreentrant(model):
        print("[GC] Non-reentrant checkpointing unavailable -> gradient checkpointing disabled for stability.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps_for_sched = max(args.max_steps, 1)  # safe when max_steps<=0
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps_for_sched)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # ---- Dataset + collate
    if args.kd_mode == 'rb':
        dataset = RBTopKIterableDataset(
            args.data,
            out_dir=os.path.join(args.save_dir, "rb_progress")
        )
        collate = collate_rb
    elif args.kd_mode == 'fb':
        dataset = FBDataset(
            args.data,
            teacher_layer=args.fb_teacher_layer,
            out_dir=os.path.join(args.save_dir, "fb_progress")
        )
        collate = collate_pad
    else:  # relb
        dataset = RelBDataset(
            args.data,
            out_dir=os.path.join(args.save_dir, "relb_progress")
        )
        collate = collate_pad

    dataset = _ShardIter(dataset, rank, world)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        drop_last=True,
        num_workers=1,
        pin_memory=False,
        worker_init_fn=_worker_init_fn,
    )

    # ---- resume / checkpoint prep
    save_dir = args.save_dir
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    projector = None
    pending_proj_sd = None

    # ---------- Signal handlers ----------
    def _save_and_note():
        if accelerator.is_main_process:
            _save_ckpt(
                step, model, tok, optimizer, scheduler, save_dir,
                dataset=dataset, projector=projector
            )
            print(f"[SIGNAL ✅] Saved checkpoint at step={step}")

    def handle_sigusr1(signum, frame):
        try:
            accelerator.wait_for_everyone()
            _save_and_note()
        finally:
            pass

    signal.signal(signal.SIGUSR1, handle_sigusr1)
    signal.signal(signal.SIGTERM, handle_sigusr1)

    if args.resume == 'auto':
        lp = _latest_ckpt(save_dir)
        if lp:
            step, pending_proj_sd = _load_ckpt(
                lp, model, tok, optimizer, scheduler,
                dataset=dataset, projector=None
            )
            if accelerator.is_main_process:
                print(f"[RESUME 💾] Resumed from {lp} at step={step}")
    elif args.resume == 'path' and args.resume_path:
        step, pending_proj_sd = _load_ckpt(
            args.resume_path, model, tok, optimizer, scheduler,
            dataset=dataset, projector=None
        )
        if accelerator.is_main_process:
            print(f"[RESUME 💾] Resumed from {args.resume_path} at step={step}")
    else:
        if accelerator.is_main_process:
            print("[RESUME ⚠️] Starting fresh")

    model.train()
    t0 = time.time()
    total_tokens = 0

    for epoch in range(args.epochs):
        for batch in loader:
            # stop early if requested
            if args.max_steps > 0 and step >= args.max_steps:
                break

            # 1. pull CPU batch
            input_ids = batch['input_ids']
            attn_mask = batch['attn_mask']

            topk_ids = batch.get('topk_ids', None)
            topk_logprobs = batch.get('topk_logprobs', None)
            teacher_feats = batch.get('teacher_feats', None)
            teacher_embed = batch.get('teacher_embed', None)

            # 2. truncate on CPU
            if args.seq_len > 0:
                S = args.seq_len

                input_ids = input_ids[:, :S]
                attn_mask = attn_mask[:, :S]

                if args.kd_mode == 'rb':
                    if topk_ids is not None:
                        topk_ids = topk_ids[:, :max(S - 1, 0), :]
                    if topk_logprobs is not None:
                        topk_logprobs = topk_logprobs[:, :max(S - 1, 0), :]

                elif args.kd_mode == 'fb':
                    if teacher_feats is not None:
                        teacher_feats = teacher_feats[:, :S, :]
                # relb: teacher_embed is [B,H] already

            # 3. move to GPU
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            if topk_ids is not None:
                topk_ids = topk_ids.to(device, non_blocking=True)
            if topk_logprobs is not None:
                topk_logprobs = topk_logprobs.to(device, non_blocking=True)
            if teacher_feats is not None:
                teacher_feats = teacher_feats.to(device, non_blocking=True)
            if teacher_embed is not None:
                teacher_embed = teacher_embed.to(device, non_blocking=True)

            # 4. forward + loss
            if args.kd_mode == 'rb':
                with accelerator.autocast():
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        use_cache=False
                    )
                    s_logits = out.logits[:, :-1, :]  # [B, T-1, V]

                    min_len = min(
                        s_logits.size(1),
                        topk_ids.size(1),
                        topk_logprobs.size(1),
                    )
                    kd = response_kd_loss(
                        s_logits[:, :min_len, :],
                        topk_ids[:, :min_len, :],
                        topk_logprobs[:, :min_len, :],
                        T=args.rb_temperature
                    )
                    loss = kd

                token_this = (attn_mask.sum() - input_ids.size(0)).item()

            elif args.kd_mode == 'fb':
                with accelerator.autocast():
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        use_cache=False,
                        output_hidden_states=True
                    )

                    s_hid = out.hidden_states[args.fb_student_layer]  # [B,T,Hs]
                    t_feats = teacher_feats  # [B,T,Ht] on device

                    if projector is None:
                        projector = LinearProjector(
                            s_hid.size(-1),
                            t_feats.size(-1)
                        ).to(device)
                        projector = accelerator.prepare(projector)
                        optimizer.add_param_group({"params": projector.parameters()})
                        if pending_proj_sd is not None:
                            projector.load_state_dict(pending_proj_sd)
                            pending_proj_sd = None

                    s_proj = projector(s_hid)
                    if t_feats.dtype != s_proj.dtype:
                        t_feats = t_feats.to(s_proj.dtype)

                    loss = feature_kd_loss(s_proj, t_feats, token_mask=attn_mask)
                    # anchors to keep graph touching model params
                    loss = loss + out.logits.mean() * 0.0
                    loss = loss + _zero_touch_all_params(model)

                token_this = attn_mask.sum().item()

            else:  # relb
                with accelerator.autocast():
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        use_cache=False,
                        output_hidden_states=True
                    )

                    last = out.hidden_states[-1]        # [B, T, H] (bf16)
                    mask = attn_mask.unsqueeze(-1)      # [B, T, 1] (long -> broadcast ok)

                    pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # [B, H]
                    t_emb = teacher_embed  # [B, H]

                pooled_f32 = pooled.float()
                t_emb_f32 = t_emb.float()

                Bsz = pooled_f32.size(0)
                if Bsz < 2:
                    # not enough samples to form stable pairwise relations
                    loss = (pooled_f32 * 0).sum()
                else:
                    loss = relation_kd_loss(
                        pooled_f32,
                        t_emb_f32,
                        lambda_dist=args.relb_lambda_dist,
                        lambda_angle=args.relb_lambda_angle
                    )

                # sanitize numerics for safety
                loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)

                token_this = attn_mask.sum().item()

            # 5. backward + step + sched
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if projector is not None:
                accelerator.clip_grad_norm_(projector.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # 6. checkpoint during training
            if args.save_every > 0 and step > 0 and (step % args.save_every == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_ckpt(
                        step, model, tok, optimizer, scheduler, save_dir,
                        dataset=dataset, projector=projector
                    )
                    print(f"[ckpt] Saved {save_dir}/ckpt_step{step:07d}")
                accelerator.wait_for_everyone()

            # 7. logging / step book-keeping
            total_tokens += token_this
            step += 1
            if accelerator.is_main_process and step % 10 == 0:
                dt = time.time() - t0
                tps = total_tokens / max(dt, 1e-6)
                print(f"[step {step}] loss={loss.item():.4f} tokens={int(total_tokens)} tok/s={tps:.1f}")

        if args.max_steps > 0 and step >= args.max_steps:
            break

    accelerator.wait_for_everyone()
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
    accelerator.wait_for_everyone()

if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
