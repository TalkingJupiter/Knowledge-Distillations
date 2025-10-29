#!/usr/bin/env python3
import argparse, os, json, gzip, gc, pathlib, math, warnings, time
from typing import Iterable, Dict, Any, List, Iterator, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
import pyarrow as pa, pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile
from tqdm import tqdm
from contextlib import nullcontext

# Quieter logs / warnings (optional)
hf_logging.set_verbosity_warning()
warnings.simplefilter("ignore", FutureWarning)

# Faster matmul on A100/H100 without hurting inference
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------------------
def open_jsonl_any(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")

def iter_json_texts(path: str) -> Iterable[str]:
    with open_jsonl_any(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                yield t

def batched(iterable: Iterable[Any], n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

# ---------------------------------------------------------------------------------------
# Resume / checkpoint helpers
# ---------------------------------------------------------------------------------------
def existing_shards(out_dir: str, stem: str) -> List[str]:
    p = pathlib.Path(out_dir)
    if not p.exists():
        return []
    return sorted(str(fp) for fp in p.glob(f"{stem}_*.parquet"))

def next_shard_index(out_dir: str, stem: str) -> int:
    idx = -1
    for fp in existing_shards(out_dir, stem):
        try:
            i = int(pathlib.Path(fp).stem.split("_")[-1])
            idx = max(idx, i)
        except Exception:
            pass
    return idx + 1

def count_rows_in_existing_shards(out_dir: str, stem: str) -> int:
    total = 0
    for fp in existing_shards(out_dir, stem):
        try:
            pf = ParquetFile(fp)
            total += pf.metadata.num_rows
        except Exception:
            pass
    return total

def skip_iter(it: Iterator[str], n: int) -> Iterator[str]:
    for _ in range(n):
        try:
            next(it)
        except StopIteration:
            return
    for x in it:
        yield x

def file_fingerprint(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": st.st_size, "mtime": int(st.st_mtime)}

def load_ckpt(ckpt_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(ckpt_path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def save_ckpt(ckpt_path: str, data: Dict[str, Any]) -> None:
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, ckpt_path)

# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input_jsonl', required=True, help='Each line {"text": ...} (.jsonl or .jsonl.gz)')
    ap.add_argument('--out_dir', required=True)

    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--batch_size', type=int, default=1, help='Keep small for very long sequences')
    ap.add_argument('--max_length', type=int, default=8192)

    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--device_map', default='auto', help="e.g. 'auto', 'cuda:0', 'balanced_low_0'")

    ap.add_argument('--shard_size', type=int, default=128)
    ap.add_argument('--stem', default='rb_topk')

    # Heartbeat checkpointing (in addition to shard flushes)
    ap.add_argument('--ckpt_every', type=int, default=512,
                    help='Write a checkpoint every N processed rows (even before first shard).')

    # Absolute cap: stop when total processed lines across all runs reaches this number (0 = unlimited)
    ap.add_argument('--max_lines', type=int, default=0,
                    help='Absolute maximum total lines to process overall (0 = no cap).')

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()

    # dtype + AMP
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    torch_dtype = dtype_map[args.dtype]
    amp_dtype = None
    if torch.cuda.is_available():
        if args.dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
        elif args.dtype == 'float16':
            amp_dtype = torch.float16
    autocast_ctx = (torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype is not None else nullcontext())

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # checkpoint plumbing (auto-resume)
    ckpt_path = os.path.join(args.out_dir, f"{args.stem}.ckpt.json")
    input_fp  = file_fingerprint(args.input_jsonl)

    shard_idx = 0
    line_start = 0
    ckpt = load_ckpt(ckpt_path)
    shards_exist = len(existing_shards(args.out_dir, args.stem)) > 0

    if ckpt and ckpt.get("input_fingerprint") == input_fp:
        line_start = int(ckpt.get("line_index", 0))
        shard_idx  = int(ckpt.get("shard_idx", 0))
        print(f"[RESUME] Using checkpoint: line={line_start}, next_shard={shard_idx}")
    elif shards_exist:
        shard_idx  = next_shard_index(args.out_dir, args.stem)
        line_start = count_rows_in_existing_shards(args.out_dir, args.stem)
        print(f"[RESUME] No valid checkpoint. Detected {line_start} rows across {shard_idx} shards; skipping that many input lines.")
    else:
        print("[START] No shards or checkpoint found; starting from the beginning.")

    # absolute cap target
    lines_target = None
    if args.max_lines and args.max_lines > 0:
        lines_target = args.max_lines
        if line_start >= lines_target:
            print(f"[EXIT] Already processed {line_start} >= max_lines={args.max_lines}. Nothing to do.")
            return

    # startup checkpoint so you always have one
    save_ckpt(ckpt_path, {
        "input_fingerprint": input_fp,
        "line_index": line_start,
        "shard_idx": shard_idx,
        "out_dir": os.path.abspath(args.out_dir),
        "stem": args.stem,
        "k": args.k,
        "model": args.model,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    })

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,          # use dtype= (torch_dtype is deprecated)
        device_map=args.device_map,
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # parquet schema (stable)
    schema = pa.schema([
        pa.field('input_ids', pa.list_(pa.int32())),
        pa.field('attn_mask', pa.list_(pa.int8())),
        pa.field('topk_ids', pa.list_(pa.list_(pa.int32()))),        # [T-1, k]
        pa.field('topk_logprobs', pa.list_(pa.list_(pa.float32()))), # [T-1, k]
    ])

    rows: List[Dict[str, Any]] = []
    processed_lines = 0

    def flush_rows():
        nonlocal rows, shard_idx
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        out_path = os.path.join(args.out_dir, f'{args.stem}_{shard_idx:06d}.parquet')
        pq.write_table(table, out_path, compression='zstd')
        print(f'[WRITE] {out_path} ({len(rows)} samples)')
        shard_idx += 1
        rows.clear()
        gc.collect()
        # checkpoint after each shard
        save_ckpt(ckpt_path, {
            "input_fingerprint": input_fp,
            "line_index": processed_lines,
            "shard_idx": shard_idx,
            "out_dir": os.path.abspath(args.out_dir),
            "stem": args.stem,
            "k": args.k,
            "model": args.model,
            "dtype": args.dtype,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
        })

    with torch.inference_mode():
        # tqdm total = remaining budget if capped; else indeterminate
        if args.max_lines and args.max_lines > 0:
            pbar_total = max(0, args.max_lines - line_start)
        else:
            pbar_total = None
        pbar = tqdm(total=pbar_total, desc="RB top-k", unit="rows")

        text_iter = iter_json_texts(args.input_jsonl)
        if line_start > 0:
            text_iter = skip_iter(iter(text_iter), line_start)
            processed_lines = line_start

        for batch_texts in batched(text_iter, args.batch_size):
            # Respect absolute cap: trim batch if we're about to overshoot
            if lines_target is not None:
                remaining = lines_target - processed_lines
                if remaining <= 0:
                    break
                if len(batch_texts) > remaining:
                    batch_texts = batch_texts[:remaining]

            # tokenize on CPU
            enc = tok(batch_texts, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors='pt')

            # forward (AMP if available)
            with autocast_ctx:
                out = model(**enc, use_cache=False, return_dict=True)
                logits = out.logits  # [B, T, V]
                logits_next = logits[:, :-1, :]  # [B, T-1, V]

                # log-softmax for proper normalized log-probs
                logprobs_all = F.log_softmax(logits_next, dim=-1)  # [B, T-1, V]
                # top-k by logits; indices reused to gather logprobs
                _, topk_idx = torch.topk(logits_next, k=args.k, dim=-1)
                topk_logprobs = torch.gather(logprobs_all, dim=-1, index=topk_idx)  # [B, T-1, k]

            input_ids = enc['input_ids'].cpu()
            attn_mask = enc['attention_mask'].cpu()
            topk_idx = topk_idx.cpu().to(torch.int32)
            topk_logprobs = topk_logprobs.cpu().to(torch.float32)

            B = input_ids.size(0)
            for b in range(B):
                L = int(attn_mask[b].sum().item())
                eff = max(L - 1, 0)  # effective length for next-token preds
                rows.append({
                    'input_ids': input_ids[b, :L].to(torch.int32).tolist(),
                    'attn_mask': attn_mask[b, :L].to(torch.int8).tolist(),
                    'topk_ids': topk_idx[b, :eff, :].tolist(),
                    'topk_logprobs': topk_logprobs[b, :eff, :].tolist(),
                })

            del enc, out, logits, logits_next, logprobs_all, topk_idx, topk_logprobs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.update(B)
            processed_lines += B

            # heartbeat checkpoint (even if no shard flushed yet)
            if args.ckpt_every > 0 and (processed_lines % args.ckpt_every) < B:
                save_ckpt(ckpt_path, {
                    "input_fingerprint": input_fp,
                    "line_index": processed_lines,
                    "shard_idx": shard_idx,  # current next shard index
                    "out_dir": os.path.abspath(args.out_dir),
                    "stem": args.stem,
                    "k": args.k,
                    "model": args.model,
                    "dtype": args.dtype,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                })

            if len(rows) >= args.shard_size:
                flush_rows()

            if lines_target is not None and processed_lines >= lines_target:
                break

        flush_rows()
        pbar.close()

    dt = time.time() - t0

    if args.max_lines and args.max_lines > 0 and processed_lines < args.max_lines:
        print(f"[WARN] Dataset ended early: processed {processed_lines} < max_lines={args.max_lines}")

    print(f"[DONE] RB top-k cache build in {dt:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
