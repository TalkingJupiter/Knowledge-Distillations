#!/usr/bin/env python3
import argparse, os, json, math, gzip, time, pathlib, warnings
from typing import Iterable, Dict, Any, List, Optional, Iterator

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging
import pyarrow as pa, pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile
from tqdm import tqdm
from contextlib import nullcontext

# Optional: quieter logs & ignore noisy FutureWarnings
hf_logging.set_verbosity_warning()
warnings.simplefilter("ignore", FutureWarning)

# Optional (nice on A100/H100): allow TF32 for faster matmul without hurting inference
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------------------------------------------
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

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

def existing_shards(out_dir: str, stem: str) -> List[str]:
    p = pathlib.Path(out_dir)
    if not p.exists():
        return []
    return sorted(str(x) for x in p.glob(f"{stem}_*.parquet"))

def next_shard_index(out_dir: str, stem: str) -> int:
    idx = -1
    for fp in existing_shards(out_dir, stem):
        base = os.path.basename(fp)
        try:
            n = int(base.split("_")[-1].split(".")[0])
            idx = max(idx, n)
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
    # burn n items, then yield the rest
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

# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=1)  # safer default for VRAM
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto", help="e.g., 'auto', 'cuda:0', 'balanced_low_0'")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls_last", "last_token"],
                    help="How to pool hidden states into a single vector")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize pooled embeddings")
    ap.add_argument("--shard_size", type=int, default=4096, help="Rows per Parquet shard")
    ap.add_argument("--stem", default="relb_embeds", help="Output filename stem")
    # Heartbeat checkpointing
    ap.add_argument("--ckpt_every", type=int, default=512,
                    help="Write a checkpoint every N processed rows (in addition to shard flushes).")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Checkpoint plumbing
    ckpt_path = os.path.join(args.out_dir, f"{args.stem}.ckpt.json")
    input_fp  = file_fingerprint(args.input_jsonl)

    # Auto-detect resume position & shard index
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
        print(f"[RESUME] No valid checkpoint. Detected {line_start} rows across {shard_idx} shards; "
              f"skipping that many input lines.")
    else:
        shard_idx = 0
        line_start = 0
        print("[START] No shards or checkpoint found; starting from the beginning.")

    # Startup checkpoint so you always have one (even pre-flush)
    save_ckpt(ckpt_path, {
        "input_fingerprint": input_fp,
        "line_index": line_start,
        "shard_idx": shard_idx,
        "out_dir": os.path.abspath(args.out_dir),
        "stem": args.stem,
        "pooling": args.pooling,
        "normalize": args.normalize,
        "model": args.model,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    })

    # Dtype + AMP selection
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    amp_dtype = None
    if torch.cuda.is_available():
        if args.dtype == "bfloat16":
            amp_dtype = torch.bfloat16
        elif args.dtype == "float16":
            amp_dtype = torch.float16
    autocast_ctx = (torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype is not None else nullcontext())

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.padding_side != "right":
        tok.padding_side = "right"

    # Model (let HF/Accelerate place across GPUs if device_map is auto/balanced)
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch_dtype,                   # (was torch_dtype=...)
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Keep inputs on CPU if HF dispatcher will scatter; otherwise move to model device
    def place_inputs(enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dm = str(args.device_map)
        if dm in ("auto", "balanced", "balanced_low_0") or "auto" in dm or "balanced" in dm:
            return enc
        dev = next(model.parameters()).device
        return {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

    # Stable Output schema
    schema = pa.schema([
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("attn_mask", pa.list_(pa.int8())),
        pa.field("pooled_embedding", pa.list_(pa.float32())),
        pa.field("length", pa.int32()),
    ])

    # Buffers and counters
    rows: List[Dict[str, Any]] = []
    total_rows = 0
    processed_lines = 0  # exact number of input lines consumed
    t0 = time.time()

    def flush_rows():
        nonlocal rows, shard_idx, total_rows
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        out_path = os.path.join(args.out_dir, f"{args.stem}_{shard_idx:06d}.parquet")
        pq.write_table(table, out_path, compression="zstd")
        print(f"[WRITE] {out_path} ({len(rows)} rows)")
        total_rows += len(rows)
        rows = []
        shard_idx += 1

        # Write/update checkpoint so future resumes are exact
        save_ckpt(ckpt_path, {
            "input_fingerprint": input_fp,
            "line_index": processed_lines,  # exact lines consumed from input
            "shard_idx": shard_idx,         # next shard to write
            "out_dir": os.path.abspath(args.out_dir),
            "stem": args.stem,
            "pooling": args.pooling,
            "normalize": args.normalize,
            "model": args.model,
            "dtype": args.dtype,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
        })

    # Streaming loop
    with torch.no_grad():
        batch: List[str] = []
        pbar = tqdm(desc="Embedding", unit="rows")

        text_iter = iter_json_texts(args.input_jsonl)
        if line_start > 0:
            text_iter = skip_iter(iter(text_iter), line_start)
            processed_lines = line_start

        for text in text_iter:
            batch.append(text)
            if len(batch) < args.batch_size:
                continue

            enc = tok(batch, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt")
            enc = place_inputs(enc)

            with autocast_ctx:
                out = model(**enc, output_hidden_states=False, use_cache=False)
            last_hidden: torch.Tensor = out.last_hidden_state  # [B, T, d]

            attn: torch.Tensor = enc["attention_mask"]         # [B, T]
            ids: torch.Tensor = enc["input_ids"]               # [B, T]

            # Pooling
            if args.pooling == "mean":
                mask = attn.unsqueeze(-1)                      # [B, T, 1]
                summed = (last_hidden * mask).sum(dim=1)       # [B, d]
                counts = mask.sum(dim=1).clamp_min(1)          # [B, 1]
                pooled = summed / counts
            elif args.pooling == "cls_last":
                pooled = last_hidden[:, 0, :]
            else:  # last_token
                lengths = attn.sum(dim=1).to(torch.long) - 1   # [B]
                pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), lengths, :]

            if args.normalize:
                pooled = l2_normalize(pooled)

            # Move to CPU immediately and free GPU tensors
            pooled_cpu = pooled.to(torch.float32).cpu()
            attn_cpu = attn.cpu()
            ids_cpu = ids.cpu()

            del out, last_hidden, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            B = pooled_cpu.size(0)
            for b in range(B):
                L = int(attn_cpu[b].sum().item())
                rows.append({
                    "input_ids": ids_cpu[b, :L].to(torch.int32).tolist(),
                    "attn_mask": attn_cpu[b, :L].to(torch.int8).tolist(),
                    "pooled_embedding": pooled_cpu[b].tolist(),
                    "length": L,
                })

            pbar.update(B)
            processed_lines += B
            batch = []

            # heartbeat checkpoint (even if no shard flushed yet)
            if args.ckpt_every > 0 and (processed_lines % args.ckpt_every) < B:
                save_ckpt(ckpt_path, {
                    "input_fingerprint": input_fp,
                    "line_index": processed_lines,
                    "shard_idx": shard_idx,  # current next shard index
                    "out_dir": os.path.abspath(args.out_dir),
                    "stem": args.stem,
                    "pooling": args.pooling,
                    "normalize": args.normalize,
                    "model": args.model,
                    "dtype": args.dtype,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                })

            if len(rows) >= args.shard_size:
                flush_rows()

        # tail
        if batch:
            enc = tok(batch, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt")
            enc = place_inputs(enc)

            with autocast_ctx:
                out = model(**enc, output_hidden_states=False, use_cache=False)
            last_hidden = out.last_hidden_state
            attn = enc["attention_mask"]
            ids = enc["input_ids"]

            if args.pooling == "mean":
                mask = attn.unsqueeze(-1)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp_min(1)
                pooled = summed / counts
            elif args.pooling == "cls_last":
                pooled = last_hidden[:, 0, :]
            else:
                lengths = attn.sum(dim=1).to(torch.long) - 1
                pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), lengths, :]

            if args.normalize:
                pooled = l2_normalize(pooled)

            pooled_cpu = pooled.to(torch.float32).cpu()
            attn_cpu = attn.cpu()
            ids_cpu = ids.cpu()

            del out, last_hidden, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            B = pooled_cpu.size(0)
            for b in range(B):
                L = int(attn_cpu[b].sum().item())
                rows.append({
                    "input_ids": ids_cpu[b, :L].to(torch.int32).tolist(),
                    "attn_mask": attn_cpu[b, :L].to(torch.int8).tolist(),
                    "pooled_embedding": pooled_cpu[b].tolist(),
                    "length": L,
                })
            processed_lines += B

        flush_rows()
        pbar.close()

    dt = time.time() - t0
    rps = 0.0 if dt <= 0 else total_rows / dt
    print(f"[DONE] rows={total_rows} time={dt:.1f}s rate={rps:.1f} rows/s -> {args.out_dir}")

if __name__ == "__main__":
    main()
