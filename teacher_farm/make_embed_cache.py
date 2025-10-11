#!/usr/bin/env python3
import argparse, os, json, math, gzip, time, sys, pathlib
from typing import Iterable, Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm


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
    # x: [*, d]
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

def existing_shards(out_dir: str, stem: str) -> List[str]:
    p = pathlib.Path(out_dir)
    if not p.exists():
        return []
    return sorted(str(x) for x in p.glob(f"{stem}_*.parquet"))

def next_shard_index(out_dir: str, stem: str) -> int:
    # Find next index based on existing files
    idx = -1
    for fp in existing_shards(out_dir, stem):
        base = os.path.basename(fp)
        try:
            n = int(base.split("_")[-1].split(".")[0])
            idx = max(idx, n)
        except Exception:
            pass
    return idx + 1


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto", help="e.g., 'auto', 'cuda:0', 'balanced_low_0'")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls_last", "last_token"],
                    help="How to pool hidden states into a single vector")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize pooled embeddings")
    ap.add_argument("--shard_size", type=int, default=4096, help="Rows per Parquet shard")
    ap.add_argument("--stem", default="relb_embeds", help="Output filename stem")
    ap.add_argument("--resume", action="store_true", help="Resume by appending after last shard index")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.padding_side != "left":
        tok.padding_side = "right" 

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )
    model.eval()

    # Stable Output schema 
    schema = pa.schema([
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("attn_mask", pa.list_(pa.int8())),
        pa.field("pooled_embedding", pa.list_(pa.float32())),
        pa.field("length", pa.int32()),
    ])

    # Resume Shard index 
    shard_idx = next_shard_index(args.out_dir, args.stem) if args.resume else 0

    # Buffers
    rows: List[Dict[str, Any]] = []
    total_rows = 0
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

    # Streaming loop
    with torch.no_grad():
        batch: List[str] = []
        pbar = tqdm(desc="Embedding", unit="rows")

        for text in iter_json_texts(args.input_jsonl):
            batch.append(text)
            if len(batch) < args.batch_size:
                continue

            enc = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}

            out = model(**enc, output_hidden_states=True, use_cache=False)
            last_hidden: torch.Tensor = out.hidden_states[-1]  # [B, T, d]
            attn: torch.Tensor = enc["attention_mask"]         # [B, T]
            ids: torch.Tensor = enc["input_ids"]               # [B, T]

            # Pooling
            if args.pooling == "mean":
                mask = attn.unsqueeze(-1)                      # [B, T, 1]
                summed = (last_hidden * mask).sum(dim=1)       # [B, d]
                counts = mask.sum(dim=1).clamp_min(1)          # [B, 1]
                pooled = summed / counts
            elif args.pooling == "cls_last":
                # First token representation (often BOS/CLS in instruct models)
                pooled = last_hidden[:, 0, :]
            else:  # last_token
                # index last non-pad token per row
                lengths = attn.sum(dim=1).to(torch.long) - 1   # [B]
                pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), lengths, :]

            if args.normalize:
                pooled = l2_normalize(pooled)

            pooled_cpu = pooled.to(torch.float32).cpu()         # storing as float32 for downstream safety
            attn_cpu = attn.cpu()
            ids_cpu = ids.cpu()

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
            batch = []

            if len(rows) >= args.shard_size:
                flush_rows()

        # tail
        if batch:
            enc = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            last_hidden = out.hidden_states[-1]
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
            B = pooled_cpu.size(0)
            for b in range(B):
                L = int(attn_cpu[b].sum().item())
                rows.append({
                    "input_ids": ids_cpu[b, :L].to(torch.int32).tolist(),
                    "attn_mask": attn_cpu[b, :L].to(torch.int8).tolist(),
                    "pooled_embedding": pooled_cpu[b].tolist(),
                    "length": L,
                })

        flush_rows()
        pbar.close()

    dt = time.time() - t0
    rps = 0.0 if dt <= 0 else total_rows / dt
    print(f"[DONE] rows={total_rows} time={dt:.1f}s rate={rps:.1f} rows/s -> {args.out_dir}")


if __name__ == "__main__":
    main()
