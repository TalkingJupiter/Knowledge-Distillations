#!/usr/bin/env python3
import argparse, os, json, gzip, gc, pathlib, math
from typing import Iterable, Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm


# ------------------------------------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------------------------------------
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

def next_shard_index(out_dir: str, stem: str) -> int:
    p = pathlib.Path(out_dir)
    if not p.exists():
        return 0
    idx = -1
    for fp in p.glob(f"{stem}_*.parquet"):
        try:
            i = int(fp.stem.split("_")[-1])
            idx = max(idx, i)
        except Exception:
            pass
    return idx + 1


# ------------------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------------------
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
    ap.add_argument('--resume', action='store_true', help='Append starting from the next shard index')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # dtype
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16':  torch.float16,
        'float32':  torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # model
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
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

    shard_idx = next_shard_index(args.out_dir, args.stem) if args.resume else 0
    rows: List[Dict[str, Any]] = []

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

    with torch.inference_mode():
        pbar = tqdm(desc="RB top-k", unit="rows")
        for batch_texts in batched(iter_json_texts(args.input_jsonl), args.batch_size):
            # tokenize on CPU
            enc = tok(batch_texts, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors='pt')

            # forward
            out = model(**enc, use_cache=False, return_dict=True)
            logits = out.logits  # [B, T, V]
            # next-token prediction: shift to exclude last position as there is no label
            logits_next = logits[:, :-1, :]  # [B, T-1, V]

            # compute log_softmax over vocab, then top-k indices + gather their logprobs
            # NOTE: computing log_softmax across V is the correct probabilistic normalization.
            logprobs_all = F.log_softmax(logits_next, dim=-1)  # [B, T-1, V]
            topk_vals, topk_idx = torch.topk(logits_next, k=args.k, dim=-1)  # indices from logits (argmax over V)
            # gather logprobs at those indices
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

            del enc, out, logits, logits_next, logprobs_all, topk_vals
            torch.cuda.empty_cache()
            pbar.update(B)

            if len(rows) >= args.shard_size:
                flush_rows()

        flush_rows()
        pbar.close()

    print("[DONE] RB top-k cache build ->", args.out_dir)


if __name__ == "__main__":
    main()
