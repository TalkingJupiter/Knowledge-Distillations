#!/usr/bin/env python3
import argparse, os, json, math, gc
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--layers', type=int, nargs='+', required=True,
                    help='Teacher layers to save, e.g. 22 30 (Llama blocks index)')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_length', type=int, default=2048)
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16','bf16','float16','fp16','float32','fp32'])
    ap.add_argument('--flush_every', type=int, default=256,
                    help='Write to parquet every N samples to avoid high RAM')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- dtype parsing
    dmap = {
        'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
        'float16': torch.float16,   'fp16': torch.float16,
        'float32': torch.float32,   'fp32': torch.float32,
    }
    torch_dtype = dmap[args.dtype]

    # --- tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- load teacher across 4 GPUs
    # Keep inputs on CPU; HF will dispatch internally with device_map='auto'
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map='auto',
    )
    model.eval()
    # Avoid KV-cache to save memory
    if hasattr(model, "config"):
        model.config.use_cache = False

    # --- forward hooks for ONLY requested layers
    target_layers: List[int] = sorted(set(args.layers))
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook(_m, _inp, out):
            # out: (B,T,H) on some CUDA device; move to CPU immediately
            captured[li] = out.detach().to('cpu')
        return hook

    # Llama-style: model.model.layers[li]
    handles = []
    for li in target_layers:
        layer_mod = model.model.layers[li]
        handles.append(layer_mod.register_forward_hook(make_hook(li)))

    # --- streaming read
    texts = (rec['text'] for rec in read_jsonl(args.input_jsonl))
    shard_idx = 0
    rows: List[Dict[str, Any]] = []

    def flush_rows():
        nonlocal rows, shard_idx
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        out_path = os.path.join(args.out_dir, f'fb_hints_{shard_idx:06d}.parquet')
        pq.write_table(table, out_path, compression='zstd')
        print('Wrote', out_path, f'({len(rows)} samples)')
        shard_idx += 1
        rows = []
        gc.collect()

    with torch.inference_mode():
        for batch in tqdm(batched(texts, args.batch_size)):
            # Tokenize on CPU; keep tensors on CPU (device_map dispatch handles placement)
            enc = tok(list(batch), padding=True, truncation=True,
                      max_length=args.max_length, return_tensors='pt')
            # Run forward WITHOUT output_hidden_states big tuple; hooks will capture our layers
            captured.clear()
            _ = model(**enc, output_hidden_states=False, use_cache=False, return_dict=True)

            input_ids = enc['input_ids'].cpu()
            attn_mask = enc['attention_mask'].cpu()

            # Build rows from captured CPU tensors
            for b in range(input_ids.size(0)):
                L = int(attn_mask[b].sum().item())
                row = {
                    'input_ids': input_ids[b, :L].tolist(),
                    'attn_mask': attn_mask[b, :L].tolist(),
                }
                for li in target_layers:
                    ht = captured[li][b, :L, :]  # (L, H) on CPU
                    # NOTE: Parquet likes lists; we keep float16/bfloat16 as float32 to be safe
                    row[f'hidden_L{li}'] = ht.to(torch.float32).tolist()
                rows.append(row)

            del enc, input_ids, attn_mask
            captured.clear()
            torch.cuda.empty_cache()

            if len(rows) >= args.flush_every:
                flush_rows()

    # final flush
    flush_rows()

    # remove hooks
    for h in handles:
        h.remove()

if __name__ == '__main__':
    main()
