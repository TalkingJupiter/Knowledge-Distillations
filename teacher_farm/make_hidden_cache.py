#!/usr/bin/env python3
import argparse, os, json, gzip, gc, pathlib
from typing import List, Dict, Any, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------
# Layer resolution
# ----------------------------------------------------------------------------------------------------------------
def resolve_block_list(model) -> List[torch.nn.Module]:
    """
    Try common HF architectures to get the transformer block list.
    """
    # LLaMA / Mistral / Qwen
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # OPT
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    # MPT
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks)
    # GPT-J / GPT-NeoX variants may have .layers or .h
    for attr in ("layers", "h", "blocks"):
        if hasattr(model, attr) and isinstance(getattr(model, attr), (list, tuple)):
            return list(getattr(model, attr))
    raise RuntimeError("Could not resolve transformer block list for this model.")


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)

    ap.add_argument('--layers', type=int, nargs='+', required=True,
                    help='Teacher block indices to capture (e.g., 22 30).')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_length', type=int, default=2048)

    ap.add_argument('--dtype', default='bfloat16',
                    choices=['bfloat16','bf16','float16','fp16','float32','fp32'])
    ap.add_argument('--device_map', default='auto',
                    help="HF device map (e.g., 'auto', 'cuda:0', 'balanced_low_0').")

    ap.add_argument('--flush_every', type=int, default=256,
                    help='Write to parquet every N samples.')
    ap.add_argument('--stem', default='fb_hints',
                    help='Output file stem (files like <stem>_000000.parquet).')
    ap.add_argument('--resume', action='store_true',
                    help='Resume at next shard index if files exist.')

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
    # Right padding is generally fine for causal pooling/export
    tok.padding_side = "right"

    # --- load teacher (multi-GPU friendly)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    blocks = resolve_block_list(model)
    n_blocks = len(blocks)

    # normalize and validate layer indices (allow negatives like -1)
    target_layers: List[int] = []
    for li in args.layers:
        if li < 0:
            li = n_blocks + li
        if not (0 <= li < n_blocks):
            raise ValueError(f"Layer index {li} out of range [0,{n_blocks-1}]")
        target_layers.append(li)
    target_layers = sorted(set(target_layers))

    # --- forward hooks for requested layers
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def hook(_m, _inp, out):
            # out: [B, T, H] on some device; move to CPU immediately to free GPU mem
            captured[li] = out.detach().to('cpu')
        return hook

    handles = []
    for li in target_layers:
        handles.append(blocks[li].register_forward_hook(make_hook(li)))

    # --- parquet schema (dynamic based on selected layers)
    fields = [
        pa.field('input_ids', pa.list_(pa.int32())),
        pa.field('attn_mask', pa.list_(pa.int8())),
    ]
    for li in target_layers:
        # Nested list for [L, H] as list(list(float32))
        fields.append(pa.field(f'hidden_L{li}', pa.list_(pa.list_(pa.float32()))))
    schema = pa.schema(fields)

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

    # --- streaming
    with torch.inference_mode():
        pbar = tqdm(desc="FB cache", unit="rows")
        for batch in batched(iter_json_texts(args.input_jsonl), args.batch_size):
            # Tokenize on CPU; HF dispatch handles device placement
            enc = tok(batch, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors='pt')

            captured.clear()
            _ = model(**enc, output_hidden_states=False, use_cache=False, return_dict=True)

            input_ids = enc['input_ids'].cpu()
            attn_mask = enc['attention_mask'].cpu()

            B = input_ids.size(0)
            for b in range(B):
                L = int(attn_mask[b].sum().item())
                row: Dict[str, Any] = {
                    'input_ids': input_ids[b, :L].to(torch.int32).tolist(),
                    'attn_mask': attn_mask[b, :L].to(torch.int8).tolist(),
                }
                for li in target_layers:
                    ht = captured[li][b, :L, :]               # (L, H) on CPU
                    row[f'hidden_L{li}'] = ht.to(torch.float32).tolist()
                rows.append(row)

            del enc, input_ids, attn_mask
            captured.clear()

            # free GPU mem from compute graphs
            torch.cuda.empty_cache()
            pbar.update(B)

            if len(rows) >= args.flush_every:
                flush_rows()

        # tail flush
        flush_rows()
        pbar.close()

    # remove hooks
    for h in handles:
        h.remove()

    print("[DONE] feature-cache build complete ->", args.out_dir)


if __name__ == '__main__':
    main()
