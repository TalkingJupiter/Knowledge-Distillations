import argparse, os, json, math
import torch
import torch.nn.functional as F
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input_jsonl', required=True, help='Each line {\"text\": ...}')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--batch_size', type=int, default=1, help='1 recommended for very long sequences')
    ap.add_argument('--max_length', type=int, default=8192)
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype_map = {'bfloat': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading teacher model {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map='auto')
    model.eval()

    texts = [json.loads(l)['text'] for l in open(args.input_jsonl) if l.strip()]
    shard_size = 128

    rows = []
    shard_idx = 0

    with torch.no_grad():
        for batch in tqdm(batched(texts, args.batch_size), total=math.ceil(len(texts)/args.batch_size)):
            enc = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, use_cache=False)
            logits = out.logits ### [B,T,V]
            topk_vals, topk_idx = torch.topk(logits[:, :-1, :], k=args.k, dim=-1)  ## [B, T-1, k]

            input_ids = enc['input_ids'].cpu()
            attn_mask = enc['attention_mask'].cpu()
            logprobs = torch.log_softmax(topk_vals, dim=-1).cpu()

            for b in range(input_ids.size(0)):
                L = int(attn_mask[b].sum().item())
                eff = max(L-1, 0)
                rows.append({
                    'input_ids': input_ids[b, :L].tolist(),
                    'attn_mask': attn_mask[b, :L].tolist(),
                    'topk_ids': topk_idx[b, :eff, :].cpu().tolist(),
                    'topk_logprobs': logprobs[b, :eff, :].tolist()
                })

            if len(rows) >= shard_size:
                table = pa.Table.from_pylist(rows)
                out_path = os.path.join(args.out_dir, f'rb_topk_{shard_idx:06d}.parquet')
                pq.write_table(table, out_path, compression='zstd')
                print('Wrote', out_path)
                rows, shard_idx = [], shard_idx+1

    if rows:
        table = pa.Table.from_pylist(rows)
        out_path = os.path.join(args.out_dir, f'rb_topk_{shard_idx:06d}.parquet')
        pq.write_table(table, out_path, compression='zstd')
        print('Wrote', out_path)

if __name__ == "__main__":
    main()