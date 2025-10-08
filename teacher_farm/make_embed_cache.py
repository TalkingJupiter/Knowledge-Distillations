import argparse, os, json, math
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--max_length', type=int, default=8192)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()

    texts = [json.loads(l)['text'] for l in open(args.input_jsonl) if l.strip()]
    shard_size = 128
    rows, shard_idx = [], 0

    with torch.no_grad():
        for batch in tqdm(batched(texts, args.batch_size), total=math.ceil(len(texts)/args.batch_size)):
            enc = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)

            last_hidden = out.hidden_states[-1]  # [B, T, d]
            attn = enc['attention_mask']
            input_ids = enc['input_ids']

            mask = attn.unsqueeze(-1)  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)  # [B, d]
            counts = mask.sum(dim=1).clamp_min(1)     # [B, 1]
            pooled = (summed / counts).cpu()          # [B, d]

            for b in range(pooled.size(0)):
                L = int(attn[b].sum().item())
                rows.append({
                    'input_ids': input_ids[b, :L].cpu().tolist(),
                    'attn_mask': attn[b, :L].cpu().tolist(),
                    'pooled_embedding': pooled[b].tolist()
                })

            if len(rows) >= shard_size:
                table = pa.Table.from_pylist(rows)
                out_path = os.path.join(args.out_dir, f'relb_embeds_{shard_idx:06d}.parquet')
                pq.write_table(table, out_path, compression='zstd')
                print('Wrote', out_path)
                rows, shard_idx = [], shard_idx + 1

    if rows:
        table = pa.Table.from_pylist(rows)
        out_path = os.path.join(args.out_dir, f'relb_embeds_{shard_idx:06d}.parquet')
        pq.write_table(table, out_path, compression='zstd')
        print('Wrote', out_path)

if __name__ == '__main__':
    main()


    