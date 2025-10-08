#!/usr/bin/env python3
import argparse, json, random, sys, os
from datasets import load_dataset, Dataset, concatenate_datasets

# --- Heuristics to extract text from various dataset schemas ---
def record_to_text(rec):
    # 1) Raw "text"
    if "text" in rec and isinstance(rec["text"], str) and rec["text"].strip():
        return rec["text"].strip()

    # 2) Instruction-style pairs
    for a,b in [("instruction","output"), ("prompt","response"), ("input","output")]:
        if a in rec and b in rec and isinstance(rec[a], str) and isinstance(rec[b], str):
            s = rec[a].strip()
            r = rec[b].strip()
            if s or r:
                return f"### Instruction:\n{s}\n\n### Response:\n{r}"

    # 3) Chat "messages": list of {role, content}
    if "messages" in rec and isinstance(rec["messages"], (list, tuple)):
        msgs = []
        for m in rec["messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Sometimes content is list of segments (OpenAI-style tools); flatten strings
                content = "\n".join([seg.get("text","") if isinstance(seg, dict) else str(seg) for seg in content])
            msgs.append(f"{role}: {content}".strip())
        text = "\n".join(msgs).strip()
        if text:
            return text

    # 4) Fallback: join all str-like fields
    parts = []
    for k,v in rec.items():
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")
    return "\n".join(parts).strip() if parts else None


def load_one(name, split, data_dir=None, streaming=True):
    kwargs = {}
    if data_dir: kwargs["data_dir"] = data_dir
    ds = load_dataset(name, split=split, streaming=streaming, **kwargs)
    return ds


def take_n(streaming_ds, n):
    # Turn streaming iterator into limited list of dicts
    out = []
    for i, rec in enumerate(streaming_ds):
        if n is not None and i >= n: break
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True,
                    help="HF dataset name (repeatable). Example: teknium/OpenHermes-2.5")
    ap.add_argument("--split", default="train", help="Split to load (default: train)")
    ap.add_argument("--weights", type=str, default="",
                    help="Comma-separated weights matching --dataset count, e.g. '0.8,0.2'")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap total samples across all datasets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/shards.jsonl")
    ap.add_argument("--streaming", action="store_true", default=True)
    ap.add_argument("--no-streaming", dest="streaming", action="store_false")
    ap.add_argument("--data_dir", default=None,
                    help="If HF offline, point to local data dir/cache")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Parse weights (optional)
    if args.weights:
        ws = [float(x) for x in args.weights.split(",")]
        if len(ws) != len(args.dataset):
            print("[ERROR] weights length must match number of --dataset entries", file=sys.stderr)
            sys.exit(2)
        total_w = sum(ws)
        ws = [w/total_w for w in ws]
    else:
        ws = [1.0/len(args.dataset)] * len(args.dataset)

    # For streaming: round-robin proportional sampling
    per_ds_quota = None
    if args.max_samples is not None:
        per_ds_quota = [max(1, int(args.max_samples * w)) for w in ws]
        # adjust rounding drift
        deficit = args.max_samples - sum(per_ds_quota)
        for i in range(abs(deficit)):
            per_ds_quota[i % len(per_ds_quota)] += 1 if deficit > 0 else -1

    # Load and write
    out_count = 0
    writers = open(args.out, "w", encoding="utf-8")

    try:
        if args.streaming:
            # streaming: iterate each dataset independently to its quota
            for i, name in enumerate(args.dataset):
                quota = None if per_ds_quota is None else per_ds_quota[i]
                ds = load_one(name, args.split, data_dir=args.data_dir, streaming=True)
                c = 0
                for rec in ds:
                    txt = record_to_text(rec)
                    if not txt: continue
                    writers.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                    out_count += 1
                    c += 1
                    if quota is not None and c >= quota:
                        break
            print(f"[INFO] Wrote {out_count} records to {args.out}")
        else:
            # non-streaming: load in memory (safer for small datasets)
            bags = []
            for name in args.dataset:
                d = load_dataset(name, split=args.split, streaming=False, data_dir=args.data_dir)
                bags.append(d)
            ds_full = concatenate_datasets(bags)
            # Shuffle then cap
            ds_full = ds_full.shuffle(seed=args.seed)
            if args.max_samples:
                ds_full = ds_full.select(range(min(args.max_samples, len(ds_full))))
            for rec in ds_full:
                txt = record_to_text(rec)
                if not txt: continue
                writers.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                out_count += 1
            print(f"[INFO] Wrote {out_count} records to {args.out}")
    finally:
        writers.close()

if __name__ == "__main__":
    main()
