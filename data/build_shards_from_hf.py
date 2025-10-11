#!/usr/bin/env python3
import argparse, json, random, sys, os, gzip
from typing import Optional, Dict, Any, List, Tuple
from datasets import load_dataset, Dataset

# ============== Spec parsing ==============
def parse_dataset_spec(spec: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse 'name[:config][@split]' into (name, config_or_None, split_or_None).
    Examples:
      "allenai/c4:en"            -> ("allenai/c4", "en", None)
      "wikipedia:20240520.en"    -> ("wikipedia", "20240520.en", None)
      "BeIR/nq@test"             -> ("BeIR/nq", None, "test")
      "allenai/c4:en@train"      -> ("allenai/c4", "en", "train")
      "hotpotqa/hotpot_qa"       -> ("hotpotqa/hotpot_qa", None, None)
    """
    if "@" in spec:
        name_cfg, split = spec.split("@", 1)
    else:
        name_cfg, split = spec, None
    if ":" in name_cfg:
        base, cfg = name_cfg.split(":", 1)
    else:
        base, cfg = name_cfg, None
    return base, (cfg or None), (split or None)

# ============== Text normalization ==============
def record_to_text(rec: Dict[str, Any]) -> Optional[str]:
    # 1) Raw "text"
    txt = rec.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # 2) Instruction-style pairs
    for a, b in [("instruction", "output"), ("prompt", "response"), ("input", "output")]:
        if isinstance(rec.get(a), str) and isinstance(rec.get(b), str):
            s = rec[a].strip()
            r = rec[b].strip()
            if s or r:
                return f"### Instruction:\n{s}\n\n### Response:\n{r}"

    # 3) Chat messages: list of {role, content}
    msgs = rec.get("messages")
    if isinstance(msgs, (list, tuple)):
        lines: List[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                lines.append(str(m)); continue
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                body = content
            elif isinstance(content, list):
                parts = []
                for seg in content:
                    if isinstance(seg, dict):
                        if isinstance(seg.get("text"), str):
                            parts.append(seg["text"])
                        elif isinstance(seg.get("content"), str):
                            parts.append(seg["content"])
                        else:
                            parts.append(str(seg))
                    else:
                        parts.append(str(seg))
                body = "\n".join(p for p in parts if p)
            elif isinstance(content, dict):
                body = content.get("text") or content.get("content") or json.dumps(content, ensure_ascii=False)
            else:
                body = str(content)
            line = f"{role}: {body}".strip()
            if line:
                lines.append(line)
        text = "\n".join(lines).strip()
        if text:
            return text

    # 4) Fallback: join all string fields
    parts = []
    for k, v in rec.items():
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")
    return "\n".join(parts).strip() if parts else None

# ============== IO helpers ==============
def safe_makedirs_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def open_out(path: str, use_gzip: bool):
    if use_gzip:
        if not path.endswith(".gz"):
            path = path + ".gz"
        return gzip.open(path, "wt", encoding="utf-8"), path
    return open(path, "w", encoding="utf-8"), path

# ============== Quotas ==============
def compute_quotas(num_datasets: int, weights: List[float], max_samples: Optional[int]) -> Optional[List[int]]:
    if max_samples is None:
        return None
    raw = [max(0, int(round(max_samples * w))) for w in weights]
    diff = max_samples - sum(raw)
    fracs = [(i, (weights[i] * max_samples) - int(weights[i] * max_samples)) for i in range(num_datasets)]
    fracs_sorted = sorted(fracs, key=lambda x: x[1], reverse=(diff > 0))
    order = [i for i, _ in fracs_sorted]
    for k in range(abs(diff)):
        i = order[k % num_datasets]
        raw[i] = max(0, raw[i] + (1 if diff > 0 else -1))
    if max_samples >= num_datasets:
        for i in range(num_datasets):
            if raw[i] == 0:
                j = max(range(num_datasets), key=lambda x: raw[x])
                if raw[j] > 1:
                    raw[j] -= 1
                    raw[i] += 1
    return raw

# ============== Loader ==============
def load_one(spec: str, default_split: str, cache_dir: Optional[str], streaming: bool, data_dir: Optional[str]):
    base, cfg, split_override = parse_dataset_spec(spec)
    split = split_override or default_split
    kwargs = {}
    if cache_dir: kwargs["cache_dir"] = cache_dir
    if data_dir:  kwargs["data_dir"]  = data_dir

    # Always trust remote dataset code (Wikipedia, MultiWOZ, etc.)
    kwargs["trust_remote_code"] = True

    try:
        ds = load_dataset(base, cfg, split=split, streaming=streaming, **kwargs)
        # IMPORTANT: for streaming datasets with complex schemas, force Python dicts (bypass Arrow casting)
        if streaming:
            ds = ds.with_format("python")
        return ds
    except Exception as e:
        # If streaming failed due to Arrow/format issues, try non-streaming just for this dataset
        msg = f"[WARN] primary load failed for '{spec}' (streaming={streaming}); attempting non-streaming fallback: {e}"
        print(msg, file=sys.stderr, flush=True)
        try:
            ds = load_dataset(base, cfg, split=split, streaming=False, **kwargs)
            return ds
        except Exception as e2:
            offline = str(os.environ.get("HF_DATASETS_OFFLINE", "0")).lower() in ("1", "true", "yes")
            final = f"[ERROR] Failed to load dataset '{spec}' (split='{split}')."
            if offline:
                final += " HF_DATASETS_OFFLINE=1 may prevent downloads; ensure cache/mirror is present."
            raise RuntimeError(final) from e2


# ============== Main ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True,
                    help="HF dataset spec (repeatable). Use name[:config][@split], e.g. 'allenai/c4:en' or 'BeIR/nq@test'")
    ap.add_argument("--split", default="train", help="Default split if dataset spec omits '@split' (default: train)")
    ap.add_argument("--weights", type=str, default="",
                    help="Comma-separated weights matching --dataset count, e.g. '0.8,0.2'")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap total samples across all datasets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/shards.jsonl")
    ap.add_argument("--streaming", action="store_true", default=True)
    ap.add_argument("--no-streaming", dest="streaming", action="store_false")
    ap.add_argument("--data_dir", default=None,
                    help="If mirroring locally, point to dataset files dir (builder-specific)")
    ap.add_argument("--cache_dir", default=None,
                    help="Local HF datasets cache dir (recommended on HPC fast storage)")
    ap.add_argument("--shuffle_streaming", action="store_true",
                    help="Buffer and shuffle per-dataset streaming (uses memory)")
    ap.add_argument("--buffer_size", type=int, default=50000,
                    help="Streaming shuffle buffer size if --shuffle_streaming is set")
    ap.add_argument("--with-meta", action="store_true",
                    help="Include metadata fields (source, row_idx) in output")
    ap.add_argument("--gzip", action="store_true",
                    help="Write gzip-compressed JSONL (.gz)")
    args = ap.parse_args()

    # Auto-disable streaming if offline env is set
    if str(os.environ.get("HF_DATASETS_OFFLINE", "0")).lower() in ("1", "true", "yes") and args.streaming:
        print("[WARN] HF_DATASETS_OFFLINE=1 and --streaming are incompatible; forcing --no-streaming.", file=sys.stderr)
        args.streaming = False

    random.seed(args.seed)
    safe_makedirs_for(args.out)

    # Weights
    if args.weights:
        ws = [float(x) for x in args.weights.split(",")]
        if len(ws) != len(args.dataset):
            print("[ERROR] weights length must match number of --dataset entries", file=sys.stderr)
            sys.exit(2)
        total_w = sum(ws)
        if total_w <= 0:
            print("[ERROR] weights must sum to > 0", file=sys.stderr)
            sys.exit(2)
        ws = [w / total_w for w in ws]
    else:
        ws = [1.0 / len(args.dataset)] * len(args.dataset)

    quotas = compute_quotas(len(args.dataset), ws, args.max_samples)

    writer, out_path = open_out(args.out, args.gzip)
    out_count = 0

    try:
        if args.streaming:
            # -------- Streaming path --------
            for i, spec in enumerate(args.dataset):
                quota = None if quotas is None else quotas[i]
                if quota == 0:
                    continue
                try:
                    print(f"[INFO] Loading dataset (streaming): {spec} | quota={quota}", flush=True)
                    ds = load_one(spec, args.split, cache_dir=args.cache_dir, data_dir=args.data_dir, streaming=True)
                except Exception as e:
                    print(f"[WARN] skipping dataset '{spec}': {e}", file=sys.stderr, flush=True)
                    continue

                if args.shuffle_streaming:
                    buf: List[Dict[str, Any]] = []
                    taken = 0
                    for row_idx, row in enumerate(ds):
                        buf.append(row)
                        if quota is not None and taken >= quota:
                            break
                        if len(buf) >= args.buffer_size:
                            random.shuffle(buf)
                            for j, b in enumerate(buf):
                                if quota is not None and taken >= quota:
                                    break
                                txt = record_to_text(b)
                                if not txt:
                                    continue
                                obj = {"text": txt}
                                if args.with_meta:
                                    obj.update({"source": spec, "row_idx": j})
                                writer.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                out_count += 1
                                taken += 1
                                # ---- DEBUG PROGRESS (quota-aware ~1%, else every 100k) ----
                                if quota and quota > 0:
                                    if taken % max(1, quota // 100) == 0:
                                        pct = (taken / quota) * 100
                                        print(f"[DEBUG] {spec} — wrote {taken:,}/{quota:,} samples ({pct:.1f}%)", flush=True)
                                elif taken % 100000 == 0:
                                    print(f"[DEBUG] {spec} — wrote {taken:,} samples", flush=True)
                            buf.clear()
                    # flush remaining
                    if quota is None or taken < quota:
                        random.shuffle(buf)
                        for j, b in enumerate(buf):
                            if quota is not None and taken >= quota:
                                break
                            txt = record_to_text(b)
                            if not txt:
                                continue
                            obj = {"text": txt}
                            if args.with_meta:
                                obj.update({"source": spec, "row_idx": j})
                            writer.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            out_count += 1
                            taken += 1
                            if quota and quota > 0:
                                if taken % max(1, quota // 100) == 0:
                                    pct = (taken / quota) * 100
                                    print(f"[DEBUG] {spec} — wrote {taken:,}/{quota:,} samples ({pct:.1f}%)", flush=True)
                            elif taken % 100000 == 0:
                                print(f"[DEBUG] {spec} — wrote {taken:,} samples", flush=True)
                    print(f"[INFO] Finished {spec} — wrote {taken:,}{' / ' + str(quota) if quota else ''} samples", flush=True)

                else:
                    # plain sequential
                    c = 0
                    for row_idx, rec in enumerate(ds):
                        if quota is not None and c >= quota:
                            break
                        txt = record_to_text(rec)
                        if not txt:
                            continue
                        obj = {"text": txt}
                        if args.with_meta:
                            obj.update({"source": spec, "row_idx": row_idx})
                        writer.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        out_count += 1
                        c += 1
                        # ---- DEBUG PROGRESS ----
                        if quota and quota > 0:
                            if c % max(1, quota // 100) == 0:
                                pct = (c / quota) * 100
                                print(f"[DEBUG] {spec} — wrote {c:,}/{quota:,} samples ({pct:.1f}%)", flush=True)
                        elif c % 100000 == 0:
                            print(f"[DEBUG] {spec} — wrote {c:,} samples", flush=True)

                    print(f"[INFO] Finished {spec} — wrote {c:,}{' / ' + str(quota) if quota else ''} samples", flush=True)

            print(f"[INFO] Wrote {out_count:,} records to {out_path}", flush=True)

        else:
            # -------- Non-streaming path (robust offline) --------
            rows: List[Dict[str, Any]] = []
            per_ds_limits = quotas if quotas is not None else [None] * len(args.dataset)

            for i, spec in enumerate(args.dataset):
                try:
                    print(f"[INFO] Loading dataset (non-streaming): {spec} | limit={per_ds_limits[i]}", flush=True)
                    d = load_one(spec, args.split, cache_dir=args.cache_dir, data_dir=args.data_dir, streaming=False)
                except Exception as e:
                    print(f"[WARN] skipping dataset '{spec}': {e}", file=sys.stderr, flush=True)
                    continue

                take = per_ds_limits[i]
                taken = 0
                row_idx_local = 0
                for rec in d:
                    if take is not None and taken >= take:
                        break
                    txt = record_to_text(rec)
                    row_idx_local += 1
                    if not txt:
                        continue
                    obj = {"text": txt}
                    if args.with_meta:
                        obj.update({"source": spec, "row_idx": row_idx_local - 1})
                    rows.append(obj)
                    taken += 1
                    # ---- DEBUG PROGRESS ----
                    if take and take > 0:
                        if taken % max(1, take // 100) == 0:
                            pct = (taken / take) * 100
                            print(f"[DEBUG] {spec} — buffered {taken:,}/{take:,} samples ({pct:.1f}%)", flush=True)
                    elif taken % 100000 == 0:
                        print(f"[DEBUG] {spec} — buffered {taken:,} samples", flush=True)

                print(f"[INFO] Finished buffering {spec} — {taken:,}{' / ' + str(take) if take else ''}", flush=True)

            if not rows:
                print("[ERROR] No datasets could be loaded / produced rows.", file=sys.stderr, flush=True)
                sys.exit(4)

            # Shuffle deterministically, then cap globally (cap already respected per-ds if quotas set)
            random.shuffle(rows)
            if args.max_samples is not None and args.max_samples < len(rows):
                rows = rows[:args.max_samples]

            # Write with periodic debug
            for idx, rec in enumerate(rows, 1):
                writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_count += 1
                if idx % max(1, len(rows) // 100) == 0:
                    pct = (idx / len(rows)) * 100
                    print(f"[DEBUG] write-out — {idx:,}/{len(rows):,} ({pct:.1f}%)", flush=True)
                elif idx % 100000 == 0:
                    print(f"[DEBUG] write-out — {idx:,} records", flush=True)

            print(f"[INFO] Wrote {out_count:,} records to {out_path}", flush=True)

    finally:
        writer.close()
        print("[INFO] Output file closed.", flush=True)

if __name__ == "__main__":
    main()