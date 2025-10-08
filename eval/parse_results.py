#!/usr/bin/env python3
import argparse, glob, json, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_csv", default="results/eval_summary.csv")
    args = ap.parse_args()

    rows = []
    for fn in glob.glob(f"{args.results_dir}/*.json"):
        with open(fn, "r") as f:
            data = json.load(f)
        model = fn.split("/")[-1].replace(".json", "")
        for task, score in data.get("results", {}).items():
            rows.append({"model": model, "task": task, "score": score})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[INFO] Wrote {args.out_csv}")

if __name__ == "__main__":
    main()
