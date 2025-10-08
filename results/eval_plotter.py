#!/usr/bin/env python3
"""
avg_plot_harness.py
Average multiple Harness JSON/JSONL runs per task & variant, then plot the averages.

Outputs (saved to ./figures):
  - grouped_bars.png     : Per-task grouped bars of MEAN accuracy by variant (+ optional error bars)
  - heatmap.png          : Task × Variant heatmap of MEAN accuracy
  - summary_mean.png     : Overall MEAN accuracy by variant
  - aggregated.csv       : Table of aggregated stats per (task, variant)

Usage:
  python avg_plot_harness.py --input results_dir_or_zip
  python avg_plot_harness.py --input results_dir_or_zip --errtype runstd
  python avg_plot_harness.py --input results_dir_or_zip --tasks arc_challenge hellaswag mmlu
  python avg_plot_harness.py --input results_dir_or_zip --topk 12

Error bars:
  - runstd  : std across runs for the chosen accuracy metric (default; shows run-to-run variation)
  - stderr  : RMS-combined Harness *_stderr of the chosen metric
  - none    : no error bars
"""

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

# ----------------------------
# Config
# ----------------------------
PREFERRED_METRICS = [
    ("acc_norm,none", "acc_norm_stderr,none"),
    ("acc,none", "acc_stderr,none"),
    ("exact_match,get-answer", None),
]

# ----------------------------
# I/O helpers
# ----------------------------
def iter_input_files(input_path: Path) -> Iterable[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    base = None
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        tmpdir = Path(tempfile.mkdtemp(prefix="harness_"))
        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(tmpdir)
        base = tmpdir
    elif input_path.is_dir():
        base = input_path
    elif input_path.is_file() and input_path.suffix.lower() in {".json", ".jsonl"}:
        yield input_path
        return
    else:
        base = input_path.parent

    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        if p.suffix.lower() in {".json", ".jsonl"}:
            yield p

def load_json_best_effort(fp: Path):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None

def iter_jsonl(fp: Path):
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

# ----------------------------
# Parsing Harness structure
# ----------------------------
def is_harness_obj(obj) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("results"), dict)

def infer_variant_from_path(path: Path, model_name: str) -> str:
    s = (str(path).lower() + " " + str(model_name or "")).lower()
    if "_relb_" in s or "relb" in s or "relation" in s or "rel-" in s:
        return "Relation"
    if "_fb_" in s or "feature" in s or "feat-" in s:
        return "Feature"
    if "_rb_" in s or "response" in s or "resp-" in s:
        return "Response"
    if "teacher" in s or "_base_" in s or "baseline" in s:
        return "Teacher"
    if "student" in s:
        return "Student"
    return "Unknown"

def extract_rows_from_harness(obj: Dict, src_path: Path) -> List[Dict]:
    rows = []
    results = obj.get("results", {})
    model = obj.get("model_name_sanitized", obj.get("model_name", "unknown_model"))
    date = obj.get("date", obj.get("end_time", obj.get("start_time", "")))
    variant = infer_variant_from_path(src_path, model)
    run_id = src_path.stem  # heuristic run identifier from filename

    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue

        chosen_metric = None
        chosen_stderr = None
        for m, sderr in PREFERRED_METRICS:
            if m in metrics:
                chosen_metric = m
                if sderr and sderr in metrics:
                    chosen_stderr = sderr
                break
        if chosen_metric is None:
            continue

        rows.append({
            "run_id": run_id,
            "variant": variant,
            "model": model,
            "date": date,
            "task": task,
            "metric": chosen_metric,
            "value": metrics.get(chosen_metric, np.nan),
            "stderr": metrics.get(chosen_stderr, np.nan) if chosen_stderr else np.nan,
        })
    return rows

def load_all_results(input_path: Path) -> pd.DataFrame:
    all_rows: List[Dict] = []
    for fp in iter_input_files(input_path):
        try:
            if fp.suffix.lower() == ".jsonl":
                for obj in iter_jsonl(fp):
                    if is_harness_obj(obj):
                        all_rows.extend(extract_rows_from_harness(obj, fp))
            else:
                obj = load_json_best_effort(fp)
                if obj and is_harness_obj(obj):
                    all_rows.extend(extract_rows_from_harness(obj, fp))
        except Exception:
            continue

    if not all_rows:
        raise RuntimeError("No Harness-style results found.")

    df = pd.DataFrame(all_rows)
    # Coerce numerics
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "stderr" in df.columns:
        df["stderr"] = pd.to_numeric(df["stderr"], errors="coerce")

    # Normalize to percentage if it looks like [0,1]
    finite_vals = df["value"].dropna()
    if not finite_vals.empty and (finite_vals.between(0, 1).mean() > 0.7):
        df["value"] = df["value"] * 100.0
        if "stderr" in df.columns:
            df["stderr"] = df["stderr"] * 100.0
    return df

# ----------------------------
# Aggregation across runs
# ----------------------------
def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across runs for each (task, variant):
      - mean_value : mean accuracy across runs
      - run_std    : std across runs (variation across your 5 runs)
      - n_runs     : number of runs aggregated
      - stderr_rms : RMS-combined Harness stderr across runs (sqrt(sum(se_i^2))/n_runs)
    """
    def rms_stderr(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return np.nan
        return float(np.sqrt(np.sum(np.square(s))) / len(s))

    agg = (
        df.groupby(["task", "variant"], dropna=False)
          .agg(
              mean_value=("value", "mean"),
              run_std=("value", "std"),
              n_runs=("value", "count"),
              stderr_rms=("stderr", rms_stderr)
          )
          .reset_index()
    )
    return agg

# ----------------------------
# Plotting
# ----------------------------
def ensure_figdir() -> Path:
    out = Path("./figures")
    out.mkdir(parents=True, exist_ok=True)
    return out

def pick_tasks(agg: pd.DataFrame, tasks: Optional[List[str]], topk: int) -> List[str]:
    if tasks:
        return tasks
    # prefer tasks appearing in the most variants
    counts = agg.groupby("task")["variant"].nunique().sort_values(ascending=False)
    ordered = counts.index.tolist()
    if topk > 0:
        ordered = ordered[:topk]
    return ordered

def _yerr_for_variant(agg_pivot_err: pd.DataFrame, variant: str, errtype: str):
    if variant not in agg_pivot_err.columns:
        return None
    arr = pd.to_numeric(agg_pivot_err[variant], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(arr).any():
        return None
    arr[~np.isfinite(arr)] = np.nan  # matplotlib accepts NaN, not None
    return arr

def plot_grouped_bars(agg: pd.DataFrame, tasks: List[str], figdir: Path, errtype: str):
    """
    Grouped bars of MEAN accuracy across runs.
    - Pretty task labels
    - Removed right/top spines
    - Hatch patterns for variants
    """

    # Map raw task IDs to nicer display names
    task_name_map = {
        "arc_challenge": "ARC-Challenge",
        "bbh": "BBH",
        "bbh_cot_fewshot_boolean_expressions": "BBH-BooleanExpr",
        "mmlu_world_religions": "MMLU-World Religions",
        "mmlu_nutrition": "MMLU-Nutrition",
        "mmlu_other": "MMLU-Other",
        "mmlu_philosophy": "MMLU-Philosophy",
        "mmlu_prehistory": "MMLU-Prehistory",
        "mmlu_professional_accounting": "MMLU-Accounting",
        "mmlu_professional_law": "MMLU-Law",
        "mmlu_professional_medicine": "MMLU-Medicine",
        "mmlu_professional_psychology": "MMLU-Psychology",
    }

    sub = agg[agg["task"].isin(tasks)].copy()
    variants = sorted(sub["variant"].dropna().unique().tolist())

    means = (
        sub.pivot(index="task", columns="variant", values="mean_value")
           .reindex(index=tasks, columns=variants)
    )
    if errtype == "runstd":
        errs = sub.pivot(index="task", columns="variant", values="run_std").reindex(index=tasks, columns=variants)
    elif errtype == "stderr":
        errs = sub.pivot(index="task", columns="variant", values="stderr_rms").reindex(index=tasks, columns=variants)
    else:
        errs = None

    fig, ax = plt.subplots(figsize=(max(8, 0.8*len(tasks)), 5))
    x = np.arange(len(tasks))
    n_vars = len(variants)
    width = 0.8 / max(n_vars, 1)

    # Define hatch patterns for each variant (add more if needed)
    hatches = ["//", "\\\\", "xx", "oo", "..", "**"]

    for i, var in enumerate(variants):
        y = pd.to_numeric(means[var], errors="coerce").to_numpy(dtype=float)
        yerr = None
        if errs is not None:
            yerr = _yerr_for_variant(errs, var, errtype)
        xpos = x + i*width - (n_vars - 1) * width / 2
        ax.bar(
            xpos, y, width, label=var, yerr=yerr,
            capsize=3 if yerr is not None else 0,
            hatch=hatches[i % len(hatches)], edgecolor="black"
        )

    # Clean up spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Labels
    ax.set_ylabel("Mean Accuracy (%)")
    title = "Per-task Mean Accuracy (Averaged over 5 Runs)"
    if errtype == "runstd":
        title += " — Error bars: run-to-run std"
    elif errtype == "stderr":
        title += " — Error bars: RMS Harness stderr"
    ax.set_title(title)

    # Use prettier task names
    ax.set_xticks(x)
    ax.set_xticklabels([task_name_map.get(t, t) for t in tasks], rotation=45, ha="right")

    ax.yaxis.set_major_locator(MaxNLocator(10))

    # --- Custom legend with hatch + edgecolor ---
    handles = []
    for i, var in enumerate(variants):
        patch = mpatches.Patch(
            facecolor="C{}".format(i),  # default matplotlib color cycle
            hatch=hatches[i % len(hatches)],
            edgecolor="black",
            label=var
        )
        handles.append(patch)
    ax.legend(handles=handles, 
              title="KD Type",
              loc="upper right",
              bbox_to_anchor=(1.12, 1),  # shift it outside the axes
              borderaxespad=0.0
              )

    fig.tight_layout()

    out = figdir / "grouped_bars.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)



def plot_heatmap(agg: pd.DataFrame, tasks: List[str], figdir: Path):
    task_name_map = {
        "arc_challenge": "ARC-Challenge",
        "bbh": "BBH",
        "bbh_cot_fewshot_boolean_expressions": "BBH-BooleanExpr",
        "mmlu_world_religions": "MMLU-World Religions",
        "mmlu_nutrition": "MMLU-Nutrition",
        "mmlu_other": "MMLU-Other",
        "mmlu_philosophy": "MMLU-Philosophy",
        "mmlu_prehistory": "MMLU-Prehistory",
        "mmlu_professional_accounting": "MMLU-Accounting",
        "mmlu_professional_law": "MMLU-Law",
        "mmlu_professional_medicine": "MMLU-Medicine",
        "mmlu_professional_psychology": "MMLU-Psychology",
    }
    
    mat = (
        agg.pivot(index="task", columns="variant", values="mean_value")
           .reindex(index=tasks)
    )
    data = mat.values
    fig, ax = plt.subplots(figsize=(max(6, 0.6*mat.shape[1]), max(6, 0.5*mat.shape[0])))
    im = ax.imshow(data, aspect="auto")
    ax.set_title("Task × Variant — Mean Accuracy (%)")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticklabels([task_name_map.get(t,t) for t in mat.index])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iat[i, j]
            if isinstance(val, (int, float)) and math.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Mean Accuracy (%)")
    fig.tight_layout()
    out = figdir / "heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

def plot_summary(agg: pd.DataFrame, figdir: Path):
    means = agg.groupby("variant")["mean_value"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(means.index)), means.values)
    ax.set_xticks(np.arange(len(means.index)))
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Overall Mean Accuracy (%)")
    ax.set_title("Overall Mean Accuracy by Variant (Averaged across Tasks & Runs)")
    ax.yaxis.set_major_locator(MaxNLocator(10))
    ax.set_ylim(0, 100)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    out = figdir / "summary_mean.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, type=Path, help="Directory or .zip with Harness JSON/JSONL")
    ap.add_argument("--tasks", nargs="*", help="Specific tasks to plot (default: auto pick common ones)")
    ap.add_argument("--topk", type=int, default=12, help="If --tasks not given, pick top-k tasks (default 12)")
    ap.add_argument("--errtype", choices=["runstd", "stderr", "none"], default="runstd",
                    help="Type of error bars for grouped bars (default: runstd)")
    ap.add_argument("--save_csv", action="store_true", help="Also save aggregated.csv")
    args = ap.parse_args()

    df = load_all_results(args.input)
    agg = aggregate_runs(df)

    tasks = pick_tasks(agg, args.tasks, args.topk)
    figdir = ensure_figdir()

    plot_grouped_bars(agg, tasks, figdir, args.errtype)
    plot_heatmap(agg, tasks, figdir)
    plot_summary(agg, figdir)

    if args.save_csv:
        agg.to_csv("aggregated.csv", index=False)

    print("[OK] Figures in:", figdir.resolve())
    if args.save_csv:
        print("[OK] Aggregated table:", Path("aggregated.csv").resolve())

if __name__ == "__main__":
    main()
