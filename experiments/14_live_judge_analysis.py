"""
Analyze live LLM judge scores as a paper-facing score function.

This intentionally writes separate artifacts rather than overwriting the broad
score-function comparison table, because live judge coverage may be a smaller
completed subset than the full 23-model measurement bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EVAL_N_VALUES, RESULTS_DIR  # noqa: E402


MEASUREMENTS_DIR = RESULTS_DIR / "measurements"
OUT_DIR = RESULTS_DIR / "du_aligned"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
LIVE_JUDGE_PATH = OUT_DIR / "model_judge" / "judged_responses_live.jsonl"
LIVE_MANIFEST_PATH = OUT_DIR / "model_judge" / "judge_subset_manifest_live.json"


def exact_curve(scores: np.ndarray, correct: np.ndarray, n_values: list[int]) -> dict[int, float]:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(scores)
    out = {}
    if n == 0:
        return {int(N): float("nan") for N in n_values}
    p = float(np.mean(correct))
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    groups = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        k_g = j - i
        c_g = int(np.sum(sorted_correct[i:j]))
        if c_g:
            groups.append((i + 1, j, k_g, c_g))
        i = j
    for N in n_values:
        if N == 1:
            out[N] = p
            continue
        if p == 0.0:
            out[N] = 0.0
            continue
        f = 0.0
        for r_min, r_max, k_g, c_g in groups:
            mass = (r_max / n) ** N - ((r_min - 1) / n) ** N
            f += (c_g / k_g) * mass
        out[int(N)] = float(f)
    return out


def auc_score(scores: np.ndarray, correct: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    pos = scores[correct]
    neg = scores[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = 0
    for score in pos:
        wins += float(np.sum(score > neg)) + 0.5 * float(np.sum(score == neg))
        total += len(neg)
    return float(wins / total) if total else float("nan")


def load_manifest_filter(path: Path) -> tuple[set[tuple[str, int]], int, dict]:
    if not path.exists():
        return set(), 0, {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    target_models = manifest.get("target_models", [])
    problem_indices = [int(pidx) for pidx in manifest.get("problem_indices", [])]
    samples_per_problem = int(manifest.get("samples_per_problem", 48) or 48)
    pairs = {(model, pidx) for model in target_models for pidx in problem_indices}
    return pairs, samples_per_problem, manifest


def load_live_judge_scores(path: Path, manifest_path: Path = LIVE_MANIFEST_PATH) -> tuple[dict[str, dict[int, np.ndarray]], dict]:
    manifest_pairs, samples_per_problem, manifest = load_manifest_filter(manifest_path)
    raw: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    if not path.exists():
        return {}, manifest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("judge_mode") != "live_no_reference":
            continue
        if str(rec.get("judge_source", "")).endswith(":failed"):
            continue
        if str(rec.get("reason_short", "")).startswith("live judge failed:"):
            continue
        model = rec["model"]
        pidx = int(rec["problem_idx"])
        sample_idx = int(rec["sample_idx"])
        if manifest_pairs and (model, pidx) not in manifest_pairs:
            continue
        if samples_per_problem and not (0 <= sample_idx < samples_per_problem):
            continue
        raw[model][pidx][sample_idx] = float(rec["judge_score"])
    out: dict[str, dict[int, np.ndarray]] = {}
    for model, by_problem in raw.items():
        out[model] = {}
        for pidx, by_sample in by_problem.items():
            pairs = sorted(by_sample.items())
            out[model][pidx] = np.asarray([score for _, score in pairs], dtype=float)
    return out, manifest


def load_measurement(model: str, pidx: int) -> dict | None:
    path = MEASUREMENTS_DIR / model / f"problem_{pidx}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def analyze() -> tuple[pd.DataFrame, dict]:
    judge_scores, manifest = load_live_judge_scores(LIVE_JUDGE_PATH)
    rows = []
    pair_rows = []
    for model, by_problem in sorted(judge_scores.items()):
        for pidx, scores in sorted(by_problem.items()):
            rec = load_measurement(model, pidx)
            if rec is None:
                continue
            correct = np.asarray(rec["all_correct"], dtype=bool)
            if len(scores) != len(correct):
                continue
            mean_logprob = np.asarray(rec["all_scores"], dtype=float)
            n_values = [N for N in EVAL_N_VALUES if N <= len(scores)]
            live_curve = exact_curve(scores, correct, n_values)
            mean_curve = exact_curve(mean_logprob, correct, n_values)
            live_auc = auc_score(scores, correct)
            mean_auc = auc_score(mean_logprob, correct)
            pair_rows.append(
                {
                    "model": model,
                    "problem_idx": pidx,
                    "num_samples": len(scores),
                    "live_judge_auc": live_auc,
                    "mean_logprob_auc": mean_auc,
                }
            )
            for N in n_values:
                rows.append(
                    {
                        "score_name": "live_llm_judge_score",
                        "model": model,
                        "problem_idx": pidx,
                        "N": int(N),
                        "actual_acc": live_curve[N],
                        "predicted_acc": live_curve[N],
                        "MAE": 0.0,
                        "mean_logprob_actual_acc": mean_curve[N],
                        "improvement_over_meanlogprob": live_curve[N] - mean_curve[N],
                        "auc": live_auc,
                    }
                )
    df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    summary: dict[str, object] = {
        "status": "completed" if not df.empty else "empty",
        "source_path": str(LIVE_JUDGE_PATH),
        "manifest_path": str(LIVE_MANIFEST_PATH),
        "score_name": "live_llm_judge_score",
        "manifest_n_problems": int(manifest.get("n_problems", 0) or len(manifest.get("problem_indices", []))),
        "manifest_samples_per_problem": int(manifest.get("samples_per_problem", 0) or 0),
        "manifest_expected_model_problem_pairs": int(
            len(manifest.get("target_models", [])) * len(manifest.get("problem_indices", []))
        ),
        "num_model_problem_pairs": int(pair_df[["model", "problem_idx"]].drop_duplicates().shape[0]) if not pair_df.empty else 0,
        "models": sorted(pair_df["model"].unique()) if not pair_df.empty else [],
        "N_values": sorted(int(n) for n in df["N"].unique()) if not df.empty else [],
        "mean_live_judge_auc": float(pair_df["live_judge_auc"].mean()) if not pair_df.empty else None,
        "mean_logprob_auc_on_same_pairs": float(pair_df["mean_logprob_auc"].mean()) if not pair_df.empty else None,
        "mean_mae": float(df["MAE"].mean()) if not df.empty else None,
    }
    if not df.empty:
        n48 = df[df["N"] == 48]
        if not n48.empty:
            summary["n48_live_judge_acc"] = float(n48["actual_acc"].mean())
            summary["n48_mean_logprob_acc_same_pairs"] = float(n48["mean_logprob_actual_acc"].mean())
            summary["n48_improvement_over_meanlogprob"] = float(n48["improvement_over_meanlogprob"].mean())
    return df, summary


def write_outputs(df: pd.DataFrame, summary: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "live_judge_score_comparison.csv", index=False)
    (OUT_DIR / "live_judge_score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(7.0, 4.6))
    if not df.empty:
        overall = (
            df.groupby("N")
            .agg(
                live_judge_acc=("actual_acc", "mean"),
                mean_logprob_acc=("mean_logprob_actual_acc", "mean"),
            )
            .reset_index()
            .sort_values("N")
        )
        plt.plot(overall["N"], overall["live_judge_acc"], marker="o", label="live LLM judge")
        plt.plot(overall["N"], overall["mean_logprob_acc"], marker="o", label="mean logprob")
    plt.xscale("log", base=2)
    plt.xticks(EVAL_N_VALUES, [str(n) for n in EVAL_N_VALUES])
    plt.xlabel("N")
    plt.ylabel("Best-of-N accuracy")
    plt.title("Live LLM judge score-function extension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "live_judge_score_comparison.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze live judge score-function artifacts.")
    parser.parse_args()
    df, summary = analyze()
    write_outputs(df, summary)
    print(f"Rows: {len(df)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
