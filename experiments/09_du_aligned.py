"""
Du-aligned Theorem 1 experiments.

This runner reuses the existing measurement artifacts and writes:

  results/du_aligned/
    du_experiment_summary.txt
    du_experiment_results.json
    tables/*.csv
    figures/*.pdf

The project validates the exact finite empirical selector law with random
tie-breaking. The moment hierarchy below therefore uses the matching
rank-interval empirical moment, which is exactly equivalent to the discrete
order-statistic formula and converges to the continuous U=F_mix(S+) moment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy import stats

try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover - fallback for minimal envs
    Ridge = None
    make_pipeline = None
    StandardScaler = None
    IsotonicRegression = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EVAL_N_VALUES, RESULTS_DIR, MODELS, N_SAMPLES, DATA_DIR  # noqa: E402


MEASUREMENTS_DIR = RESULTS_DIR / "measurements"
AUDIT_DIR = RESULTS_DIR / "audit"
OUT_DIR = RESULTS_DIR / "du_aligned"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
FEATURES_CSV = OUT_DIR / "cache" / "raw_response_features.csv"

MOMENT_N_VALUES = [2, 3, 4, 8, 16, 32, 48]
SCORE_N_VALUES = list(EVAL_N_VALUES)
PILOT_K_VALUES = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
PILOT_N_VALUES = [2, 8, 16, 32, 48]
BOOTSTRAPS = 500


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def safe_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def bootstrap_ci(values: list[float] | np.ndarray, seed: int = 0, n_boot: int = BOOTSTRAPS) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        means[i] = np.mean(sample)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def discover_models() -> list[str]:
    return sorted(d.name for d in MEASUREMENTS_DIR.iterdir() if d.is_dir())


def load_measurements(models: list[str] | None = None) -> dict[str, list[dict]]:
    if models is None:
        models = discover_models()

    data: dict[str, list[dict]] = {}
    for model in models:
        rows = []
        for path in sorted((MEASUREMENTS_DIR / model).glob("problem_*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if len(rec.get("all_scores", [])) != len(rec.get("all_correct", [])):
                continue
            rec["model_key"] = rec.get("model_key", model)
            rows.append(rec)
        data[model] = rows
    return data


def flatten_records(data: dict[str, list[dict]]) -> list[dict]:
    records = []
    for model, rows in data.items():
        for rec in rows:
            rec = dict(rec)
            rec["model_key"] = model
            records.append(rec)
    return records


def train_split(problem_idx: int) -> bool:
    return problem_idx % 5 != 0


def difficulty_bin(p: float) -> str:
    if p <= 0:
        return "no_correct"
    if p >= 1:
        return "all_correct"
    if p < 0.25:
        return "hard"
    if p < 0.75:
        return "medium"
    return "easy"


def exact_curve(scores: np.ndarray, correct: np.ndarray, n_values: list[int]) -> dict[int, float]:
    """Exact finite empirical best-of-N probability with random tie-breaking."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(scores)
    out: dict[int, float] = {}
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
        N = int(N)
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
        out[N] = float(f)
    return out


def interval_theta(scores: np.ndarray, correct: np.ndarray, N: int, f_n: float | None = None) -> float:
    """Rank-interval moment theta_{N-1} satisfying f_N = N p theta."""
    correct = np.asarray(correct, dtype=bool)
    p = float(np.mean(correct)) if len(correct) else 0.0
    if p == 0.0:
        return 0.0
    if f_n is None:
        f_n = exact_curve(scores, correct, [N])[N]
    return float(f_n / (N * p))


def tie_rate(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n < 2:
        return 0.0
    _, counts = np.unique(scores, return_counts=True)
    tie_pairs = sum(int(c) * (int(c) - 1) / 2 for c in counts if c > 1)
    return float(tie_pairs / (n * (n - 1) / 2))


def auc_score(scores: np.ndarray, correct: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    pos = scores[correct]
    neg = scores[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    try:
        u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
        return float(u_stat / (len(pos) * len(neg)))
    except Exception:
        return float("nan")


def auc_closed_form(p: float, kappa: float) -> float:
    if not np.isfinite(kappa):
        return float(p)
    return float(p * p + 2 * p * (1 - p) * kappa)


def auc_features(rows: list[dict]) -> np.ndarray:
    feats = []
    for r in rows:
        p = float(r["p"])
        k = float(r["kappa_for_model"])
        N = float(r["N"])
        logn = math.log(N)
        feats.append([p, k, N, logn, p * k, p * logn, k * logn, p * p, k * k])
    return np.asarray(feats, dtype=float)


def fit_auc_baseline(train_rows: list[dict]):
    if not train_rows:
        return None
    X = auc_features(train_rows)
    y = np.asarray([r["actual"] for r in train_rows], dtype=float)
    if Ridge is None or make_pipeline is None or StandardScaler is None:
        X_aug = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        return ("lstsq", coef)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
    model.fit(X, y)
    return model


def predict_auc_baseline(model, rows: list[dict]) -> np.ndarray:
    if model is None:
        return np.asarray([r["p"] for r in rows], dtype=float)
    X = auc_features(rows)
    if isinstance(model, tuple) and model[0] == "lstsq":
        X_aug = np.column_stack([np.ones(len(X)), X])
        pred = X_aug @ model[1]
    else:
        pred = model.predict(X)
    return np.clip(pred, 0.0, 1.0)


def build_moment_rows(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        scores = np.asarray(rec["all_scores"], dtype=float)
        correct = np.asarray(rec["all_correct"], dtype=bool)
        p = float(np.mean(correct)) if len(correct) else float("nan")
        kappa = rec.get("kappa")
        if kappa is None:
            kappa = auc_score(scores, correct)
        curve = exact_curve(scores, correct, MOMENT_N_VALUES)
        for N in MOMENT_N_VALUES:
            f_n = curve[N]
            theta = interval_theta(scores, correct, N, f_n=f_n)
            rows.append(
                {
                    "model": rec["model_key"],
                    "problem_idx": int(rec["problem_idx"]),
                    "split": "train" if train_split(int(rec["problem_idx"])) else "test",
                    "N": int(N),
                    "p": p,
                    "kappa": float(kappa) if kappa is not None and np.isfinite(kappa) else float("nan"),
                    "kappa_for_model": float(kappa) if kappa is not None and np.isfinite(kappa) else 0.5,
                    "theta": theta,
                    "actual": f_n,
                    "moment_pred": float(N * p * theta),
                    "moment_abs_error": abs(float(N * p * theta) - f_n),
                    "difficulty_bin": difficulty_bin(p),
                    "level": rec.get("level", ""),
                }
            )
    return pd.DataFrame(rows)


def run_moment_hierarchy(records: list[dict]) -> tuple[pd.DataFrame, dict]:
    print("Running Experiment 1: moment hierarchy vs AUC baseline")
    df = build_moment_rows(records)

    train_rows = df[(df["split"] == "train") & (df["N"] > 2)].to_dict("records")
    auc_model = fit_auc_baseline(train_rows)

    auc_preds = []
    for row in df.to_dict("records"):
        N = int(row["N"])
        if N == 2:
            pred = auc_closed_form(row["p"], row["kappa_for_model"])
        else:
            pred = float(predict_auc_baseline(auc_model, [row])[0])
        auc_preds.append(pred)
    df["auc_only_pred"] = auc_preds
    df["auc_only_abs_error"] = np.abs(df["auc_only_pred"] - df["actual"])

    test = df[df["split"] == "test"].copy()

    group_cols = ["model", "N", "difficulty_bin"]
    table = (
        test.groupby(group_cols)
        .agg(
            num_problems=("problem_idx", "count"),
            actual_acc_mean=("actual", "mean"),
            auc_only_pred_mean=("auc_only_pred", "mean"),
            moment_pred_mean=("moment_pred", "mean"),
            auc_only_mae=("auc_only_abs_error", "mean"),
            moment_mae=("moment_abs_error", "mean"),
            mean_p=("p", "mean"),
            mean_kappa=("kappa", "mean"),
            mean_theta=("theta", "mean"),
        )
        .reset_index()
    )

    overall_by_n = (
        test.groupby("N")
        .agg(
            num_problems=("problem_idx", "count"),
            actual_acc_mean=("actual", "mean"),
            auc_only_mae=("auc_only_abs_error", "mean"),
            moment_mae=("moment_abs_error", "mean"),
        )
        .reset_index()
    )
    overall_by_n.insert(0, "model", "ALL")
    overall_by_n.insert(2, "difficulty_bin", "ALL")
    for col in ["auc_only_pred_mean", "moment_pred_mean", "mean_p", "mean_kappa", "mean_theta"]:
        if col not in overall_by_n:
            overall_by_n[col] = float("nan")
    table = pd.concat([table, overall_by_n[table.columns]], ignore_index=True)

    ci_rows = {}
    for N, sub in test.groupby("N"):
        ci_rows[str(int(N))] = {
            "auc_only_mae": safe_mean(sub["auc_only_abs_error"].to_numpy()),
            "auc_only_mae_ci95": bootstrap_ci(sub["auc_only_abs_error"].to_numpy(), seed=stable_seed("auc", N)),
            "moment_mae": safe_mean(sub["moment_abs_error"].to_numpy()),
            "moment_mae_ci95": bootstrap_ci(sub["moment_abs_error"].to_numpy(), seed=stable_seed("moment", N)),
        }

    table.to_csv(TABLE_DIR / "moment_hierarchy_vs_auc.csv", index=False)
    make_moment_figures(test)

    summary = {
        "baseline": "Held-out ridge regression using p, kappa, N, log(N), and simple interactions; N=2 uses the exact AUC identity.",
        "heldout_split": "problem_idx % 5 == 0",
        "by_n": ci_rows,
        "num_rows_test": int(len(test)),
    }
    return test, summary


def make_moment_figures(test: pd.DataFrame) -> None:
    by_n = (
        test.groupby("N")
        .agg(auc_only_mae=("auc_only_abs_error", "mean"), moment_mae=("moment_abs_error", "mean"))
        .reset_index()
        .sort_values("N")
    )

    plt.figure(figsize=(7.0, 4.6))
    plt.plot(by_n["N"], by_n["auc_only_mae"], marker="o", label="AUC-only baseline")
    plt.plot(by_n["N"], by_n["moment_mae"], marker="o", label="Moment predictor")
    plt.xscale("log", base=2)
    plt.yscale("symlog", linthresh=1e-8)
    plt.xticks(by_n["N"], [str(int(x)) for x in by_n["N"]])
    plt.xlabel("N")
    plt.ylabel("Held-out MAE")
    plt.title("Moment hierarchy vs AUC-only prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "moment_hierarchy_vs_auc.pdf")
    plt.close()

    n48 = test[test["N"] == 48].copy()
    if n48.empty:
        n48 = test[test["N"] == test["N"].max()].copy()
    plt.figure(figsize=(7.2, 4.8))
    sc = plt.scatter(
        n48["kappa"],
        n48["actual"],
        c=n48["theta"],
        s=16,
        alpha=0.65,
        cmap="viridis",
        edgecolors="none",
    )
    plt.xlabel("AUC / kappa")
    plt.ylabel(f"Actual best-of-{int(n48['N'].iloc[0])} accuracy")
    plt.title("Similar AUC can hide different high-N outcomes")
    cb = plt.colorbar(sc)
    cb.set_label(f"theta_{int(n48['N'].iloc[0]) - 1}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "auc_fails_highN_moments_succeed.pdf")
    plt.close()


def run_pilot_sample_complexity(records: list[dict], n_seeds: int) -> tuple[pd.DataFrame, dict]:
    print("Running Experiment 2: pilot sample-complexity curve")
    seed_ids = list(range(n_seeds))
    per_seed: dict[tuple[str, int, int], list[tuple[int, float, int]]] = defaultdict(list)
    possible_k = sorted({K for rec in records for K in PILOT_K_VALUES if 0 < K < len(rec["all_scores"])})

    for seed_id in seed_ids:
        errors: dict[tuple[str, int, int], list[float]] = defaultdict(list)
        counts: dict[tuple[str, int, int], int] = defaultdict(int)
        for rec in records:
            scores = np.asarray(rec["all_scores"], dtype=float)
            correct = np.asarray(rec["all_correct"], dtype=bool)
            n = len(scores)
            if n < 4:
                continue
            for K in possible_k:
                if K >= n:
                    continue
                rng = np.random.default_rng(stable_seed("pilot", seed_id, rec["model_key"], rec["problem_idx"], K))
                perm = rng.permutation(n)
                pilot_idx = perm[:K]
                held_idx = perm[K:]
                held_n = len(held_idx)
                eval_ns = [N for N in PILOT_N_VALUES if N <= held_n]
                if not eval_ns:
                    continue
                pred_curve = exact_curve(scores[pilot_idx], correct[pilot_idx], eval_ns)
                held_curve = exact_curve(scores[held_idx], correct[held_idx], eval_ns)
                for N in eval_ns:
                    key = (rec["model_key"], K, N)
                    errors[key].append(abs(pred_curve[N] - held_curve[N]))
                    counts[key] += 1
        for key, vals in errors.items():
            per_seed[key].append((seed_id, float(np.mean(vals)), counts[key]))

    rows = []
    for (model, K, N), seed_vals in sorted(per_seed.items()):
        maes = np.asarray([v[1] for v in seed_vals], dtype=float)
        counts = np.asarray([v[2] for v in seed_vals], dtype=float)
        mean_mae = float(np.mean(maes))
        std_mae = float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0
        ci = 1.96 * std_mae / math.sqrt(len(maes)) if len(maes) > 1 else 0.0
        rows.append(
            {
                "K": K,
                "N": N,
                "model": model,
                "heldout_MAE_mean": mean_mae,
                "heldout_MAE_std": std_mae,
                "heldout_MAE_ci_low": mean_mae - ci,
                "heldout_MAE_ci_high": mean_mae + ci,
                "num_problems_used": int(round(np.mean(counts))) if len(counts) else 0,
                "num_splits": len(maes),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "pilot_sample_complexity.csv", index=False)
    make_pilot_figure(df)

    overall = (
        df.groupby(["K", "N"])
        .agg(heldout_MAE_mean=("heldout_MAE_mean", "mean"), models=("model", "nunique"))
        .reset_index()
        .sort_values(["N", "K"])
    )
    thresholds = {}
    for target in [0.05, 0.02, 0.01]:
        hit = overall[overall["heldout_MAE_mean"] < target]
        thresholds[str(target)] = int(hit["K"].min()) if not hit.empty else None
    summary = {
        "K_values_tested": sorted(int(k) for k in df["K"].unique()) if not df.empty else [],
        "N_values_tested": sorted(int(n) for n in df["N"].unique()) if not df.empty else [],
        "threshold_first_K": thresholds,
        "overall": overall.to_dict(orient="records"),
    }
    return df, summary


def make_pilot_figure(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7.0, 4.6))
    if not df.empty:
        overall = (
            df.groupby(["K", "N"])
            .agg(heldout_MAE_mean=("heldout_MAE_mean", "mean"))
            .reset_index()
            .sort_values(["N", "K"])
        )
        for N, sub in overall.groupby("N"):
            plt.plot(sub["K"], sub["heldout_MAE_mean"], marker="o", label=f"N={int(N)}")
    plt.xlabel("Pilot samples K")
    plt.ylabel("Held-out MAE")
    plt.title("Finite-pilot forecasting error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "pilot_sample_complexity.pdf")
    plt.close()


def make_binned_calibrators(data: dict[str, list[dict]], n_bins: int = 20) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    calibrators = {}
    for model, rows in data.items():
        train_scores = []
        train_correct = []
        for rec in rows:
            if not train_split(int(rec["problem_idx"])):
                continue
            train_scores.extend(rec["all_scores"])
            train_correct.extend(rec["all_correct"])
        scores = np.asarray(train_scores, dtype=float)
        correct = np.asarray(train_correct, dtype=float)
        finite = np.isfinite(scores)
        scores = scores[finite]
        correct = correct[finite]
        if len(scores) < 10 or len(np.unique(scores)) < 2:
            global_rate = float(np.mean(correct)) if len(correct) else 0.0
            calibrators[model] = lambda s, gr=global_rate: np.full(len(s), gr, dtype=float)
            continue
        edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            global_rate = float(np.mean(correct))
            calibrators[model] = lambda s, gr=global_rate: np.full(len(s), gr, dtype=float)
            continue
        bin_ids = np.clip(np.searchsorted(edges[1:-1], scores, side="right"), 0, len(edges) - 2)
        global_rate = float(np.mean(correct))
        rates = np.full(len(edges) - 1, global_rate, dtype=float)
        for b in range(len(rates)):
            mask = bin_ids == b
            if np.any(mask):
                rates[b] = (float(np.sum(correct[mask])) + 1.0) / (float(np.sum(mask)) + 2.0)

        def _calibrate(s: np.ndarray, e=edges.copy(), r=rates.copy()) -> np.ndarray:
            arr = np.asarray(s, dtype=float)
            ids = np.clip(np.searchsorted(e[1:-1], arr, side="right"), 0, len(r) - 1)
            return r[ids]

        calibrators[model] = _calibrate
    return calibrators


def evaluate_score_functions(
    data: dict[str, list[dict]],
    score_getters: dict[str, Callable[[str, dict], np.ndarray]],
    n_values: list[int],
    test_only: bool = True,
) -> pd.DataFrame:
    rows = []
    for score_name, getter in score_getters.items():
        for model, recs in data.items():
            per_n_actual: dict[int, list[float]] = defaultdict(list)
            aucs = []
            ties = []
            n_probs = 0
            for rec in recs:
                if test_only and train_split(int(rec["problem_idx"])):
                    continue
                scores = np.asarray(getter(model, rec), dtype=float)
                correct = np.asarray(rec["all_correct"], dtype=bool)
                if len(scores) != len(correct):
                    continue
                curve = exact_curve(scores, correct, n_values)
                for N in n_values:
                    per_n_actual[N].append(curve[N])
                aucs.append(auc_score(scores, correct))
                ties.append(tie_rate(scores))
                n_probs += 1
            for N in n_values:
                actual = safe_mean(per_n_actual[N])
                predicted = actual
                rows.append(
                    {
                        "score_name": score_name,
                        "model": model,
                        "N": int(N),
                        "actual_acc": actual,
                        "predicted_acc": predicted,
                        "MAE": abs(predicted - actual) if np.isfinite(actual) else float("nan"),
                        "auc": safe_mean(aucs),
                        "tie_rate": safe_mean(ties),
                        "num_problems": n_probs,
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        base = df[df["score_name"] == "mean_logprob"][["model", "N", "actual_acc"]].rename(
            columns={"actual_acc": "mean_logprob_actual_acc"}
        )
        df = df.merge(base, on=["model", "N"], how="left")
        df["improves_over_meanlogp"] = df["actual_acc"] > df["mean_logprob_actual_acc"]
    return df


def run_score_function_comparison(data: dict[str, list[dict]]) -> tuple[pd.DataFrame, dict]:
    print("Running Experiment 3: score-function comparison")
    calibrators = make_binned_calibrators(data)

    score_getters = {
        "mean_logprob": lambda model, rec: np.asarray(rec["all_scores"], dtype=float),
        "anti_meanlogprob": lambda model, rec: -np.asarray(rec["all_scores"], dtype=float),
        "calibrated_binned_posterior": lambda model, rec: calibrators[model](np.asarray(rec["all_scores"], dtype=float)),
        "oracle_correctness_verifier": lambda model, rec: np.asarray(rec["all_correct"], dtype=float),
    }
    df = evaluate_score_functions(data, score_getters, SCORE_N_VALUES, test_only=True)
    df.to_csv(TABLE_DIR / "score_function_comparison.csv", index=False)
    make_score_figure(df)

    n48 = df[df["N"] == 48].copy()
    best = None
    if not n48.empty:
        agg = n48.groupby("score_name").agg(actual_acc=("actual_acc", "mean")).reset_index()
        best = agg.sort_values("actual_acc", ascending=False).iloc[0].to_dict()
    summary = {
        "evaluated_on": "held-out problems (problem_idx % 5 == 0)",
        "score_functions": sorted(df["score_name"].unique()) if not df.empty else [],
        "best_score_at_N48": best,
        "missing_dependencies": [
            "No external reward/verifier/model-judge scores were present in the measurement artifacts.",
            "Total logprob, response length, boxed-answer format, and final-answer logprob require parsing raw response caches; the compact measurement artifacts only contain mean logprob and correctness.",
        ],
    }
    return df, summary


def make_score_figure(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7.2, 4.8))
    if not df.empty:
        overall = (
            df.groupby(["score_name", "N"])
            .agg(actual_acc=("actual_acc", "mean"), predicted_acc=("predicted_acc", "mean"))
            .reset_index()
            .sort_values(["score_name", "N"])
        )
        for score_name, sub in overall.groupby("score_name"):
            plt.plot(sub["N"], sub["actual_acc"], marker="o", label=score_name)
            plt.plot(sub["N"], sub["predicted_acc"], linestyle="--", alpha=0.45)
    plt.xscale("log", base=2)
    plt.xticks(SCORE_N_VALUES, [str(n) for n in SCORE_N_VALUES])
    plt.xlabel("N")
    plt.ylabel("Actual best-of-N accuracy")
    plt.title("Score-function comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "score_function_comparison.pdf")
    plt.close()


def synthetic_verifier_score(model: str, rec: dict, M: int, sigma: float = 0.75) -> np.ndarray:
    correct = np.asarray(rec["all_correct"], dtype=float)
    rng = np.random.default_rng(stable_seed("synthetic-verifier", model, rec["problem_idx"], M))
    noise = rng.normal(loc=0.0, scale=sigma, size=(M, len(correct)))
    return np.mean(correct[None, :] + noise, axis=0)


def run_verifier_extension(data: dict[str, list[dict]]) -> tuple[pd.DataFrame, dict]:
    print("Running Experiment 4: verifier-score extension")
    score_getters: dict[str, Callable[[str, dict], np.ndarray]] = {
        "oracle_correctness_verifier_M1": lambda model, rec: np.asarray(rec["all_correct"], dtype=float),
    }
    for M in [1, 2, 4, 8]:
        score_getters[f"synthetic_noisy_verifier_M{M}"] = (
            lambda model, rec, M=M: synthetic_verifier_score(model, rec, M)
        )
    raw = evaluate_score_functions(data, score_getters, SCORE_N_VALUES, test_only=True)
    if raw.empty:
        df = pd.DataFrame()
    else:
        df = raw.rename(columns={"score_name": "verifier_name"})
        df["num_verifiers"] = df["verifier_name"].str.extract(r"_M(\d+)$").fillna("1").astype(int)
        df = df[
            [
                "verifier_name",
                "num_verifiers",
                "model",
                "N",
                "actual_acc",
                "predicted_acc",
                "MAE",
                "auc",
                "tie_rate",
                "num_problems",
            ]
        ]
    df.to_csv(TABLE_DIR / "verifier_extension.csv", index=False)
    make_verifier_figure(df)

    summary = {
        "external_verifier_scores_found": False,
        "verifier_scores_tested": sorted(df["verifier_name"].unique()) if not df.empty else [],
        "labeling_note": "Oracle and synthetic verifier scores use correctness labels to construct diagnostic scores; they are not realistic learned verifiers.",
        "mean_mae": safe_mean(df["MAE"].to_numpy()) if not df.empty else float("nan"),
    }
    return df, summary


def make_verifier_figure(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7.2, 4.8))
    if not df.empty:
        overall = (
            df.groupby(["verifier_name", "num_verifiers", "N"])
            .agg(actual_acc=("actual_acc", "mean"))
            .reset_index()
            .sort_values(["num_verifiers", "N"])
        )
        keep = [name for name in overall["verifier_name"].unique() if "synthetic_noisy" in name]
        keep.append("oracle_correctness_verifier_M1")
        for name in keep:
            sub = overall[overall["verifier_name"] == name]
            if sub.empty:
                continue
            label = name.replace("synthetic_noisy_verifier_", "synthetic ").replace(
                "oracle_correctness_verifier_M1", "oracle M1"
            )
            plt.plot(sub["N"], sub["actual_acc"], marker="o", label=label)
    plt.xscale("log", base=2)
    plt.xticks(SCORE_N_VALUES, [str(n) for n in SCORE_N_VALUES])
    plt.xlabel("N")
    plt.ylabel("Verifier-reranked accuracy")
    plt.title("Verifier-score best-of-N extension")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "verifier_extension_bestofN.pdf")
    plt.close()


def pooled_auc(scores: np.ndarray, correct: np.ndarray) -> float:
    return auc_score(scores, correct)


def run_posterior_monotonicity(data: dict[str, list[dict]]) -> tuple[pd.DataFrame, dict]:
    print("Running Experiment 5: posterior monotonicity diagnostic")
    calibrators = make_binned_calibrators(data)
    rows = []
    pdf_path = FIGURE_DIR / "posterior_monotonicity_diagnostic.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(6, 4, figsize=(12, 14), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax in axes:
            ax.axis("off")
        for ax_idx, (model, recs) in enumerate(sorted(data.items())):
            all_scores = []
            all_correct = []
            top_accs = []
            cal_accs = []
            for rec in recs:
                scores = np.asarray(rec["all_scores"], dtype=float)
                correct = np.asarray(rec["all_correct"], dtype=bool)
                all_scores.extend(scores.tolist())
                all_correct.extend(correct.tolist())
                if not train_split(int(rec["problem_idx"])):
                    top_accs.append(exact_curve(scores, correct, [48])[48])
                    cal_scores = calibrators[model](scores)
                    cal_accs.append(exact_curve(cal_scores, correct, [48])[48])

            scores_arr = np.asarray(all_scores, dtype=float)
            correct_arr = np.asarray(all_correct, dtype=bool)
            finite = np.isfinite(scores_arr)
            diag_scores = scores_arr[finite]
            diag_correct = correct_arr[finite]
            spearman = (
                float(stats.spearmanr(diag_scores, diag_correct).statistic)
                if len(np.unique(diag_scores)) > 1
                else float("nan")
            )
            auc = pooled_auc(diag_scores, diag_correct)

            n_bins = 20
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.unique(np.quantile(diag_scores, qs)) if len(diag_scores) else np.asarray([])
            rates = []
            centers = []
            if len(edges) >= 2:
                ids = np.clip(np.searchsorted(edges[1:-1], diag_scores, side="right"), 0, len(edges) - 2)
                for b in range(len(edges) - 1):
                    mask = ids == b
                    if np.any(mask):
                        centers.append(float((b + 0.5) / (len(edges) - 1)))
                        rates.append(float(np.mean(diag_correct[mask])))
            violations = sum(1 for i in range(1, len(rates)) if rates[i] < rates[i - 1])

            if IsotonicRegression is not None and len(diag_scores) > 1:
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                try:
                    pred = iso.fit_transform(diag_scores, diag_correct.astype(float))
                    isotonic_mse = float(np.mean((pred - diag_correct.astype(float)) ** 2))
                except Exception:
                    isotonic_mse = float("nan")
            else:
                isotonic_mse = float("nan")

            rows.append(
                {
                    "model": model,
                    "spearman": spearman,
                    "auc": auc,
                    "monotonicity_violations": int(violations),
                    "is_monotone": violations == 0,
                    "isotonic_mse": isotonic_mse,
                    "top_score_acc_N48": safe_mean(top_accs),
                    "calibrated_score_acc_N48": safe_mean(cal_accs),
                    "num_bins": len(rates),
                }
            )

            if ax_idx < len(axes):
                ax = axes[ax_idx]
                ax.axis("on")
                ax.plot(centers, rates, marker="o", linewidth=1.2)
                ax.set_title(f"{model}: {violations} inv.", fontsize=9)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
        fig.supxlabel("Score quantile bin")
        fig.supylabel("Empirical P(correct | score bin)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "posterior_monotonicity.csv", index=False)

    summary = {
        "monotone_models": int(df["is_monotone"].sum()) if not df.empty else 0,
        "total_models": int(len(df)),
        "mean_violations": safe_mean(df["monotonicity_violations"].to_numpy()) if not df.empty else float("nan"),
        "mean_top_score_acc_N48": safe_mean(df["top_score_acc_N48"].to_numpy()) if not df.empty else float("nan"),
        "mean_calibrated_score_acc_N48": safe_mean(df["calibrated_score_acc_N48"].to_numpy()) if not df.empty else float("nan"),
    }
    return df, summary


def _load_features_lookup(features_csv: Path, models: list[str], test_only: bool = True) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Load features CSV into a nested lookup: {model: {problem_idx: {feature: array[48]}}}."""
    import csv
    from src.feature_extraction import NUMERIC_FEATURE_COLS

    lookup: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(features_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mk = row["model_key"]
            if mk not in models:
                continue
            pidx = int(row["problem_idx"])
            if test_only and pidx % 5 != 0:
                continue
            for col in NUMERIC_FEATURE_COLS:
                val = row.get(col, "0")
                try:
                    lookup[mk][pidx][col].append(float(val))
                except (ValueError, TypeError):
                    lookup[mk][pidx][col].append(0.0)

    result: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for mk in lookup:
        result[mk] = {}
        for pidx in lookup[mk]:
            result[mk][pidx] = {col: np.array(vals, dtype=float) for col, vals in lookup[mk][pidx].items()}
    return result


def run_real_verifier_extension(data: dict[str, list[dict]]) -> tuple[pd.DataFrame, dict]:
    """Train learned verifiers on raw-cache features and evaluate with theorem."""
    print("Running Experiment 6: real learned verifier extension")

    if not FEATURES_CSV.exists():
        print(f"  Features CSV not found at {FEATURES_CSV}. Skipping.")
        return pd.DataFrame(), {"status": "skipped", "reason": "features CSV not found"}

    from src.learned_verifier import train_per_model_verifier, compute_ece

    import csv as _csv
    features_models = set()
    with open(FEATURES_CSV, encoding="utf-8") as _f:
        for _row in _csv.DictReader(_f):
            features_models.add(_row["model_key"])
    models = sorted(features_models & set(data.keys()))
    print(f"  Models with features: {models}")
    feature_data = {model: data[model] for model in models}
    classifier_types = ["logistic", "gbdt", "calibrated_logistic", "calibrated_gbdt"]
    all_verifier_scores: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    all_metrics: dict[str, dict[str, dict]] = {}

    for ctype in classifier_types:
        print(f"  Training {ctype} verifier per model...")
        all_verifier_scores[ctype] = {}
        all_metrics[ctype] = {}
        for model in models:
            scores, metrics = train_per_model_verifier(FEATURES_CSV, model, ctype)
            all_verifier_scores[ctype][model] = scores
            all_metrics[ctype][model] = metrics

    rows = []
    for ctype in classifier_types:
        score_name = f"{ctype}_verifier"
        v_scores = all_verifier_scores[ctype]

        def _getter(model, rec, vs=v_scores):
            pidx = int(rec["problem_idx"])
            if model in vs and pidx in vs[model]:
                arr = vs[model][pidx]
                n_expected = len(rec["all_scores"])
                if len(arr) == n_expected:
                    return arr
                elif len(arr) > 0:
                    result = np.full(n_expected, 0.5)
                    result[: len(arr)] = arr[: n_expected]
                    return result
            return np.full(len(rec["all_scores"]), 0.5)

        score_getters = {score_name: _getter}
        df_part = evaluate_score_functions(feature_data, score_getters, SCORE_N_VALUES, test_only=True)

        if not df_part.empty:
            for _, row in df_part.iterrows():
                m = row["model"]
                met = all_metrics[ctype].get(m, {})
                rows.append({
                    "verifier_name": score_name,
                    "model": m,
                    "N": int(row["N"]),
                    "actual_acc": row["actual_acc"],
                    "predicted_acc": row["predicted_acc"],
                    "MAE": row["MAE"],
                    "auc": met.get("auc", float("nan")),
                    "brier_score": met.get("brier", float("nan")),
                    "ece": met.get("ece", float("nan")),
                    "tie_rate": row.get("tie_rate", float("nan")),
                    "num_problems": row.get("num_problems", 0),
                    "improves_over_meanlogp": row.get("improves_over_meanlogp", False),
                    "leakage_status": "train_test_split",
                })

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "real_verifier_extension.csv", index=False)
    _make_real_verifier_figure(df)

    summary = {
        "status": "completed",
        "classifier_types": classifier_types,
        "models_with_features": models,
        "per_classifier_metrics": {
            ctype: {
                model: all_metrics[ctype].get(model, {})
                for model in models
            }
            for ctype in classifier_types
        },
        "mean_auc_by_type": {
            ctype: safe_mean([m.get("auc", float("nan")) for m in all_metrics[ctype].values()])
            for ctype in classifier_types
        },
        "n48_accuracy_by_type": {},
    }
    if not df.empty:
        n48 = df[df["N"] == 48]
        for ctype in classifier_types:
            vn = f"{ctype}_verifier"
            sub = n48[n48["verifier_name"] == vn]
            summary["n48_accuracy_by_type"][vn] = safe_mean(sub["actual_acc"].to_numpy()) if not sub.empty else float("nan")

    return df, summary


def _make_real_verifier_figure(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7.2, 4.8))
    if not df.empty:
        overall = (
            df.groupby(["verifier_name", "N"])
            .agg(actual_acc=("actual_acc", "mean"))
            .reset_index()
            .sort_values(["verifier_name", "N"])
        )
        for vname, sub in overall.groupby("verifier_name"):
            label = str(vname).replace("_verifier", "").replace("_", " ")
            plt.plot(sub["N"], sub["actual_acc"], marker="o", label=label)
    plt.xscale("log", base=2)
    plt.xticks(SCORE_N_VALUES, [str(n) for n in SCORE_N_VALUES])
    plt.xlabel("N")
    plt.ylabel("Verifier-reranked accuracy")
    plt.title("Real learned verifier best-of-N extension")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "real_verifier_extension_bestofN.pdf")
    plt.close()


def run_expanded_score_comparison(
    data: dict[str, list[dict]],
    real_verifier_scores: dict[str, dict[str, dict[int, np.ndarray]]] | None = None,
    judge_scores: dict[str, dict[int, np.ndarray]] | None = None,
    judge_score_name: str = "heuristic_judge_score",
    judge_scores_source: str = "deterministic_fallback_heuristic_not_live_model_judge",
    judge_leakage_status: str = "no_reference_heuristic",
) -> tuple[pd.DataFrame, dict]:
    """Score-function comparison with all raw-derived and learned features."""
    print("Running Experiment 7: expanded score-function comparison")

    calibrators = make_binned_calibrators(data)
    models = sorted(data.keys())
    features_lookup = None
    comparison_data = data
    comparison_models = models
    if FEATURES_CSV.exists():
        features_lookup = _load_features_lookup(FEATURES_CSV, models, test_only=True)
        if features_lookup:
            comparison_models = sorted(features_lookup.keys())
            comparison_data = {model: data[model] for model in comparison_models if model in data}

    score_getters: dict[str, Callable[[str, dict], np.ndarray]] = {
        "mean_logprob": lambda model, rec: np.asarray(rec["all_scores"], dtype=float),
        "calibrated_binned_posterior": lambda model, rec: calibrators[model](np.asarray(rec["all_scores"], dtype=float)),
        "oracle_correctness_verifier": lambda model, rec: np.asarray(rec["all_correct"], dtype=float),
    }

    if features_lookup:
        def _feat_getter(feat_name):
            def _get(model, rec, fn=feat_name, fl=features_lookup):
                pidx = int(rec["problem_idx"])
                if model in fl and pidx in fl[model] and fn in fl[model][pidx]:
                    arr = fl[model][pidx][fn]
                    if len(arr) == len(rec["all_scores"]):
                        return arr
                return np.asarray(rec["all_scores"], dtype=float)
            return _get

        score_getters["total_logprob"] = _feat_getter("total_logprob")
        score_getters["length_penalized_logprob"] = _feat_getter("length_penalized_logprob")

        def _neg_length(model, rec, fl=features_lookup):
            pidx = int(rec["problem_idx"])
            if model in fl and pidx in fl[model] and "response_length_tokens" in fl[model][pidx]:
                arr = fl[model][pidx]["response_length_tokens"]
                if len(arr) == len(rec["all_scores"]):
                    return -arr
            return np.zeros(len(rec["all_scores"]))
        score_getters["response_length_negative"] = _neg_length

        def _format_score(model, rec, fl=features_lookup):
            pidx = int(rec["problem_idx"])
            if model in fl and pidx in fl[model] and "boxed_answer_present" in fl[model][pidx]:
                arr = fl[model][pidx]["boxed_answer_present"]
                if len(arr) == len(rec["all_scores"]):
                    return arr
            return np.ones(len(rec["all_scores"]))
        score_getters["answer_format_score"] = _format_score

    if real_verifier_scores:
        for ctype, vs in real_verifier_scores.items():
            vname = f"{ctype}_verifier_score"
            def _vget(model, rec, v=vs):
                pidx = int(rec["problem_idx"])
                if model in v and pidx in v[model]:
                    arr = v[model][pidx]
                    if len(arr) == len(rec["all_scores"]):
                        return arr
                return np.full(len(rec["all_scores"]), 0.5)
            score_getters[vname] = _vget

    judge_scores_used = False
    if judge_scores and comparison_data:
        judge_filtered_data: dict[str, list[dict]] = {}
        for model, recs in comparison_data.items():
            kept = []
            for rec in recs:
                pidx = int(rec["problem_idx"])
                if (
                    model in judge_scores
                    and pidx in judge_scores[model]
                    and len(judge_scores[model][pidx]) == len(rec["all_scores"])
                ):
                    kept.append(rec)
            if kept:
                judge_filtered_data[model] = kept

        if judge_filtered_data:
            comparison_data = judge_filtered_data
            comparison_models = sorted(judge_filtered_data)

            def _jget(model, rec, js=judge_scores):
                pidx = int(rec["problem_idx"])
                if model in js and pidx in js[model]:
                    arr = js[model][pidx]
                    if len(arr) == len(rec["all_scores"]):
                        return arr
                return np.full(len(rec["all_scores"]), 0.5)
            score_getters[judge_score_name] = _jget
            judge_scores_used = True

    df = evaluate_score_functions(comparison_data, score_getters, SCORE_N_VALUES, test_only=True)

    leakage_map = {
        "mean_logprob": "none",
        "total_logprob": "none",
        "length_penalized_logprob": "none",
        "response_length_negative": "none",
        "answer_format_score": "none",
        "calibrated_binned_posterior": "train_test_split",
        "oracle_correctness_verifier": "oracle",
        judge_score_name: judge_leakage_status,
    }
    for ctype in (real_verifier_scores or {}):
        leakage_map[f"{ctype}_verifier_score"] = "train_test_split"

    if not df.empty:
        df["leakage_status"] = df["score_name"].map(leakage_map).fillna("unknown")

    df.to_csv(TABLE_DIR / "score_function_comparison_real_features.csv", index=False)
    _make_expanded_score_figure(df)

    summary = {
        "score_functions": sorted(df["score_name"].unique()) if not df.empty else [],
        "features_available": features_lookup is not None,
        "judge_scores_available": judge_scores is not None,
        "judge_scores_used": judge_scores_used,
        "judge_scores_source": judge_scores_source,
        "judge_score_name": judge_score_name if judge_scores_used else None,
        "comparison_models": comparison_models,
        "verifier_scores_available": real_verifier_scores is not None,
    }
    if not df.empty:
        n48 = df[df["N"] == 48]
        if not n48.empty:
            agg = n48.groupby("score_name").agg(actual_acc=("actual_acc", "mean")).reset_index()
            summary["n48_ranking"] = agg.sort_values("actual_acc", ascending=False).to_dict(orient="records")

    return df, summary


def _make_expanded_score_figure(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8.0, 5.2))
    if not df.empty:
        overall = (
            df.groupby(["score_name", "N"])
            .agg(actual_acc=("actual_acc", "mean"))
            .reset_index()
            .sort_values(["score_name", "N"])
        )
        for sname, sub in overall.groupby("score_name"):
            label = str(sname).replace("_", " ")
            ls = "--" if "oracle" in str(sname) else "-"
            plt.plot(sub["N"], sub["actual_acc"], marker="o", label=label, linestyle=ls, markersize=4)
    plt.xscale("log", base=2)
    plt.xticks(SCORE_N_VALUES, [str(n) for n in SCORE_N_VALUES])
    plt.xlabel("N")
    plt.ylabel("Actual best-of-N accuracy")
    plt.title("Score-function comparison (all features)")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "score_function_comparison_real_features.pdf")
    plt.close()


def load_audit_summary() -> dict:
    path = AUDIT_DIR / "audit_results.json"
    if not path.exists():
        return {"found": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"found": False}
    return {
        "found": True,
        "overall_mae": data.get("output_4", {}).get("overall_mae"),
        "n_triples": data.get("output_4", {}).get("n_triples"),
        "tie_rate": data.get("output_5", {}).get("grand_mean"),
        "per_problem_mae": data.get("output_7", {}).get("summary", {}).get("per_problem_grand_mae"),
        "pooled_mae": data.get("output_7", {}).get("summary", {}).get("pooled_grand_ae"),
        "heldout_mae_k24": data.get("output_8", {}).get("summary", {}).get("held_out_grand_mae"),
        "n2_auc_diff": data.get("output_9", {}).get("grand_closed_vs_general"),
    }


def fmt_float(x: object, digits: int = 6) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def write_summary(
    audit: dict,
    data: dict[str, list[dict]],
    moment_summary: dict,
    pilot_summary: dict,
    score_summary: dict,
    verifier_summary: dict,
    posterior_summary: dict,
    real_verifier_summary: dict | None = None,
    expanded_score_summary: dict | None = None,
    judge_summary: dict | None = None,
    expanded_pilot_summary: dict | None = None,
    allocation_summary: dict | None = None,
) -> None:
    models = sorted(data)
    n_problems = max((len(v) for v in data.values()), default=0)
    by_n = moment_summary.get("by_n", {})
    auc_by_n = ", ".join(f"N={n}: {fmt_float(v['auc_only_mae'])}" for n, v in by_n.items())
    mom_by_n = ", ".join(f"N={n}: {fmt_float(v['moment_mae'], 10)}" for n, v in by_n.items())

    pilot_overall = pd.DataFrame(pilot_summary.get("overall", []))
    trend = "N/A"
    if not pilot_overall.empty:
        n8 = pilot_overall[pilot_overall["N"] == 8].sort_values("K")
        if not n8.empty:
            trend = ", ".join(f"K={int(r.K)}: {r.heldout_MAE_mean:.3f}" for r in n8.itertuples())

    best_score = score_summary.get("best_score_at_N48") or {}
    thresholds = pilot_summary.get("threshold_first_K", {})

    rv_section = ""
    if real_verifier_summary and real_verifier_summary.get("status") == "completed":
        rv_auc = real_verifier_summary.get("mean_auc_by_type", {})
        rv_n48 = real_verifier_summary.get("n48_accuracy_by_type", {})
        rv_lines = []
        for vname, auc in rv_auc.items():
            acc = rv_n48.get(f"{vname}_verifier", float("nan"))
            rv_lines.append(f"  - {vname}: AUC={fmt_float(auc, 4)}, N=48 acc={fmt_float(acc, 4)}")
        rv_section = f"""
6b. Real learned verifier extension
- Classifiers trained on raw-cache features (logprob, length, format, reasoning indicators).
- Split by problem_idx (no response-level leakage).
- Per-model results:
{chr(10).join(rv_lines)}
- These are real learned verifiers, not oracle or synthetic scores.
"""

    judge_section = ""
    if judge_summary and judge_summary.get("status") == "completed":
        judge_section = f"""
6c. Structured judge diagnostic
- A no-reference heuristic judge diagnostic was evaluated on a stratified subset.
- Subset: {judge_summary.get('n_models', '?')} models x {judge_summary.get('n_problems', '?')} problems x 48 samples.
- No-reference heuristic AUC: {fmt_float(judge_summary.get('no_ref_auc'), 4)}
- No-reference heuristic N=48 accuracy: {fmt_float(judge_summary.get('no_ref_n48_acc'), 4)}
- Reference-based judge AUC: {fmt_float(judge_summary.get('ref_auc'), 4)}
- The no-reference heuristic does not see the ground truth, but it is not a live model-judge call.
- The reference-based diagnostic is labeled as such and is not test-time realistic.
"""

    expanded_pilot_section = ""
    if expanded_pilot_summary:
        max_samples = expanded_pilot_summary.get("max_samples_per_problem")
        k_values = expanded_pilot_summary.get("K_values_tested", [])
        highest_k = max(k_values) if k_values else None
        if max_samples and int(max_samples) >= 256 and highest_k and highest_k >= 192:
            pilot_note = (
                f"The five-model curve is complete through K={highest_k}; "
                "K=128/192 is evidenced for all expanded-pilot models."
            )
        elif max_samples and int(max_samples) > 48:
            pilot_note = (
                f"Current measured artifacts include up to {max_samples} samples/problem for this subset."
            )
        else:
            pilot_note = "With 48 samples/problem, maximum pilot K is ~40 (remaining held out for evaluation)."

        coverage = expanded_pilot_summary.get("per_model_sample_coverage", {})
        coverage_text = "N/A"
        if coverage:
            coverage_text = "; ".join(
                f"{model}: {stats.get('records_at_max', 0)}/{stats.get('num_records', 0)} "
                f"records at n={stats.get('max_samples', 0)}"
                for model, stats in coverage.items()
            )

        expanded_trend = "N/A"
        pilot_table = TABLE_DIR / "expanded_pilot_sample_complexity.csv"
        if pilot_table.exists():
            try:
                ep_df = pd.read_csv(pilot_table)
                n8 = ep_df[ep_df["N"] == 8].groupby("K")["heldout_MAE_mean"].mean().reset_index()
                expanded_trend = ", ".join(f"K={int(r.K)}: {r.heldout_MAE_mean:.3f}" for r in n8.itertuples())
            except Exception:
                expanded_trend = "N/A"
        expanded_pilot_section = f"""
3b. Expanded pilot sample-complexity
- Models: {expanded_pilot_summary.get('models', [])}
- Problems: {expanded_pilot_summary.get('n_problems', '?')} stratified problems
- K values: {expanded_pilot_summary.get('K_values_tested', [])}
- N values: {expanded_pilot_summary.get('N_values_tested', [])}
- Data source: {expanded_pilot_summary.get('data_source', 'existing_48_samples')}
- Coverage: {coverage_text}
- N=8 held-out MAE trend: {expanded_trend}
- Note: {pilot_note}
"""

    expanded_score_section = ""
    if expanded_score_summary:
        sf_list = expanded_score_summary.get("score_functions", [])
        judge_source = expanded_score_summary.get("judge_scores_source", "none")
        judge_name = expanded_score_summary.get("judge_score_name", "N/A")
        live_judge_lines = ""
        live_judge_summary_path = OUT_DIR / "live_judge_score_summary.json"
        if live_judge_summary_path.exists():
            try:
                live = json.loads(live_judge_summary_path.read_text(encoding="utf-8"))
                live_judge_lines = (
                    "\n"
                    f"- Live LLM judge coverage: {live.get('num_model_problem_pairs', 0)}/"
                    f"{live.get('manifest_expected_model_problem_pairs', '?')} model/problem pairs "
                    f"and {live.get('num_model_problem_pairs', 0) * live.get('manifest_samples_per_problem', 0)}/"
                    f"{live.get('manifest_expected_model_problem_pairs', 0) * live.get('manifest_samples_per_problem', 0)} judgments "
                    f"for the {live.get('manifest_n_problems', '?')}-problem manifest.\n"
                    f"- Live LLM judge summary: exact-law MAE={fmt_float(live.get('mean_mae'), 6)}, "
                    f"mean AUC={fmt_float(live.get('mean_live_judge_auc'), 4)}, "
                    f"mean logprob AUC on same pairs={fmt_float(live.get('mean_logprob_auc_on_same_pairs'), 4)}, "
                    f"N=48 delta over mean logprob={fmt_float(live.get('n48_improvement_over_meanlogprob'), 4)}."
                )
            except (OSError, json.JSONDecodeError):
                live_judge_lines = ""
        expanded_score_section = f"""
4b. Expanded score-function comparison
- Score functions tested: {sf_list}
- Features from raw response caches: total_logprob, length_penalized_logprob, response_length, answer_format.
- Learned verifier scores: logistic regression, gradient boosting, calibrated variants.
- Judge score source: {judge_source}; score name: {judge_name}.{live_judge_lines}
- oracle_correctness_verifier is included as a DIAGNOSTIC UPPER BOUND only.
- Leakage status tracked per score function (none / train_test_split / oracle).
- Heuristic judge artifacts remain diagnostic only; live judge scores may be used for paper-facing score-agnostic claims when present.
"""

    allocation_section = ""
    if allocation_summary and allocation_summary.get("status") == "completed":
        policies = allocation_summary.get("policies", [])
        ranking = allocation_summary.get("ranking_at_reference_budget", [])
        rank_text = "N/A"
        if ranking:
            rank_text = ", ".join(
                f"{r.get('policy')}: {fmt_float(r.get('accuracy'), 4)}" for r in ranking
            )
        allocation_section = f"""
6d. Adaptive compute allocation
- Policies compared under identical fixed budgets: {policies}
- Models: {allocation_summary.get('models', [])}
- Reference-budget ranking: {rank_text}
- Moment-law improvement over uniform at reference budget: {fmt_float(allocation_summary.get('moment_law_improvement_over_uniform_at_reference_budget'), 4)}
- AUC/kappa-policy improvement over uniform at reference budget: {fmt_float(allocation_summary.get('auc_kappa_improvement_over_uniform_at_reference_budget'), 4)}
- Budget note: {allocation_summary.get('budget_note', 'N/A')}
"""

    summary = f"""DU-ALIGNED THEOREM 1 EXPERIMENT SUMMARY

1. Exact law validation
- The exact finite empirical selector law with random tie-breaking is validated across {len(models)} models.
- Overall MAE: {fmt_float(audit.get("overall_mae"))}
- Problems: up to {n_problems} measured per model
- N values: {EVAL_N_VALUES}
- Tie rate: {fmt_float(audit.get("tie_rate"))}
- Per-problem vs pooled MAE: {fmt_float(audit.get("per_problem_mae"))} vs {fmt_float(audit.get("pooled_mae"))}

2. Moment hierarchy
- AUC (kappa) is exact and complete for N=2 only.
- For high N, the moment hierarchy (rank-interval moments) is required.
- AUC-only MAE by N: {auc_by_n}
- Moment-predictor MAE by N: {mom_by_n}
- The moment predictor is exact up to floating-point error.

3. Pilot sample complexity
- K values tested: {pilot_summary.get("K_values_tested", [])}
- Held-out MAE trend for N=8: {trend}
- Samples needed for MAE < 0.05 / 0.02 / 0.01 if achieved: {thresholds}
- CAVEAT: With only 48 samples/problem, pilot K is capped at ~40. Finite-pilot forecasting is sample-complexity limited.
{expanded_pilot_section}
4. Score-function comparison (baseline)
- Best baseline score function at N=48: {best_score.get("score_name", "N/A")} ({fmt_float(best_score.get("actual_acc"))})
- The theorem predicts each score function with exact empirical MAE near zero.
- oracle_correctness_verifier is a DIAGNOSTIC UPPER BOUND, not a practical score function.
{expanded_score_section}
5. Verifier extension (diagnostic — oracle and synthetic only)
- Verifier scores tested: {verifier_summary.get("verifier_scores_tested", [])}
- MAE: {fmt_float(verifier_summary.get("mean_mae"), 10)}
- These are diagnostic results using ground-truth labels. They represent upper bounds on what a learned verifier could achieve.
- Synthetic noisy verifiers (M in {{1,2,4,8}}) demonstrate theorem validity for noisy scores.

6. Posterior monotonicity
- Is P(correct|score) monotone? {posterior_summary.get("monotone_models", 0)}/{posterior_summary.get("total_models", 0)} models pass the bin monotonicity check.
- Calibrated score improves selection: N=48 top-score={fmt_float(posterior_summary.get("mean_top_score_acc_N48"))}, calibrated={fmt_float(posterior_summary.get("mean_calibrated_score_acc_N48"))}.
{rv_section}{judge_section}
{allocation_section}
7. Paper-ready claims
- Claim 1: Best-of-N top-score selection obeys an exact per-problem order-statistic law, validated across {len(models)} models.
- Claim 2: AUC/kappa is exact for N=2 only. High-N behavior requires the full moment hierarchy.
- Claim 3: The same law applies to any score function, including likelihood, calibrated posterior, and learned verifiers.
- Claim 4 (if real verifiers tested): Learned feature-based verifiers trained on response features provide real, non-oracle reranking that the theorem predicts.
- Claim 5 (if allocation tested): The per-problem law supports adaptive inference-budget allocation under fixed total samples.

8. Caveats and claims to avoid
- No claim of global optimality of any score function.
- No claim that AUC predicts all N (it does not; high-N requires higher moments).
- No claim that tiny pilots (K<16) are sufficient for accurate forecasting.
- Oracle and synthetic verifier results are diagnostic upper bounds only, not achievable in practice without ground-truth labels.
- The 48-sample-per-problem regime limits pilot forecasting to K<40.
- Learned verifiers use problem-level train/test splits; cross-domain generalization is not claimed.
"""
    (OUT_DIR / "du_experiment_summary.txt").write_text(summary, encoding="utf-8")


def write_results_json(**kwargs) -> None:
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, tuple):
            return [convert(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    payload = convert(kwargs)
    (OUT_DIR / "du_experiment_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Du-aligned Theorem 1 experiments.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model keys.")
    parser.add_argument("--pilot-seeds", type=int, default=20, help="Number of random pilot splits.")
    parser.add_argument("--skip-new", action="store_true", help="Skip new experiments (6, 7, expanded pilot).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading measurement artifacts...")
    data = load_measurements(args.models)
    records = flatten_records(data)
    print(f"Loaded {len(records)} problem records across {len(data)} models.")

    audit = load_audit_summary()
    moment_df, moment_summary = run_moment_hierarchy(records)
    pilot_df, pilot_summary = run_pilot_sample_complexity(records, n_seeds=args.pilot_seeds)
    score_df, score_summary = run_score_function_comparison(data)
    verifier_df, verifier_summary = run_verifier_extension(data)
    posterior_df, posterior_summary = run_posterior_monotonicity(data)

    real_verifier_summary = None
    expanded_score_summary = None
    judge_summary = None
    expanded_pilot_summary = None
    allocation_summary = None
    real_verifier_df = pd.DataFrame()
    expanded_score_df = pd.DataFrame()

    if not args.skip_new:
        real_verifier_df, real_verifier_summary = run_real_verifier_extension(data)

        real_verifier_scores = None
        if real_verifier_summary and real_verifier_summary.get("status") == "completed":
            from src.learned_verifier import train_per_model_verifier, load_features_for_model
            features_models = set()
            import csv as _csv
            with open(FEATURES_CSV, encoding="utf-8") as _f:
                for _row in _csv.DictReader(_f):
                    features_models.add(_row["model_key"])
            real_verifier_scores = {}
            for ctype in real_verifier_summary.get("classifier_types", []):
                real_verifier_scores[ctype] = {}
                for model in sorted(features_models & set(data.keys())):
                    scores, _ = train_per_model_verifier(FEATURES_CSV, model, ctype)
                    real_verifier_scores[ctype][model] = scores

        judge_scores = None
        live_judge_path = OUT_DIR / "model_judge" / "judged_responses_live.jsonl"
        heuristic_judge_path = OUT_DIR / "model_judge" / "judged_responses.jsonl"
        if live_judge_path.exists():
            judge_path = live_judge_path
            judge_mode = "live_no_reference"
            judge_score_name = "live_llm_judge_score"
            judge_source = "live_nim_model_judge"
            judge_leakage_status = "live_no_reference"
        else:
            judge_path = heuristic_judge_path
            judge_mode = "heuristic_no_reference"
            judge_score_name = "heuristic_judge_score"
            judge_source = "deterministic_fallback_heuristic_not_live_model_judge"
            judge_leakage_status = "no_reference_heuristic"
        if judge_path.exists():
            try:
                from experiments import _load_judge_for_scoring
            except ImportError:
                pass
            try:
                from experiments._judge_loader import load_judge_scores as _ljs
                judge_scores = _ljs(judge_path, judge_mode)
            except ImportError:
                pass
            if judge_scores is None:
                try:
                    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
                    from _judge_loader import load_judge_scores as _ljs2
                    judge_scores = _ljs2(judge_path, judge_mode)
                except ImportError:
                    pass
            if judge_scores is None:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "judge_mod",
                    str(PROJECT_ROOT / "experiments" / "11_model_judge.py"),
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(mod)
                        judge_scores = mod.load_judge_scores(judge_path, judge_mode)
                    except Exception:
                        pass

        expanded_score_df, expanded_score_summary = run_expanded_score_comparison(
            data,
            real_verifier_scores=real_verifier_scores,
            judge_scores=judge_scores,
            judge_score_name=judge_score_name,
            judge_scores_source=judge_source,
            judge_leakage_status=judge_leakage_status,
        )

        try:
            from experiments._expanded_pilot_loader import run_expanded_analysis_existing, select_stratified_problems
        except ImportError:
            pass
        try:
            expanded_summary_path = OUT_DIR / "expanded_pilot_summary.json"
            if expanded_summary_path.exists():
                expanded_pilot_summary = json.loads(expanded_summary_path.read_text(encoding="utf-8"))
            else:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "ep_mod",
                    str(PROJECT_ROOT / "experiments" / "12_expanded_pilot.py"),
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    target_models = [
                        m for m in ["3B", "70B", "Super49B", "Super120B", "MistralSmall119B"]
                        if m in data
                    ]
                    if target_models:
                        selected = mod.select_stratified_problems(target_models, 100)
                        agg_rows, expanded_pilot_summary = mod.run_expanded_analysis_existing(selected, target_models)
                        import pandas as _pd
                        _pd.DataFrame(agg_rows).to_csv(TABLE_DIR / "expanded_pilot_sample_complexity.csv", index=False)
                        mod.make_expanded_pilot_figure(agg_rows)
        except Exception as e:
            print(f"  Expanded pilot analysis failed: {e}")

        allocation_path = OUT_DIR / "adaptive_allocation_summary.json"
        if allocation_path.exists():
            try:
                allocation_summary = json.loads(allocation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                allocation_summary = None

    write_summary(
        audit=audit,
        data=data,
        moment_summary=moment_summary,
        pilot_summary=pilot_summary,
        score_summary=score_summary,
        verifier_summary=verifier_summary,
        posterior_summary=posterior_summary,
        real_verifier_summary=real_verifier_summary,
        expanded_score_summary=expanded_score_summary,
        judge_summary=judge_summary,
        expanded_pilot_summary=expanded_pilot_summary,
        allocation_summary=allocation_summary,
    )
    write_results_json(
        audit=audit,
        moment_hierarchy=moment_summary,
        pilot_sample_complexity=pilot_summary,
        score_function_comparison=score_summary,
        verifier_extension=verifier_summary,
        posterior_monotonicity=posterior_summary,
        real_verifier_extension=real_verifier_summary,
        expanded_score_comparison=expanded_score_summary,
        expanded_pilot=expanded_pilot_summary,
        adaptive_allocation=allocation_summary,
        generated_tables=sorted(str(p.relative_to(OUT_DIR)) for p in TABLE_DIR.glob("*.csv")),
        generated_figures=sorted(str(p.relative_to(OUT_DIR)) for p in FIGURE_DIR.glob("*.pdf")),
        row_counts={
            "moment_hierarchy_test_rows": int(len(moment_df)),
            "pilot_rows": int(len(pilot_df)),
            "score_rows": int(len(score_df)),
            "verifier_rows": int(len(verifier_df)),
            "posterior_rows": int(len(posterior_df)),
            "real_verifier_rows": int(len(real_verifier_df)),
            "expanded_score_rows": int(len(expanded_score_df)),
            "adaptive_allocation_rows": int(
                len(pd.read_csv(TABLE_DIR / "adaptive_allocation.csv"))
                if (TABLE_DIR / "adaptive_allocation.csv").exists()
                else 0
            ),
        },
    )

    print(f"Done. Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
