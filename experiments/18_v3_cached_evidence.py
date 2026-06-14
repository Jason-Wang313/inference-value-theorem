"""Build v3 cached-evidence summaries for the score-rank LLM paper.

This script is intentionally offline: it reads checked-in result artifacts and
does not call model APIs. Its job is to turn the existing cache into compact
reviewer-facing tables, figures, and LaTeX macros.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


DU = ROOT / "results" / "du_aligned"
MAXED = ROOT / "results" / "maxed_out"
BENCH = ROOT / "results" / "benchmarks"
OUT = ROOT / "results" / "v3_cached_evidence"
PAPER = ROOT / "paper" / "iclr2027"
PAPER_FIG = PAPER / "figures"

HELDOUT_NS = [1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512, 1024]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def finite_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(val):
        return "NA"
    if abs(val) < 1e-3 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{digits}f}"


def macro(name: str, value: Any) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def exact_curve(scores: np.ndarray, correct: np.ndarray, ns: list[int]) -> dict[int, float]:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(scores)
    if n == 0:
        return {draws: 0.0 for draws in ns}
    p = float(np.mean(correct))
    if p == 0.0:
        return {draws: 0.0 for draws in ns}
    if p == 1.0:
        return {draws: 1.0 for draws in ns}

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    totals = {draws: p if draws == 1 else 0.0 for draws in ns}
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        c_g = int(np.sum(sorted_correct[i:j]))
        if c_g:
            k_g = j - i
            r_min = i + 1
            r_max = j
            correctness = c_g / k_g
            for draws in ns:
                if draws == 1:
                    continue
                totals[draws] += correctness * ((r_max / n) ** draws - ((r_min - 1) / n) ** draws)
        i = j
    return {draws: float(value) for draws, value in totals.items()}


def summarize_heldout_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    meas_dir = MAXED / "heldout_forecasting" / "measurements" / "3B"
    for path in sorted(meas_dir.glob("problem_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        rec = load_json(path)
        n_samples = int(rec.get("n_samples", 0) or len(rec.get("all_scores", [])))
        if n_samples < 4096:
            continue
        scores = np.asarray(rec["all_scores"], dtype=float)
        correct = np.asarray(rec["all_correct"], dtype=bool)
        row: dict[str, Any] = {
            "problem_idx": int(rec["problem_idx"]),
            "n_samples": n_samples,
            "p": float(rec["p"]),
            "kappa": float(rec["kappa"]) if rec.get("kappa") is not None else float("nan"),
            "n_correct": int(rec["n_correct"]),
            "n_incorrect": int(rec["n_incorrect"]),
            "level": rec.get("level"),
            "type": rec.get("type"),
        }
        curve = exact_curve(scores, correct, HELDOUT_NS)
        for n in HELDOUT_NS:
            row[f"f_N{n}"] = curve[n]
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_rows = []
    for col in ["p", "kappa", "f_N8", "f_N48", "f_N128", "f_N256", "f_N1024"]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        summary_rows.append(
            {
                "quantity": col,
                "count": int(vals.count()),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "q10": float(vals.quantile(0.10)),
                "median": float(vals.median()),
                "q90": float(vals.quantile(0.90)),
                "max": float(vals.max()),
            }
        )
    return df, pd.DataFrame(summary_rows)


def summarize_gates() -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    report = load_json(MAXED / "claim_gate_report.json")
    rows = []
    for gate in report["gates"]:
        value = gate.get("value")
        if isinstance(value, (dict, list)):
            compact = json.dumps(value, sort_keys=True)
            if len(compact) > 220:
                compact = compact[:217] + "..."
        else:
            compact = value
        rows.append(
            {
                "name": gate["name"],
                "category": gate.get("category"),
                "claim_blocking": bool(gate.get("claim_blocking")),
                "status": gate["status"],
                "value": compact,
                "threshold": gate.get("threshold"),
                "requirement": gate.get("requirement"),
            }
        )
    df = pd.DataFrame(rows)
    counts = df["status"].value_counts().to_dict()
    return df, {str(k): int(v) for k, v in counts.items()}, report


def summarize_cross_benchmarks() -> pd.DataFrame:
    df = pd.read_csv(BENCH / "cross_benchmark_summary.csv")
    df["acc_gain_N48_over_N1"] = df["acc_N48"] - df["acc_N1"]
    df["records_per_family"] = df["measurement_records"].astype(int)
    return df


def summarize_pilots() -> pd.DataFrame:
    df = pd.read_csv(DU / "tables" / "expanded_pilot_sample_complexity.csv")
    grouped = (
        df.groupby(["K", "N"], as_index=False)
        .agg(
            heldout_MAE_mean=("heldout_MAE_mean", "mean"),
            heldout_MAE_ci_low=("heldout_MAE_ci_low", "mean"),
            heldout_MAE_ci_high=("heldout_MAE_ci_high", "mean"),
            models=("model", "nunique"),
        )
        .sort_values(["K", "N"])
    )
    return grouped


def summarize_moment_table(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for n, vals in results["moment_hierarchy"]["by_n"].items():
        rows.append(
            {
                "N": int(n),
                "auc_only_mae": float(vals["auc_only_mae"]),
                "moment_mae": float(vals["moment_mae"]),
                "auc_only_mae_ci_low": float(vals["auc_only_mae_ci95"][0]),
                "auc_only_mae_ci_high": float(vals["auc_only_mae_ci95"][1]),
            }
        )
    return pd.DataFrame(rows).sort_values("N")


def copy_existing_figures() -> None:
    for name in [
        "moment_hierarchy_vs_auc.pdf",
        "expanded_pilot_sample_complexity.pdf",
        "live_judge_score_comparison.pdf",
        "adaptive_allocation_accuracy.pdf",
        "auc_fails_highN_moments_succeed.pdf",
    ]:
        src = DU / "figures" / name
        if src.exists():
            (PAPER_FIG / name).write_bytes(src.read_bytes())


def plot_heldout_profile(heldout: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    axes[0].hist(heldout["p"], bins=18, color="#4c78a8", alpha=0.86)
    axes[0].set_title("Base accuracy p")
    axes[0].set_xlabel("p over 4096 samples")
    axes[0].set_ylabel("problem count")

    axes[1].scatter(heldout["p"], heldout["f_N256"], s=18, alpha=0.72, color="#f58518")
    axes[1].plot([0, 1], [0, 1], color="#555555", linewidth=1, linestyle="--")
    axes[1].set_title("Selection lift at N=256")
    axes[1].set_xlabel("base p")
    axes[1].set_ylabel("exact selected accuracy")

    axes[2].hist(heldout["kappa"], bins=18, color="#54a24b", alpha=0.86)
    axes[2].set_title("AUC/kappa distribution")
    axes[2].set_xlabel("kappa")
    axes[2].set_ylabel("problem count")
    for ax in axes:
        ax.grid(True, alpha=0.18)
    fig.tight_layout()
    for target in [OUT / "v3_heldout_slice_profile.pdf", PAPER_FIG / "v3_heldout_slice_profile.pdf"]:
        fig.savefig(target)
    plt.close(fig)


def plot_gate_status(gates: pd.DataFrame) -> None:
    order = ["PASS", "WARN", "INFO", "MISSING", "FAIL"]
    counts = gates["status"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    colors = ["#2ca25f", "#f0ad4e", "#6baed6", "#de2d26", "#7b3294"]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("gate count")
    ax.set_title("V3 claim gates: passed evidence and disclosed missing scale")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    for target in [OUT / "v3_gate_status.pdf", PAPER_FIG / "v3_gate_status.pdf"]:
        fig.savefig(target)
    plt.close(fig)


def plot_cross_benchmark(cross: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    labels = [b.replace("_", "\n") for b in cross["benchmark"]]
    ax.bar(labels, cross["acc_gain_N48_over_N1"], color="#4c78a8", alpha=0.88)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("accuracy gain: N=48 minus N=1")
    ax.set_title("Pilot cross-benchmark selection gain")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    for target in [OUT / "v3_cross_benchmark_delta.pdf", PAPER_FIG / "v3_cross_benchmark_delta.pdf"]:
        fig.savefig(target)
    plt.close(fig)


def plot_heldout_error(summary: pd.DataFrame) -> None:
    focus = summary[summary["K"].isin([8, 128, 256, 512])]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for k, part in focus.groupby("K"):
        ax.plot(part["N"], part["MAE"], marker="o", label=f"K={k}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("selection draw count N")
    ax.set_ylabel("held-out MAE")
    ax.set_title("4096-depth held-out slice: error grows with N")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for target in [OUT / "v3_heldout_error_by_N.pdf", PAPER_FIG / "v3_heldout_error_by_N.pdf"]:
        fig.savefig(target)
    plt.close(fig)


def plot_auc_gap(moment: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(moment["N"], moment["auc_only_mae"], marker="o", label="AUC-only")
    ax.plot(moment["N"], moment["moment_mae"], marker="o", label="score-rank moment")
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel("selection draw count N")
    ax.set_ylabel("MAE")
    ax.set_title("Pairwise AUC loses high-N score-rank information")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for target in [OUT / "v3_auc_gap_by_n.pdf", PAPER_FIG / "v3_auc_gap_by_n.pdf"]:
        fig.savefig(target)
    plt.close(fig)


def build_summary(
    du_results: dict[str, Any],
    live: dict[str, Any],
    allocation: dict[str, Any],
    heldout: pd.DataFrame,
    heldout_quantiles: pd.DataFrame,
    gates: pd.DataFrame,
    gate_counts: dict[str, int],
    gate_report: dict[str, Any],
    cross: pd.DataFrame,
    pilots: pd.DataFrame,
    moment: pd.DataFrame,
    heldout_error: pd.DataFrame,
) -> dict[str, Any]:
    audit = du_results["audit"]
    moment_by_n = du_results["moment_hierarchy"]["by_n"]
    pilot_k128_n8 = pilots[(pilots["K"] == 128) & (pilots["N"] == 8)].iloc[0]
    pilot_k192_n8 = pilots[(pilots["K"] == 192) & (pilots["N"] == 8)].iloc[0]
    heldout_k128_n8 = heldout_error[(heldout_error["K"] == 128) & (heldout_error["N"] == 8)].iloc[0]
    heldout_k512_n8 = heldout_error[(heldout_error["K"] == 512) & (heldout_error["N"] == 8)].iloc[0]
    heldout_k512_n256 = heldout_error[(heldout_error["K"] == 512) & (heldout_error["N"] == 256)].iloc[0]
    coverage_gate = gates[gates["name"] == "maxed_heldout_coverage"].iloc[0]
    coverage_value = next(g["value"] for g in gate_report["gates"] if g["name"] == "maxed_heldout_coverage")
    required_records = int(coverage_value["required"]["records"])
    observed_target = int(coverage_value["observed"]["records_at_target"])
    coverage_pct = 100.0 * observed_target / required_records
    ranking = allocation.get("ranking_at_reference_budget", [])
    uniform = next(r for r in ranking if r["policy"] == "uniform")
    moment_policy = next(r for r in ranking if r["policy"] == "moment_law")
    auc_policy = next(r for r in ranking if r["policy"] == "auc_kappa_based")

    return {
        "core": {
            "exact_law_mae": audit["overall_mae"],
            "n_triples": audit["n_triples"],
            "tie_rate": audit["tie_rate"],
            "per_problem_mae": audit["per_problem_mae"],
            "pooled_mae": audit["pooled_mae"],
            "n2_auc_diff": audit["n2_auc_diff"],
            "auc_only_mae_N48": moment_by_n["48"]["auc_only_mae"],
            "moment_mae_max": float(moment["moment_mae"].max()),
        },
        "pilot": {
            "models": int(pilot_k128_n8["models"]),
            "k128_n8_mae": float(pilot_k128_n8["heldout_MAE_mean"]),
            "k192_n8_mae": float(pilot_k192_n8["heldout_MAE_mean"]),
        },
        "heldout_4096": {
            "records": int(len(heldout)),
            "required_records": required_records,
            "records_at_target": observed_target,
            "coverage_percent": coverage_pct,
            "coverage_status": str(coverage_gate["status"]),
            "p_median": float(heldout["p"].median()),
            "p_q10": float(heldout["p"].quantile(0.10)),
            "p_q90": float(heldout["p"].quantile(0.90)),
            "kappa_median": float(heldout["kappa"].median()),
            "exact_f256_median": float(heldout["f_N256"].median()),
            "k128_n8_mae": float(heldout_k128_n8["MAE"]),
            "k512_n8_mae": float(heldout_k512_n8["MAE"]),
            "k512_n256_mae": float(heldout_k512_n256["MAE"]),
        },
        "live_judge": {
            "pairs": live["num_model_problem_pairs"],
            "judgments": live["num_model_problem_pairs"] * live["manifest_samples_per_problem"],
            "n48_live_acc": live["n48_live_judge_acc"],
            "n48_logprob_acc": live["n48_mean_logprob_acc_same_pairs"],
            "n48_delta": live["n48_improvement_over_meanlogprob"],
            "mean_live_auc": live["mean_live_judge_auc"],
            "mean_logprob_auc": live["mean_logprob_auc_on_same_pairs"],
        },
        "cross_benchmark": {
            "families": int(cross["benchmark"].nunique()),
            "records": int(cross["measurement_records"].sum()),
            "nondegenerate": int(cross["nondegenerate_records"].sum()),
            "mean_gain_N48_over_N1": finite_mean(cross["acc_gain_N48_over_N1"]),
            "max_exact_law_mae": float(cross["mean_exact_law_mae"].max()),
        },
        "allocation": {
            "uniform_acc": uniform["accuracy"],
            "moment_acc": moment_policy["accuracy"],
            "auc_acc": auc_policy["accuracy"],
            "moment_delta_over_uniform": moment_policy["accuracy"] - uniform["accuracy"],
            "auc_delta_over_uniform": auc_policy["accuracy"] - uniform["accuracy"],
        },
        "gates": {
            "counts": gate_counts,
            "all_passed": bool(gate_report["all_passed"]),
            "claim_pass": int(gate_report["claim_summary"].get("PASS", 0)),
            "claim_missing": int(gate_report["claim_summary"].get("MISSING", 0)),
        },
    }


def write_macros(summary: dict[str, Any]) -> None:
    c = summary["core"]
    p = summary["pilot"]
    h = summary["heldout_4096"]
    lj = summary["live_judge"]
    cb = summary["cross_benchmark"]
    alloc = summary["allocation"]
    gates = summary["gates"]
    text = ""
    text += macro("VThreeCoreMAE", fmt(c["exact_law_mae"], 6))
    text += macro("VThreeTriples", f"{int(c['n_triples']):,}")
    text += macro("VThreePooledMAE", fmt(c["pooled_mae"], 3))
    text += macro("VThreeAucNFortyEightMAE", fmt(c["auc_only_mae_N48"], 4))
    text += macro("VThreeMomentMaxMAE", fmt(c["moment_mae_max"], 2))
    text += macro("VThreePilotModels", int(p["models"]))
    text += macro("VThreePilotKOneTwoEightNEightMAE", fmt(p["k128_n8_mae"], 3))
    text += macro("VThreePilotKOneNineTwoNEightMAE", fmt(p["k192_n8_mae"], 3))
    text += macro("VThreeHeldoutRecords", int(h["records"]))
    text += macro("VThreeHeldoutRequiredRecords", f"{int(h['required_records']):,}")
    text += macro("VThreeHeldoutCoveragePct", fmt(h["coverage_percent"], 2) + "\\%")
    text += macro("VThreeHeldoutPMedian", fmt(h["p_median"], 3))
    text += macro("VThreeHeldoutPQTen", fmt(h["p_q10"], 3))
    text += macro("VThreeHeldoutPQNinety", fmt(h["p_q90"], 3))
    text += macro("VThreeHeldoutKappaMedian", fmt(h["kappa_median"], 3))
    text += macro("VThreeHeldoutFTwoFiveSixMedian", fmt(h["exact_f256_median"], 3))
    text += macro("VThreeHeldoutKOneTwoEightNEightMAE", fmt(h["k128_n8_mae"], 4))
    text += macro("VThreeHeldoutKFiveOneTwoNEightMAE", fmt(h["k512_n8_mae"], 4))
    text += macro("VThreeHeldoutKFiveOneTwoNTwoFiveSixMAE", fmt(h["k512_n256_mae"], 4))
    text += macro("VThreeLiveJudgePairs", int(lj["pairs"]))
    text += macro("VThreeLiveJudgeJudgments", f"{int(lj['judgments']):,}")
    text += macro("VThreeLiveJudgeNFortyEight", fmt(lj["n48_live_acc"], 3))
    text += macro("VThreeLogprobNFortyEight", fmt(lj["n48_logprob_acc"], 3))
    text += macro("VThreeLiveJudgeDelta", fmt(lj["n48_delta"], 3))
    text += macro("VThreeCrossFamilies", int(cb["families"]))
    text += macro("VThreeCrossRecords", f"{int(cb['records']):,}")
    text += macro("VThreeCrossNondegenerate", f"{int(cb['nondegenerate']):,}")
    text += macro("VThreeCrossMeanGain", fmt(cb["mean_gain_N48_over_N1"], 3))
    text += macro("VThreeCrossMaxMAE", fmt(cb["max_exact_law_mae"], 3))
    text += macro("VThreeMomentDelta", fmt(alloc["moment_delta_over_uniform"], 4))
    text += macro("VThreeAucDelta", fmt(alloc["auc_delta_over_uniform"], 4))
    text += macro("VThreeGatePass", int(gates["claim_pass"]))
    text += macro("VThreeGateMissing", int(gates["claim_missing"]))
    (PAPER / "v3_results_macros.tex").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    du_results = load_json(DU / "du_experiment_results.json")
    live = load_json(DU / "live_judge_score_summary.json")
    allocation = load_json(DU / "adaptive_allocation_summary.json")
    heldout, heldout_quantiles = summarize_heldout_records()
    gates, gate_counts, gate_report = summarize_gates()
    cross = summarize_cross_benchmarks()
    pilots = summarize_pilots()
    moment = summarize_moment_table(du_results)
    heldout_error = pd.read_csv(MAXED / "heldout_forecasting" / "tables" / "heldout_locked_estimator_summary.csv")

    heldout.to_csv(OUT / "heldout_slice_records.csv", index=False)
    heldout_quantiles.to_csv(OUT / "heldout_slice_quantiles.csv", index=False)
    gates.to_csv(OUT / "claim_gate_status.csv", index=False)
    cross.to_csv(OUT / "cross_benchmark_delta.csv", index=False)
    pilots.to_csv(OUT / "expanded_pilot_by_k_n.csv", index=False)
    moment.to_csv(OUT / "moment_auc_gap_by_n.csv", index=False)
    heldout_error.to_csv(OUT / "heldout_locked_estimator_summary.csv", index=False)

    copy_existing_figures()
    plot_heldout_profile(heldout)
    plot_gate_status(gates)
    plot_cross_benchmark(cross)
    plot_heldout_error(heldout_error)
    plot_auc_gap(moment)

    summary = build_summary(
        du_results,
        live,
        allocation,
        heldout,
        heldout_quantiles,
        gates,
        gate_counts,
        gate_report,
        cross,
        pilots,
        moment,
        heldout_error,
    )
    write_json(OUT / "summary.json", summary)
    write_macros(summary)
    print(f"v3 cached evidence complete: {OUT}")
    print(f"heldout_records={summary['heldout_4096']['records']} claim_missing={summary['gates']['claim_missing']}")


if __name__ == "__main__":
    main()
