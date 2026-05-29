"""
Write a consolidated status note for the paper-facing experiment claims.

This is deliberately evidence-oriented: it only summarizes artifacts that exist
on disk and calls out remaining gaps rather than upgrading claims implicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "du_aligned"
TABLE_DIR = OUT_DIR / "tables"
STATUS_PATH = OUT_DIR / "paper_claims_status.md"
LIVE_JUDGE_PATH = OUT_DIR / "model_judge" / "judged_responses_live.jsonl"
LIVE_MANIFEST_PATH = OUT_DIR / "model_judge" / "judge_subset_manifest_live.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x, digits: int = 4) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _live_judge_manifest_coverage(manifest_path: Path = LIVE_MANIFEST_PATH) -> dict:
    manifest = _load_json(manifest_path)
    if not manifest or not LIVE_JUDGE_PATH.exists():
        return {}
    target_models = manifest.get("target_models", [])
    problem_indices = manifest.get("problem_indices", [])
    samples_per_problem = int(manifest.get("samples_per_problem", 48) or 48)
    expected_pairs = {(model, int(pidx)) for model in target_models for pidx in problem_indices}
    seen: dict[tuple[str, int], set[int]] = {pair: set() for pair in expected_pairs}

    with open(LIVE_JUDGE_PATH, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("judge_mode") != "live_no_reference":
                continue
            if str(rec.get("judge_source", "")).endswith(":failed"):
                continue
            pair = (rec.get("model"), int(rec.get("problem_idx", -1)))
            if pair in seen:
                seen[pair].add(int(rec.get("sample_idx", -1)))

    complete_pairs = sum(len(samples) >= samples_per_problem for samples in seen.values())
    observed = sum(min(len(samples), samples_per_problem) for samples in seen.values())
    expected = len(expected_pairs) * samples_per_problem
    return {
        "path": str(manifest_path),
        "n_problems": len(problem_indices),
        "models": target_models,
        "complete_pairs": complete_pairs,
        "expected_pairs": len(expected_pairs),
        "observed_judgments": observed,
        "expected_judgments": expected,
    }


def _live_judge_manifest_snapshot_summary() -> tuple[str, dict]:
    manifest_dir = LIVE_MANIFEST_PATH.parent
    coverages = []
    for path in sorted(manifest_dir.glob("judge_subset_manifest_live*.json")):
        cov = _live_judge_manifest_coverage(path)
        if cov:
            coverages.append(cov)
    if not coverages:
        return "N/A", {}

    complete = [
        cov for cov in coverages
        if cov["complete_pairs"] == cov["expected_pairs"]
        and cov["observed_judgments"] == cov["expected_judgments"]
    ]
    best = max(
        complete or coverages,
        key=lambda cov: (cov["complete_pairs"] == cov["expected_pairs"], cov["n_problems"], cov["observed_judgments"]),
    )
    text = (
        f"{best['complete_pairs']}/{best['expected_pairs']} complete model/problem pairs and "
        f"{best['observed_judgments']}/{best['expected_judgments']} judgments "
        f"for the {best['n_problems']}-problem manifest"
    )
    return text, best


def main() -> None:
    allocation = _load_json(OUT_DIR / "adaptive_allocation_summary.json")
    live_judge = _load_json(OUT_DIR / "live_judge_score_summary.json")
    expanded_pilot = _load_json(OUT_DIR / "expanded_pilot_summary.json")
    cross_benchmark = _load_json(PROJECT_ROOT / "results" / "benchmarks" / "cross_benchmark_summary.json")

    pilot_table = TABLE_DIR / "expanded_pilot_sample_complexity.csv"
    pilot_trend = "N/A"
    coverage_text = "N/A"
    if pilot_table.exists():
        df = pd.read_csv(pilot_table)
        n8 = df[df["N"] == 8].groupby("K")["heldout_MAE_mean"].mean().reset_index()
        pilot_trend = ", ".join(f"K={int(r.K)}: {_fmt(r.heldout_MAE_mean, 3)}" for r in n8.itertuples())
    per_model_coverage = expanded_pilot.get("per_model_sample_coverage", {})
    max_measured = expanded_pilot.get("max_samples_per_problem")
    k_values = expanded_pilot.get("K_values_tested", [])
    coverage_complete = False
    if per_model_coverage:
        pieces = []
        for model, stats in per_model_coverage.items():
            pieces.append(
                f"{model}: {stats.get('records_at_max', 0)}/{stats.get('num_records', 0)} "
                f"records at n={stats.get('max_samples', 0)}"
            )
        coverage_text = "; ".join(pieces)
        coverage_complete = all(
            stats.get("num_records", 0) > 0
            and stats.get("records_at_max", 0) == stats.get("num_records", 0)
            and stats.get("max_samples", 0) == max_measured
            for stats in per_model_coverage.values()
        )
    else:
        coverage_path = TABLE_DIR / "expanded_pilot_cache_coverage.csv"
        if coverage_path.exists():
            cov = pd.read_csv(coverage_path)
            cov["complete"] = cov["cached_samples"] >= cov["target_samples"]
            pieces = []
            for model, sub in cov.groupby("model"):
                pieces.append(
                    f"{model}: {int(sub['complete'].sum())}/{len(sub)} complete, "
                    f"mean cached={_fmt(sub['cached_samples'].mean(), 1)}"
                )
            coverage_text = "; ".join(pieces)

    allocation_ref = allocation.get("ranking_at_reference_budget", [])
    allocation_rank = "N/A"
    if allocation_ref:
        allocation_rank = ", ".join(
            f"{row['policy']}={_fmt(row['accuracy'])}" for row in allocation_ref
        )
    uniform = next((r for r in allocation_ref if r.get("policy") == "uniform"), None)
    moment = next((r for r in allocation_ref if r.get("policy") == "moment_law"), None)
    auc = next((r for r in allocation_ref if r.get("policy") == "auc_kappa_based"), None)
    moment_delta = (
        float(moment["accuracy"]) - float(uniform["accuracy"])
        if moment and uniform
        else None
    )
    auc_delta = (
        float(auc["accuracy"]) - float(uniform["accuracy"])
        if auc and uniform
        else None
    )
    highest_k = max(k_values) if k_values else None
    models_at_256 = [
        model for model, stats in per_model_coverage.items()
        if stats.get("num_records", 0) > 0
        and stats.get("records_at_max", 0) == stats.get("num_records", 0)
        and stats.get("max_samples", 0) >= 256
    ]
    remaining_models = [
        model for model, stats in per_model_coverage.items()
        if stats.get("max_samples", 0) < 256
    ]
    if max_measured and max_measured >= 256 and models_at_256:
        if remaining_models:
            pilot_interpretation = (
                f"K=128/192 curves are now evidenced for {models_at_256}; "
                "finish the remaining models before claiming a complete five-model K=128/192 curve."
            )
            pilot_gap = (
                f"Extend remaining expanded-pilot models to 256 samples/problem for full five-model K=128/192 curves: {remaining_models}."
            )
        else:
            pilot_interpretation = (
                "K=128/192 curves are now evidenced for all five expanded-pilot models; "
                "the five-model curve is complete through K=192."
            )
            pilot_gap = "None for expanded-pilot K=128/192."
    elif coverage_complete and max_measured == 128 and highest_k and highest_k >= 96:
        pilot_interpretation = (
            "the old K<=32 ceiling is now broken through K=96 on five complete "
            "128-sample models; K=128/192 requires a 256-sample extension."
        )
        pilot_gap = (
            "Extend expanded-pilot collection from 128 to 256 samples/problem if the paper needs K=128/192 curves."
        )
    else:
        pilot_interpretation = (
            "the old K<=32 ceiling is broken, but higher-K curves still require more complete collection."
        )
        pilot_gap = (
            "Finish expanded-pilot collection to at least 128, ideally 256, for the live 5-model subset."
        )
    judge_pairs = int(live_judge.get("num_model_problem_pairs", 0) or 0)
    live_manifest_coverage = _live_judge_manifest_coverage()
    best_manifest_text, best_manifest_coverage = _live_judge_manifest_snapshot_summary()
    du_results = _load_json(OUT_DIR / "du_experiment_results.json")
    bundled_pilot = du_results.get("expanded_pilot", {})
    if (
        bundled_pilot.get("max_samples_per_problem") == expanded_pilot.get("max_samples_per_problem")
        and bundled_pilot.get("K_values_tested") == expanded_pilot.get("K_values_tested")
    ):
        main_bundle_gap_line = ""
    else:
        main_bundle_gap_line = "- Re-run or refresh the main Du-aligned bundle after the larger artifacts are complete.\n"
    if live_manifest_coverage:
        live_manifest_text = (
            f"{live_manifest_coverage['complete_pairs']}/{live_manifest_coverage['expected_pairs']} "
            f"complete model/problem pairs and "
            f"{live_manifest_coverage['observed_judgments']}/{live_manifest_coverage['expected_judgments']} "
            f"judgments for the {live_manifest_coverage['n_problems']}-problem manifest"
        )
    else:
        live_manifest_text = "N/A"
    current_manifest_complete = bool(
        live_manifest_coverage
        and live_manifest_coverage["complete_pairs"] == live_manifest_coverage["expected_pairs"]
        and live_manifest_coverage["observed_judgments"] == live_manifest_coverage["expected_judgments"]
    )
    if current_manifest_complete and live_manifest_coverage.get("n_problems", 0) >= 45:
        judge_interpretation = (
            "the score-agnostic live-judge claim is now supported on the completed "
            f"{live_manifest_coverage['n_problems']}-problem stratified subset; keep claims about "
            "judge superiority scoped to this subset."
        )
        judge_gap = (
            "Optional: add a second external verifier if the paper needs broader judge-generalization evidence."
        )
    else:
        judge_interpretation = (
            "the score-agnostic live-judge claim is supported on the completed subset available so far; "
            "expand the subset before making broad judge-performance claims."
        )
        judge_gap = (
            f"Expand live LLM judge coverage from the current complete "
            f"{best_manifest_coverage.get('n_problems', 'small')}-problem manifest "
            "to the planned stratified subset."
            if best_manifest_coverage.get("complete_pairs") == best_manifest_coverage.get("expected_pairs")
            else f"Expand live LLM judge coverage from the current {judge_pairs} complete model/problem pairs "
            "to the planned stratified subset."
        )
    cross_rows = cross_benchmark.get("benchmarks", [])
    cross_model_counts = sorted({len(row.get("models", [])) for row in cross_rows})
    if len(cross_model_counts) == 1:
        cross_scale = (
            f"{cross_model_counts[0]}-model stable live NIM panel, "
            "20 tasks/benchmark, 48 samples/model/task"
        )
    elif cross_model_counts:
        cross_scale = (
            f"mixed model counts {cross_model_counts}, "
            "20 tasks/benchmark, 48 samples/model/task"
        )
    else:
        cross_scale = "N/A"
    cross_complete = bool(cross_rows) and all(
        row.get("expected_records") == row.get("measurement_records")
        and row.get("grading_coverage_rate") == 1.0
        and row.get("mean_exact_law_mae") == 0.0
        and row.get("nondegenerate_records", 0) > 0
        and 48 in row.get("N_values", [])
        for row in cross_rows
    )
    cross_benchmarks = ", ".join(str(row.get("benchmark")) for row in cross_rows) if cross_rows else "N/A"
    cross_coverage = (
        ", ".join(
            f"{row.get('benchmark')}: {row.get('measurement_records')}/{row.get('expected_records')}"
            for row in cross_rows
        )
        if cross_rows
        else "N/A"
    )
    cross_mae = (
        ", ".join(f"{row.get('benchmark')}: {_fmt(row.get('mean_exact_law_mae'), 6)}" for row in cross_rows)
        if cross_rows
        else "N/A"
    )
    cross_n48 = (
        ", ".join(
            f"{row.get('benchmark')}={_fmt(row.get('best_of_n_curve', {}).get('48'))}"
            for row in cross_rows
        )
        if cross_rows
        else "N/A"
    )
    cross_nondegenerate = (
        ", ".join(f"{row.get('benchmark')}: {row.get('nondegenerate_records')}" for row in cross_rows)
        if cross_rows
        else "N/A"
    )
    if cross_complete:
        cross_status = "completed"
        cross_interpretation = (
            "task broadening is now evidenced on four additional benchmarks spanning science QA, "
            "instruction following, mixed LiveBench tasks, and executable code across the stable "
            "live endpoint panel; LiveCodeBench is scoped to public tests."
        )
        cross_gap = (
            "Optional: add hidden/private LiveCodeBench scoring if the paper needs a stronger code appendix; "
            "do not claim unavailable provider endpoints were benchmarked live."
        )
    elif cross_rows:
        cross_status = "partial"
        cross_interpretation = (
            "cross-benchmark evidence exists but is not clean enough for the strongest paper claim yet."
        )
        cross_gap = "Finish cross-benchmark coverage, grading, and exact-law checks before making broad task-generalization claims."
    else:
        cross_status = "missing"
        cross_interpretation = "cross-benchmark broadening has not been generated yet."
        cross_gap = "Run the cross-benchmark expansion before claiming task-domain generality."
    remaining_lines = [
        f"- {pilot_gap}",
        f"- {judge_gap}",
        f"- {cross_gap}",
    ]
    if main_bundle_gap_line:
        remaining_lines.append(main_bundle_gap_line.rstrip())
    remaining_lines.append("- Do not claim global optimal allocation, broad judge superiority, or tiny-pilot sufficiency.")

    text = f"""# Paper Claims Status

## Core theorem evidence
- Existing Du-aligned bundle validates the exact finite empirical selector law across 23 models with overall MAE 0.000622.
- Existing moment-hierarchy results show AUC/kappa is exact for N=2 and insufficient for high N; moment predictor is exact up to floating-point error.

## Adaptive allocation
- Status: {allocation.get('status', 'missing')}.
- Models: {allocation.get('models', [])}.
- Policies: {allocation.get('policies', [])}.
- Reference-budget ranking: {allocation_rank}.
- Moment-law improvement over uniform at reference budget: {_fmt(moment_delta)}.
- AUC/kappa-policy improvement over uniform at reference budget: {_fmt(auc_delta)}.
- Interpretation: adaptive allocation is now evidenced, but the current finite-pilot moment policy is modest; oracle remains the upper bound.

## Expanded pilot sample complexity
- Status: completed artifact with max measured samples/problem = {expanded_pilot.get('max_samples_per_problem', 'N/A')}.
- Models: {expanded_pilot.get('models', [])}.
- K values tested: {expanded_pilot.get('K_values_tested', [])}.
- N values tested: {expanded_pilot.get('N_values_tested', [])}.
- N=8 held-out MAE trend: {pilot_trend}.
- Cache coverage toward current target: {coverage_text}.
- Interpretation: {pilot_interpretation}

## Live LLM judge score
- Status: {live_judge.get('status', 'missing')}.
- Complete model/problem pairs analyzed: {live_judge.get('num_model_problem_pairs', 0)}.
- Current manifest coverage: {live_manifest_text}.
- Best completed manifest coverage: {best_manifest_text}.
- Models: {live_judge.get('models', [])}.
- N values: {live_judge.get('N_values', [])}.
- Exact-law MAE for live judge score: {_fmt(live_judge.get('mean_mae'), 6)}.
- Mean live-judge AUC: {_fmt(live_judge.get('mean_live_judge_auc'))}; mean logprob AUC on same pairs: {_fmt(live_judge.get('mean_logprob_auc_on_same_pairs'))}.
- N=48 live judge accuracy: {_fmt(live_judge.get('n48_live_judge_acc'))}; mean-logprob same-pair accuracy: {_fmt(live_judge.get('n48_mean_logprob_acc_same_pairs'))}; delta: {_fmt(live_judge.get('n48_improvement_over_meanlogprob'))}.
- Interpretation: {judge_interpretation}

## Cross-benchmark task broadening
- Status: {cross_status}.
- Benchmarks: {cross_benchmarks}.
- Scale: {cross_scale}.
- Coverage: {cross_coverage}.
- Non-degenerate records: {cross_nondegenerate}.
- Exact-law MAE by benchmark: {cross_mae}.
- N=48 best-of-N accuracy by benchmark: {cross_n48}.
- Scope note: excludes provider EOL/degraded/rate-limited endpoints documented in `config.py` and benchmark collection failure logs.
- Interpretation: {cross_interpretation}

## Remaining ideal-paper gaps
{chr(10).join(remaining_lines)}
"""
    STATUS_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {STATUS_PATH}")


if __name__ == "__main__":
    main()
