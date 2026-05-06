"""
Usage: python experiments/05_phase.py

Phase transition at κ = 1/2.
Uses analytical one-sided test P(κ > 0.5 | data) instead of bootstrap CI.
Reports continuous posterior probability — no binary indeterminate classification.
"""

import sys
import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
MEASUREMENTS_DIR = RESULTS_DIR / "measurements"
ACTUALS_DIR = RESULTS_DIR / "actuals"


def classify_kappa(scores_correct, scores_incorrect, threshold=0.99):
    """Analytical one-sided test: P(κ > 0.5 | data).

    Uses the normal approximation to the Mann-Whitney U distribution.
    Returns (classification, kappa_hat, p_above_half).
    """
    sc = np.asarray(scores_correct, dtype=float)
    si = np.asarray(scores_incorrect, dtype=float)
    nc, ni = len(sc), len(si)

    if nc == 0 or ni == 0:
        return "insufficient", None, None

    try:
        u, _ = sp_stats.mannwhitneyu(sc, si, alternative='greater')
    except ValueError:
        return "insufficient", None, None

    kappa_hat = u / (nc * ni)
    sigma = np.sqrt((nc + ni + 1) / (12 * nc * ni))

    if sigma == 0:
        return "insufficient", kappa_hat, None

    z = (kappa_hat - 0.5) / sigma
    p_above = float(sp_stats.norm.cdf(z))

    if p_above > threshold:
        return "above", kappa_hat, p_above
    elif p_above < (1 - threshold):
        return "below", kappa_hat, p_above
    else:
        return "uncertain", kappa_hat, p_above


def run_phase_analysis():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []

    for label in MODELS:
        meas_dir = MEASUREMENTS_DIR / label
        actuals_path = ACTUALS_DIR / label / "validation.json"

        if not meas_dir.exists():
            print(f"{label}: no measurements, skipping")
            continue

        actual_by_prob = {}
        if actuals_path.exists():
            with open(actuals_path) as f:
                val_data = json.load(f)
            for rec in val_data.get("per_problem", []):
                idx = rec["problem_idx"]
                if "16" in rec.get("n_values", {}):
                    actual_by_prob[idx] = rec["n_values"]["16"]["actual_acc"]

        n_loaded = 0

        for meas_file in sorted(meas_dir.glob("problem_*.json")):
            try:
                with open(meas_file) as f:
                    meas = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            kappa = meas.get("kappa")
            p = meas.get("p")
            prob_idx = meas.get("problem_idx")

            if kappa is None or p is None:
                continue

            n_loaded += 1

            if prob_idx in actual_by_prob:
                acc_16 = actual_by_prob[prob_idx]
            else:
                from src.theorem import simulate_best_of_n
                scores = np.array(meas["all_scores"])
                correct = np.array(meas["all_correct"])
                if len(scores) < 2:
                    continue
                acc_16 = simulate_best_of_n(scores, correct, N=16, n_trials=2000)

            gain = acc_16 - p

            sc = meas.get("scores_correct", [])
            si = meas.get("scores_incorrect", [])
            classification, kappa_hat, p_above = classify_kappa(sc, si)

            all_records.append({
                "model": label, "kappa": kappa, "gain": gain,
                "classification": classification,
                "p_above_half": p_above,
            })

        print(f"{label}: {sum(1 for r in all_records if r['model'] == label)} problems with valid κ (out of {n_loaded})")

    if not all_records:
        print("No valid records found.")
        return

    # Generate figure with continuous coloring by P(κ > 0.5)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    kappas = [r["kappa"] for r in all_records]
    gains = [r["gain"] for r in all_records]
    p_above_vals = [r["p_above_half"] if r["p_above_half"] is not None else 0.5 for r in all_records]

    sc = ax.scatter(kappas, gains, c=p_above_vals, cmap='RdYlGn', s=12, alpha=0.6,
                    vmin=0, vmax=1, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax, label=r'$P(\kappa > 0.5 \mid \mathrm{data})$')

    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label=r'$\kappa = 1/2$')
    ax.axhline(y=0.0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel(r'$\kappa$ (score-quality AUC)', fontsize=13)
    ax.set_ylabel('Gain = acc(N=16) − acc(N=1)', fontsize=13)
    ax.set_title('Phase Transition at $\\kappa = 1/2$', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()

    out_path = FIGURES_DIR / "phase_transition.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")

    # Classification statistics
    n_total = len(all_records)
    n_above = sum(1 for r in all_records if r["classification"] == "above")
    n_below = sum(1 for r in all_records if r["classification"] == "below")
    n_uncertain = sum(1 for r in all_records if r["classification"] == "uncertain")
    n_insufficient = sum(1 for r in all_records if r["classification"] == "insufficient")

    classified = [r for r in all_records if r["classification"] in ("above", "below")]
    n_classified = len(classified)
    n_violations = sum(1 for r in classified
                       if (r["classification"] == "below" and r["gain"] > 0.005)
                       or (r["classification"] == "above" and r["gain"] < -0.005))
    frac_violating = n_violations / n_classified if n_classified > 0 else 0

    print(f"\nPhase transition statistics (analytical one-sided test, α=0.05):")
    print(f"  Total: {n_total}")
    print(f"  Classified above 0.5: {n_above}")
    print(f"  Classified below 0.5: {n_below}")
    print(f"  Uncertain: {n_uncertain}")
    print(f"  Insufficient data: {n_insufficient}")
    print(f"  Classified total: {n_classified}")
    print(f"  Violations (among classified): {n_violations} ({frac_violating:.1%})")

    summary = {
        "total_problems": n_total,
        "above": n_above,
        "below": n_below,
        "uncertain": n_uncertain,
        "insufficient": n_insufficient,
        "classified": n_classified,
        "violations": n_violations,
        "fraction_violating_classified": frac_violating,
        "per_model": {},
    }

    model_labels = sorted(set(r["model"] for r in all_records))
    for lbl in model_labels:
        recs = [r for r in all_records if r["model"] == lbl]
        cls = [r for r in recs if r["classification"] in ("above", "below")]
        n = len(cls)
        v = sum(1 for r in cls
                if (r["classification"] == "below" and r["gain"] > 0.005)
                or (r["classification"] == "above" and r["gain"] < -0.005))
        summary["per_model"][lbl] = {
            "n_problems": len(recs),
            "above": sum(1 for r in recs if r["classification"] == "above"),
            "below": sum(1 for r in recs if r["classification"] == "below"),
            "uncertain": sum(1 for r in recs if r["classification"] == "uncertain"),
            "classified": n,
            "violations": v,
            "fraction_violating_classified": v / n if n > 0 else 0,
            "mean_kappa": float(np.mean([r["kappa"] for r in recs])),
            "mean_gain": float(np.mean([r["gain"] for r in recs])),
        }

    summary_path = MEASUREMENTS_DIR / "phase_transition_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    run_phase_analysis()
