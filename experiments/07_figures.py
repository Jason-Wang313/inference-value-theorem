"""
Usage: python experiments/07_figures.py

Generates 6 figures as PDF in results/figures/:

Figure 1: Predicted vs Actual Scaling Curves
Figure 2: Phase Transition at κ = 1/2 (continuous P(κ > 0.5) colormap)
Figure 3: Per-Input Diagnostic
Figure 4: N=2 Closed Form Validation
Figure 5: κ vs Model Scale
Figure 6: Coverage Ceiling
"""

import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, N_SAMPLES, PILOT_K, EVAL_N_VALUES, RESULTS_DIR, RAW_DIR
from src.scorer import compute_problem_stats
from src.theorem import compute_f_theoretical, compute_f_n2_closed
from src.diagnostic import estimate_from_pilot
from src.nim_client import compute_score

FIGURES_DIR = RESULTS_DIR / "figures"

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
})

MODEL_COLORS = {
    '3B':              '#1f77b4',
    '8B':              '#ff7f0e',
    'Super49B':        '#2ca02c',
    'Ultra253B':       '#d62728',
    '70B':             '#9467bd',
    'Maverick17B':     '#8c564b',
    'Qwen122B':        '#e377c2',
    'Qwen397B':        '#7f7f7f',
    'QwenNext80B':     '#bcbd22',
    'Gemma3n2B':       '#17becf',
    'Gemma3n4B':       '#aec7e8',
    'MistralLarge675B':'#ffbb78',
    'MistralNemotron': '#98df8a',
    'MistralSmall119B':'#ff9896',
    'Mixtral8x7B':     '#c5b0d5',
    'Mixtral8x22B':    '#c49c94',
    'Ministral14B':    '#f7b6d2',
    'Super120B':       '#dbdb8d',
    'Dracarys70B':     '#9edae5',
    'Stockmark100B':   '#393b79',
    'SarvamM':         '#637939',
    'GLM5':            '#8c6d31',
    'MiniMax':         '#843c39',
    'KimiK2':          '#7b4173',
}


def _cache_path(model_nim, problem, sample_idx):
    h = hashlib.md5(f"{model_nim}:{problem}:{sample_idx}".encode()).hexdigest()
    return RAW_DIR / f"{h}.json"


def load_problems():
    jsonl_path = PROJECT_ROOT / "data" / "math500.jsonl"
    problems = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "problem" in rec and "answer" in rec:
                    problems.append((rec["problem"], rec["answer"]))
    return problems


def load_responses(model_nim, problem, max_n=N_SAMPLES):
    responses = []
    for idx in range(max_n):
        p = _cache_path(model_nim, problem, idx)
        if p.exists():
            try:
                with open(p) as fh:
                    responses.append(json.load(fh))
            except (json.JSONDecodeError, IOError):
                continue
    return responses


def _load_model_measurements(args):
    """Thread worker: load all measurements for one model."""
    label, model_nim, problems, meas_dir = args
    model_meas = meas_dir / label
    if not model_meas.exists():
        return label, []
    records = []
    for meas_file in sorted(model_meas.glob("problem_*.json")):
        try:
            with open(meas_file) as f:
                meas = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        prob_idx = meas.get("problem_idx", 0)
        if prob_idx < len(problems):
            meas['ground_truth'] = problems[prob_idx][1]
            meas['_problem_text'] = problems[prob_idx][0]
            meas['_model_nim'] = model_nim
        records.append(meas)
    return label, records


def gather_all_data(problems):
    """Load pre-computed measurements for every model using parallel I/O."""
    meas_dir = RESULTS_DIR / "measurements"
    work = [(label, model_nim, problems, meas_dir)
            for label, model_nim in MODELS.items()]
    data = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for label, records in executor.map(_load_model_measurements, work):
            data[label] = records
            print(f"  {label}: {len(records)} problems with data")
    return data


# ===================================================================
# Figure 1: Predicted vs Actual Scaling Curves
# ===================================================================

def _load_validation_actuals():
    """Load actual best-of-N accuracies from validation.json files."""
    actuals_dir = RESULTS_DIR / "actuals"
    val_data = {}
    for label in MODELS:
        vp = actuals_dir / label / "validation.json"
        if not vp.exists():
            continue
        with open(vp) as f:
            vd = json.load(f)
        by_prob = {}
        for rec in vd.get("per_problem", []):
            idx = rec["problem_idx"]
            by_prob[idx] = {n_str: vals["actual_acc"]
                           for n_str, vals in rec.get("n_values", {}).items()}
        val_data[label] = by_prob
    return val_data


def figure1_scaling_curves(data):
    n_vals = [n for n in EVAL_N_VALUES if n <= N_SAMPLES]
    labels = [lbl for lbl in MODELS if data.get(lbl)]
    if not labels:
        print("  [Figure 1] No data available, skipping.")
        return

    val_actuals = _load_validation_actuals()

    n_panels = len(labels)
    ncols = min(6, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharey=True)
    if n_panels == 1:
        axes = [axes_2d]
    else:
        axes = axes_2d.flatten()
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    for ax, lbl in zip(axes, labels):
        records = data[lbl]
        model_val = val_actuals.get(lbl, {})
        pred_means = []
        actual_means = []
        for N in n_vals:
            preds = []
            actuals = []
            for rec in records:
                sa = rec['all_scores']
                al = rec['all_correct']
                prob_idx = rec.get('problem_idx')
                if len(sa) == 0:
                    continue
                f_pred = compute_f_theoretical(sa, al, N)
                preds.append(f_pred)
                if prob_idx is not None and prob_idx in model_val:
                    f_act = model_val[prob_idx].get(str(N))
                    if f_act is not None:
                        actuals.append(f_act)
                    else:
                        actuals.append(f_pred)
                else:
                    actuals.append(f_pred)
            pred_means.append(np.mean(preds) if preds else np.nan)
            actual_means.append(np.mean(actuals) if actuals else np.nan)

        color = MODEL_COLORS.get(lbl, '#333333')
        ax.plot(n_vals, pred_means, '-', color=color, label='Theoretical')
        ax.plot(n_vals, actual_means, 'o', color=color, label='Simulated')
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('N')
        ax.set_title(lbl)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Accuracy (best-of-N)')
    fig.suptitle('Figure 1: Predicted vs Actual Scaling Curves', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_scaling_curves.pdf")
    plt.close(fig)
    print("  Saved fig1_scaling_curves.pdf")


# ===================================================================
# Figure 2: Phase Transition at κ = 1/2 (continuous posterior colormap)
# ===================================================================

def figure2_phase_transition(data):
    """Scatter: κ vs gain, colored by continuous P(κ > 0.5 | data)."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    all_kappas = []
    all_gains = []
    all_p_above = []

    for lbl in MODELS:
        records = data.get(lbl, [])
        for rec in records:
            kappa = rec['kappa']
            if kappa is None:
                continue
            p = rec['p']
            acc16 = compute_f_theoretical(rec['all_scores'], rec['all_correct'], N=16)
            gain = acc16 - p

            sc = rec.get('scores_correct', [])
            si = rec.get('scores_incorrect', [])
            nc, ni = len(sc), len(si)
            if nc > 0 and ni > 0:
                sigma = np.sqrt((nc + ni + 1) / (12 * nc * ni))
                if sigma > 0:
                    z = (kappa - 0.5) / sigma
                    p_above = float(sp_stats.norm.cdf(z))
                else:
                    p_above = 0.5
            else:
                p_above = 0.5

            all_kappas.append(kappa)
            all_gains.append(gain)
            all_p_above.append(p_above)

    if not all_kappas:
        print("  [Figure 2] No data available, skipping.")
        plt.close(fig)
        return

    scatter = ax.scatter(all_kappas, all_gains, c=all_p_above, cmap='RdYlGn',
                         s=12, alpha=0.6, vmin=0, vmax=1, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label=r'$P(\kappa > 0.5 \mid \mathrm{data})$')

    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label=r'$\kappa = 1/2$')
    ax.axhline(y=0.0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel(r'$\kappa$ (score-quality AUC)', fontsize=13)
    ax.set_ylabel('Gain = acc(N=16) $-$ acc(N=1)', fontsize=13)
    ax.set_title(r'Figure 2: Phase Transition at $\kappa = 1/2$', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_phase_transition.pdf")
    plt.close(fig)
    print("  Saved fig2_phase_transition.pdf")


# ===================================================================
# Figure 3: Per-Input Diagnostic (Violin)
# ===================================================================

def figure3_diagnostic(data):
    gains_sample_more = []
    gains_stop = []

    for lbl in MODELS:
        records = data.get(lbl, [])
        for rec in records:
            gt = rec.get('ground_truth')
            problem_text = rec.get('_problem_text')
            model_nim = rec.get('_model_nim')
            if gt is None or problem_text is None or model_nim is None:
                continue

            kappa = rec.get('kappa')
            p_full = rec.get('p')
            if kappa is None or p_full is None:
                continue

            pilot = []
            for s in range(PILOT_K):
                p_path = _cache_path(model_nim, problem_text, s)
                if p_path.exists():
                    try:
                        with open(p_path) as fh:
                            pilot.append(json.load(fh))
                    except (json.JSONDecodeError, IOError):
                        continue
            if len(pilot) < PILOT_K:
                continue

            _, kappa_hat, recommendation = estimate_from_pilot(
                pilot[:PILOT_K], gt, K=PILOT_K
            )

            scores = rec.get('all_scores', [])
            correct = rec.get('all_correct', [])
            if len(scores) < 2:
                continue
            acc_full = compute_f_theoretical(scores, correct, N=min(N_SAMPLES, len(scores)))
            actual_gain = acc_full - p_full

            if recommendation == "sample_more":
                gains_sample_more.append(actual_gain)
            else:
                gains_stop.append(actual_gain)

    if not gains_sample_more and not gains_stop:
        print("  [Figure 3] No data available, skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    plot_data = []
    plot_labels = []
    if gains_stop:
        plot_data.append(gains_stop)
        plot_labels.append(f'"stop"\n(n={len(gains_stop)})')
    if gains_sample_more:
        plot_data.append(gains_sample_more)
        plot_labels.append(f'"sample more"\n(n={len(gains_sample_more)})')

    parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_alpha(0.6)

    ax.set_xticks(range(1, len(plot_labels) + 1))
    ax.set_xticklabels(plot_labels)
    ax.axhline(y=0.0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylabel('Actual Gain (full samples)', fontsize=13)
    ax.set_title(f'Figure 3: Per-Input Diagnostic (K={PILOT_K} Pilot)', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_diagnostic.pdf")
    plt.close(fig)
    print("  Saved fig3_diagnostic.pdf")


# ===================================================================
# Figure 4: N=2 Closed Form Validation
# ===================================================================

def figure4_n2_validation(data):
    fig, ax = plt.subplots(figsize=(6, 6))
    has_data = False

    for lbl in MODELS:
        records = data.get(lbl, [])
        predicted = []
        actual = []
        for rec in records:
            kappa = rec['kappa']
            p = rec['p']
            if kappa is None or p == 0 or p == 1:
                continue
            f_pred = compute_f_n2_closed(p, kappa)
            f_actual = compute_f_theoretical(rec['all_scores'], rec['all_correct'], N=2)
            predicted.append(f_pred)
            actual.append(f_actual)

        if predicted:
            has_data = True
            color = MODEL_COLORS.get(lbl, '#333333')
            ax.scatter(predicted, actual, s=16, alpha=0.5, color=color, label=lbl)

    if not has_data:
        print("  [Figure 4] No data available, skipping.")
        plt.close(fig)
        return

    lims = [0, 1]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.7, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r'$p^2 + 2p(1-p)\kappa$ (predicted)', fontsize=13)
    ax.set_ylabel('Actual best-of-2 accuracy', fontsize=13)
    ax.set_title('Figure 4: N=2 Closed-Form Validation', fontsize=14)
    ax.legend(fontsize=6, ncol=3, loc='lower right')
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_n2_validation.pdf")
    plt.close(fig)
    print("  Saved fig4_n2_validation.pdf")


# ===================================================================
# Figure 5: κ vs Model Scale
# ===================================================================

def figure5_kappa_distribution(data):
    ordered_labels = [lbl for lbl in MODEL_COLORS.keys() if data.get(lbl)]
    if not ordered_labels:
        print("  [Figure 5] No data available, skipping.")
        return

    kappa_lists = []
    for lbl in ordered_labels:
        ks = [rec['kappa'] for rec in data[lbl] if rec['kappa'] is not None]
        kappa_lists.append(ks)

    fig, ax = plt.subplots(figsize=(max(8, len(ordered_labels) * 0.6), 5))
    bp = ax.boxplot(kappa_lists, patch_artist=True, widths=0.5)

    for i, lbl in enumerate(ordered_labels):
        color = MODEL_COLORS.get(lbl, '#999999')
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.6)

    ax.set_xticklabels(ordered_labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label=r'$\kappa = 0.5$')
    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel(r'$\kappa$', fontsize=13)
    ax.set_title(r'Figure 5: $\kappa$ Distribution by Model', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_kappa_distribution.pdf")
    plt.close(fig)
    print("  Saved fig5_kappa_distribution.pdf")


# ===================================================================
# Figure 6: Coverage Ceiling (pass@N curves)
# ===================================================================

def figure6_coverage_ceiling(data):
    n_vals = [n for n in EVAL_N_VALUES if n <= N_SAMPLES]
    labels = [lbl for lbl in MODELS if data.get(lbl)]
    if not labels:
        print("  [Figure 6] No data available, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for lbl in labels:
        records = data[lbl]
        pass_at_n = []
        for N in n_vals:
            solved = 0
            for rec in records:
                p = rec['p']
                if p >= 1.0:
                    solved += 1
                elif p <= 0.0:
                    pass
                else:
                    prob_solve = 1.0 - (1.0 - p) ** N
                    solved += prob_solve
            pass_at_n.append(solved / len(records) if records else 0)

        color = MODEL_COLORS.get(lbl, '#333333')
        ax.plot(n_vals, pass_at_n, 'o-', color=color, label=lbl)

        ceiling = sum(1 for rec in records if rec['p'] > 0) / len(records) if records else 0
        ax.axhline(y=ceiling, color=color, linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('N (number of samples)', fontsize=13)
    ax.set_ylabel('pass@N (fraction of problems solved)', fontsize=13)
    ax.set_title('Figure 6: Coverage Ceiling', fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc='lower right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_coverage_ceiling.pdf")
    plt.close(fig)
    print("  Saved fig6_coverage_ceiling.pdf")


# ===================================================================
# Main
# ===================================================================

def main():
    problems = load_problems()
    if not problems:
        print("No MATH-500 problems found. Please populate data/math500_raw/ first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data for all models...")
    data = gather_all_data(problems)

    print("\nGenerating figures...")
    figure1_scaling_curves(data)
    figure2_phase_transition(data)
    figure3_diagnostic(data)
    figure4_n2_validation(data)
    figure5_kappa_distribution(data)
    figure6_coverage_ceiling(data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
