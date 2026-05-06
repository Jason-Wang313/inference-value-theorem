"""
Theoretical prediction (exact discrete form):
f = sum_{i: correct} [(r_i/n)^N - ((r_i-1)/n)^N]

With ties handled via uniform random tie-breaking:
f = sum over tie groups g: (c_g/k_g) * [(r_max_g/n)^N - ((r_min_g-1)/n)^N]

Also: N=2 closed form: f(p, kappa, 2) = p^2 + 2*p*(1-p)*kappa
"""
import numpy as np


def compute_f_theoretical(all_scores, all_correct, N):
    """Exact discrete Best-of-N probability with random tie-breaking.

    all_scores: array of scores for ALL responses (correct + incorrect)
    all_correct: boolean array, True where response is correct
    N: number of samples in best-of-N
    """
    scores = np.asarray(all_scores, dtype=float)
    correct = np.asarray(all_correct, dtype=bool)
    n = len(scores)

    if n == 0:
        return 0.0

    p = float(np.mean(correct))
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    if N == 1:
        return p

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_correct = correct[order]

    f = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        k_g = j - i
        c_g = int(np.sum(sorted_correct[i:j]))
        r_min = i + 1
        r_max = j
        if c_g > 0:
            prob_mass = (r_max / n) ** N - ((r_min - 1) / n) ** N
            f += (c_g / k_g) * prob_mass
        i = j

    return float(f)


def compute_f_n2_closed(p, kappa):
    """N=2 closed form: f = p^2 + 2*p*(1-p)*kappa"""
    return p**2 + 2 * p * (1 - p) * kappa


def predict_all_n(all_scores, all_correct, p, kappa, n_values):
    """Compute theoretical f for all N values using the exact discrete formula.
    Returns dict {N: f}.
    Also validates N=2 closed form matches general formula.
    """
    predictions = {}
    for N in n_values:
        predictions[N] = compute_f_theoretical(all_scores, all_correct, N)

    if kappa is not None and 2 in n_values:
        f2_closed = compute_f_n2_closed(p, kappa)
        f2_general = predictions[2]
        if abs(f2_closed - f2_general) > 0.05:
            print(f"WARNING: N=2 mismatch: closed={f2_closed:.4f}, general={f2_general:.4f}")

    return predictions


def simulate_best_of_n(scores, correct, N, n_trials=10000, rng=None):
    """Simulate best-of-N selection with random tie-breaking (vectorized)."""
    if rng is None:
        rng = np.random.default_rng(42)

    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n_samples = len(scores)

    indices = rng.choice(n_samples, size=(n_trials, N), replace=True)
    picked_scores = scores[indices] + rng.uniform(0, 1e-12, size=(n_trials, N))
    best_per_trial = np.argmax(picked_scores, axis=1)
    selected = indices[np.arange(n_trials), best_per_trial]
    return float(np.mean(correct[selected]))
