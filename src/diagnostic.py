"""
Per-input triage from K pilot samples.
Given K responses, estimate p̂ and κ̂, then use the exact discrete formula
to predict whether sampling more (best-of-N at N=16) will yield meaningful gain.
"""
import numpy as np
from scipy import stats


GAIN_THRESHOLD = 0.01


def estimate_from_pilot(responses, ground_truth, K=8):
    """From K pilot samples, estimate p and κ, predict gain from best-of-N.

    Returns
    -------
    (p_hat, kappa_hat, recommendation)
        p_hat : float – estimated fraction correct
        kappa_hat : float or None - estimated AUC (kappa)
        recommendation : str – "sample_more" if predicted gain > threshold, else "stop"
    """
    from .utils import check_correctness
    from .nim_client import compute_score
    from .theorem import compute_f_theoretical

    scores = []
    correct = []
    for resp in responses[:K]:
        s = compute_score(resp.get('logprobs'))
        c, _ = check_correctness(resp['content'], ground_truth)
        scores.append(s)
        correct.append(c)

    p_hat = sum(correct) / K

    scores_correct = [s for s, c in zip(scores, correct) if c]
    scores_incorrect = [s for s, c in zip(scores, correct) if not c]

    kappa_hat = None
    if len(scores_correct) > 0 and len(scores_incorrect) > 0:
        try:
            u, _ = stats.mannwhitneyu(
                scores_correct, scores_incorrect, alternative='greater'
            )
            kappa_hat = u / (len(scores_correct) * len(scores_incorrect))
        except ValueError:
            kappa_hat = 0.5

    if p_hat == 0.0:
        return p_hat, kappa_hat, "sample_more"
    if p_hat == 1.0:
        return p_hat, kappa_hat, "stop"

    predicted_f16 = compute_f_theoretical(scores, correct, N=16)
    predicted_gain = predicted_f16 - p_hat

    if predicted_gain > GAIN_THRESHOLD:
        recommendation = "sample_more"
    else:
        recommendation = "stop"

    return p_hat, kappa_hat, recommendation
