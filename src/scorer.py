"""
For each model, for each problem:
1. Load all 64 cached responses
2. Check correctness of each response
3. Compute score s(y) = mean logprob
4. Compute:
   - p = fraction correct
   - F+ = scores of correct responses
   - F- = scores of incorrect responses
   - κ = AUC = P(s+ > s-) via Mann-Whitney U
"""
import numpy as np
from scipy import stats
from pathlib import Path
import json


def compute_problem_stats(responses, ground_truth):
    """Compute (p, kappa, scores_correct, scores_incorrect, all_scores, all_correct)
    from a list of response dicts.

    Each response dict has keys: 'content', 'logprobs'
    ground_truth: the correct answer string
    """
    from .utils import check_correctness
    from .nim_client import compute_score

    all_scores = []
    all_correct = []
    scores_correct = []
    scores_incorrect = []

    for resp in responses:
        score = compute_score(resp.get('logprobs'))
        is_correct, _ = check_correctness(resp['content'], ground_truth)

        all_scores.append(score)
        all_correct.append(is_correct)

        if is_correct:
            scores_correct.append(score)
        else:
            scores_incorrect.append(score)

    n = len(responses)
    p = sum(all_correct) / n if n > 0 else 0.0

    # Compute kappa (AUC) if we have both correct and incorrect
    kappa = None
    if len(scores_correct) > 0 and len(scores_incorrect) > 0:
        # Mann-Whitney U test
        # AUC = P(s+ > s-) = U / (n1 * n2)
        u_stat, _ = stats.mannwhitneyu(
            scores_correct, scores_incorrect, alternative='greater'
        )
        kappa = u_stat / (len(scores_correct) * len(scores_incorrect))

    return {
        'p': p,
        'kappa': kappa,
        'n_correct': len(scores_correct),
        'n_incorrect': len(scores_incorrect),
        'scores_correct': scores_correct,
        'scores_incorrect': scores_incorrect,
        'all_scores': all_scores,
        'all_correct': all_correct,
    }
