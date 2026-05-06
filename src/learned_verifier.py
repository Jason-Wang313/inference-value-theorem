"""
Learned verifier training and evaluation.

Trains sklearn classifiers on raw-cache features to predict correctness,
then uses predict_proba as verifier scores for best-of-N reranking.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from src.feature_extraction import LEARNABLE_FEATURE_COLS


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if not np.any(mask):
            continue
        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        ece += np.sum(mask) * abs(bin_acc - bin_conf)
    return float(ece / len(y_true)) if len(y_true) > 0 else 0.0


def load_features_for_model(csv_path: Path, model_key: str) -> dict:
    rows_by_problem: dict[int, list[dict]] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model_key"] != model_key:
                continue
            parsed = {
                "problem_idx": int(row["problem_idx"]),
                "sample_idx": int(row["sample_idx"]),
                "correct": int(float(row["correct"])),
            }
            for col in LEARNABLE_FEATURE_COLS:
                try:
                    val = float(row[col])
                except (TypeError, ValueError):
                    val = 0.0
                parsed[col] = val if math.isfinite(val) else 0.0
            rows_by_problem[parsed["problem_idx"]].append(parsed)
    return rows_by_problem


def _build_arrays(
    rows_by_problem: dict[int, list[dict]],
    problem_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    X_rows = []
    y_rows = []
    for pidx in problem_indices:
        for row in rows_by_problem.get(pidx, []):
            feat = [row.get(c, 0.0) for c in LEARNABLE_FEATURE_COLS]
            X_rows.append(feat)
            y_rows.append(row["correct"])
    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=-100.0)
    return X, y


def make_classifier(classifier_type: str):
    if classifier_type == "logistic":
        return LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    elif classifier_type == "gbdt":
        return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    elif classifier_type == "calibrated_logistic":
        base = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        return CalibratedClassifierCV(base, method="isotonic", cv=5)
    elif classifier_type == "calibrated_gbdt":
        base = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        return CalibratedClassifierCV(base, method="sigmoid", cv=5)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def train_per_model_verifier(
    csv_path: Path,
    model_key: str,
    classifier_type: str = "logistic",
) -> tuple[dict[int, np.ndarray], dict]:
    rows_by_problem = load_features_for_model(csv_path, model_key)
    all_problems = sorted(rows_by_problem.keys())

    train_problems = [p for p in all_problems if p % 5 != 0]
    test_problems = [p for p in all_problems if p % 5 == 0]

    X_train, y_train = _build_arrays(rows_by_problem, train_problems)
    X_test, y_test = _build_arrays(rows_by_problem, test_problems)

    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        scores = {}
        for pidx in test_problems:
            n = len(rows_by_problem.get(pidx, []))
            scores[pidx] = np.full(n, 0.5)
        return scores, {"auc": float("nan"), "brier": float("nan"), "ece": float("nan")}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = make_classifier(classifier_type)
    clf.fit(X_train_s, y_train)

    if hasattr(clf, "predict_proba"):
        y_prob_test = clf.predict_proba(X_test_s)[:, 1]
    else:
        y_prob_test = clf.decision_function(X_test_s)

    auc = float("nan")
    if len(np.unique(y_test)) >= 2:
        try:
            auc = roc_auc_score(y_test, y_prob_test)
        except Exception:
            pass

    brier = brier_score_loss(y_test, y_prob_test) if len(y_test) > 0 else float("nan")
    ece = compute_ece(y_test, y_prob_test)

    scores_by_problem: dict[int, np.ndarray] = {}
    idx = 0
    for pidx in test_problems:
        n = len(rows_by_problem.get(pidx, []))
        if n > 0:
            scores_by_problem[pidx] = y_prob_test[idx : idx + n]
            idx += n

    all_prob_scores: dict[int, np.ndarray] = {}
    all_X = []
    all_pidx_list = []
    all_counts = []
    for pidx in all_problems:
        rows = rows_by_problem.get(pidx, [])
        if not rows:
            continue
        feats = np.array([[r.get(c, 0.0) for c in LEARNABLE_FEATURE_COLS] for r in rows], dtype=float)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=-100.0)
        all_X.append(feats)
        all_pidx_list.append(pidx)
        all_counts.append(len(rows))

    if all_X:
        X_all = np.vstack(all_X)
        X_all_s = scaler.transform(X_all)
        if hasattr(clf, "predict_proba"):
            all_probs = clf.predict_proba(X_all_s)[:, 1]
        else:
            all_probs = clf.decision_function(X_all_s)

        offset = 0
        for pidx, cnt in zip(all_pidx_list, all_counts):
            all_prob_scores[pidx] = all_probs[offset : offset + cnt]
            offset += cnt

    metrics = {
        "auc": auc,
        "brier": brier,
        "ece": ece,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_pos_rate": float(np.mean(y_train)) if len(y_train) > 0 else 0.0,
        "test_pos_rate": float(np.mean(y_test)) if len(y_test) > 0 else 0.0,
    }

    return all_prob_scores, metrics


def train_cross_model_verifier(
    csv_path: Path,
    held_out_model: str,
    all_model_keys: list[str],
    classifier_type: str = "logistic",
) -> tuple[dict[int, np.ndarray], dict]:
    X_train_parts = []
    y_train_parts = []
    for mk in all_model_keys:
        if mk == held_out_model:
            continue
        rows_by_problem = load_features_for_model(csv_path, mk)
        all_problems = sorted(rows_by_problem.keys())
        X_part, y_part = _build_arrays(rows_by_problem, all_problems)
        if len(X_part) > 0:
            X_train_parts.append(X_part)
            y_train_parts.append(y_part)
        del rows_by_problem

    if not X_train_parts:
        return {}, {"auc": float("nan"), "brier": float("nan"), "ece": float("nan")}

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    del X_train_parts, y_train_parts

    held_rows = load_features_for_model(csv_path, held_out_model)
    held_problems = sorted(held_rows.keys())
    X_test, y_test = _build_arrays(held_rows, held_problems)

    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        scores = {}
        for pidx in held_problems:
            n = len(held_rows.get(pidx, []))
            scores[pidx] = np.full(n, 0.5)
        return scores, {"auc": float("nan"), "brier": float("nan"), "ece": float("nan")}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    del X_train

    clf = make_classifier(classifier_type)
    clf.fit(X_train_s, y_train)
    del X_train_s, y_train

    X_test_s = scaler.transform(X_test)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test_s)[:, 1]
    else:
        y_prob = clf.decision_function(X_test_s)

    auc = float("nan")
    if len(np.unique(y_test)) >= 2:
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            pass

    brier = brier_score_loss(y_test, y_prob) if len(y_test) > 0 else float("nan")
    ece = compute_ece(y_test, y_prob)

    scores: dict[int, np.ndarray] = {}
    offset = 0
    for pidx in held_problems:
        n = len(held_rows.get(pidx, []))
        if n > 0:
            scores[pidx] = y_prob[offset : offset + n]
            offset += n

    return scores, {"auc": auc, "brier": brier, "ece": ece}
