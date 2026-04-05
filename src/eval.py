"""
Core evaluation metrics library for Relation Extraction.

Pure functions — no I/O, no side effects.
All metrics follow sklearn conventions but are implemented from scratch.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_relation(text: str) -> str:
    """Normalise a generated relation label for comparison.

    Steps:
      1. Convert to string, strip outer whitespace.
      2. Take only the *first line* (models sometimes emit multi-line text).
      3. Lowercase.
      4. Collapse internal whitespace runs to single space.
      5. Strip common quoting artifacts (" or ').
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip()
    # First line only
    s = s.split("\n")[0].strip()
    # Lowercase
    s = s.lower()
    # Collapse whitespace
    s = _WHITESPACE_RE.sub(" ", s)
    # Strip surrounding quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


# ---------------------------------------------------------------------------
# Low-level metric helpers
# ---------------------------------------------------------------------------


def _confusion_counts(
    gold: List[str],
    pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """Compute per-class TP, FP, FN counts.

    Parameters
    ----------
    gold : list[str]   — ground-truth labels (normalised)
    pred : list[str]   — predicted labels (normalised)
    labels : list[str]  — if given, restrict to these classes

    Returns
    -------
    dict  label -> {"tp": int, "fp": int, "fn": int, "support": int}
    """
    if labels is None:
        labels_set = set(gold) | set(pred)
    else:
        labels_set = set(labels)

    counts: Dict[str, Dict[str, int]] = {
        lab: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lab in labels_set
    }

    for g, p in zip(gold, pred):
        if g in labels_set:
            counts[g]["support"] += 1
        if g == p:
            if g in labels_set:
                counts[g]["tp"] += 1
        else:
            if g in labels_set:
                counts[g]["fn"] += 1
            if p in labels_set:
                counts[p]["fp"] += 1

    return counts


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Micro-averaged metrics
# ---------------------------------------------------------------------------


def compute_micro_metrics(
    gold: List[str],
    pred: List[str],
    exclude_labels: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Compute micro-averaged Precision, Recall, F1.

    For single-label classification micro-P = micro-R = micro-F1 = accuracy,
    but we compute via the TP/FP/FN route so the function generalises.

    Parameters
    ----------
    gold, pred : parallel lists of normalised labels
    exclude_labels : labels to drop from *both* gold and pred rows before eval

    Returns
    -------
    dict with keys: micro_precision, micro_recall, micro_f1, accuracy, support
    """
    assert len(gold) == len(pred), f"Length mismatch: {len(gold)} vs {len(pred)}"

    if exclude_labels:
        pairs = [
            (g, p) for g, p in zip(gold, pred) if g not in exclude_labels
        ]
        if not pairs:
            return {
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                "accuracy": 0.0,
                "support": 0,
            }
        gold, pred = zip(*pairs)
        gold, pred = list(gold), list(pred)

    counts = _confusion_counts(gold, pred)

    total_tp = sum(c["tp"] for c in counts.values())
    total_fp = sum(c["fp"] for c in counts.values())
    total_fn = sum(c["fn"] for c in counts.values())

    precision = _safe_div(total_tp, total_tp + total_fp)
    recall = _safe_div(total_tp, total_tp + total_fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(total_tp, len(gold))

    return {
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "micro_f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
        "support": len(gold),
    }


# ---------------------------------------------------------------------------
# Macro-averaged metrics
# ---------------------------------------------------------------------------


def compute_macro_metrics(
    gold: List[str],
    pred: List[str],
    exclude_labels: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Compute macro-averaged Precision, Recall, F1.

    Each class contributes equally regardless of support.
    """
    if exclude_labels:
        pairs = [(g, p) for g, p in zip(gold, pred) if g not in exclude_labels]
        if not pairs:
            return {
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
            }
        gold, pred = zip(*pairs)
        gold, pred = list(gold), list(pred)

    # Only macro-average over classes that appear in gold
    gold_labels = sorted(set(gold))
    counts = _confusion_counts(gold, pred, labels=gold_labels)

    precisions, recalls, f1s = [], [], []
    for lab in gold_labels:
        c = counts[lab]
        p = _safe_div(c["tp"], c["tp"] + c["fp"])
        r = _safe_div(c["tp"], c["tp"] + c["fn"])
        f = _safe_div(2 * p * r, p + r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "macro_precision": round(float(np.mean(precisions)), 6) if precisions else 0.0,
        "macro_recall": round(float(np.mean(recalls)), 6) if recalls else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 6) if f1s else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-class breakdown
# ---------------------------------------------------------------------------


def compute_per_class_metrics(
    gold: List[str],
    pred: List[str],
    exclude_labels: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Return a DataFrame with per-class P / R / F1 / support.

    Rows are sorted by support (descending).
    """
    if exclude_labels:
        pairs = [(g, p) for g, p in zip(gold, pred) if g not in exclude_labels]
        if not pairs:
            return pd.DataFrame(
                columns=["label", "precision", "recall", "f1", "support"]
            )
        gold, pred = zip(*pairs)
        gold, pred = list(gold), list(pred)

    gold_labels = sorted(set(gold))
    counts = _confusion_counts(gold, pred, labels=gold_labels)

    rows = []
    for lab in gold_labels:
        c = counts[lab]
        p = _safe_div(c["tp"], c["tp"] + c["fp"])
        r = _safe_div(c["tp"], c["tp"] + c["fn"])
        f = _safe_div(2 * p * r, p + r)
        rows.append(
            {
                "label": lab,
                "precision": round(p, 6),
                "recall": round(r, 6),
                "f1": round(f, 6),
                "support": c["support"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("support", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Schema / output quality metrics
# ---------------------------------------------------------------------------


def compute_schema_valid_rate(
    pred: List[str],
    allowed_labels: Set[str],
) -> float:
    """Fraction of predictions that match a label in the allowed set.

    Both sides are normalised before comparison.
    """
    if not pred:
        return 0.0
    allowed_norm = {normalize_relation(lab) for lab in allowed_labels}
    valid = sum(1 for p in pred if normalize_relation(p) in allowed_norm)
    return round(valid / len(pred), 6)


def compute_malformed_rate(pred: List[str]) -> float:
    """Fraction of predictions that are null, empty, or clearly not a label.

    'Malformed' = null/NaN, empty string, or length > 60 chars after
    normalisation (a relation label should never be that long).
    """
    if not pred:
        return 0.0
    bad = 0
    for p in pred:
        norm = normalize_relation(p)
        if norm == "" or len(norm) > 60:
            bad += 1
    return round(bad / len(pred), 6)


# ---------------------------------------------------------------------------
# Top confused label pairs
# ---------------------------------------------------------------------------


def top_confused_pairs(
    gold: List[str],
    pred: List[str],
    top_k: int = 10,
    exclude_labels: Optional[Set[str]] = None,
) -> List[Tuple[str, str, int]]:
    """Return the most frequent (gold, pred) mismatch pairs.

    Useful for identifying near-neighbour label confusions.
    """
    if exclude_labels:
        pairs = [(g, p) for g, p in zip(gold, pred) if g not in exclude_labels]
    else:
        pairs = list(zip(gold, pred))

    confusion = Counter()
    for g, p in pairs:
        if g != p:
            confusion[(g, p)] += 1

    top = confusion.most_common(top_k)
    return [(g, p, cnt) for (g, p), cnt in top]


# ---------------------------------------------------------------------------
# Full evaluation for one slice
# ---------------------------------------------------------------------------


def evaluate_slice(
    gold: List[str],
    pred: List[str],
    allowed_labels: Optional[Set[str]] = None,
    exclude_labels: Optional[Set[str]] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """Run all metrics on a single (model_config × dataset) slice.

    Parameters
    ----------
    gold : raw ground-truth labels
    pred : raw generated labels
    allowed_labels : the dataset's valid label set (for schema-valid rate)
    exclude_labels : labels to exclude from P/R/F1 computation
    normalize : whether to apply normalize_relation() first

    Returns
    -------
    dict with all Tier-1 and Tier-2 metrics
    """
    if normalize:
        gold_n = [normalize_relation(g) for g in gold]
        pred_n = [normalize_relation(p) for p in pred]
    else:
        gold_n = list(gold)
        pred_n = list(pred)

    excl = None
    if exclude_labels:
        excl = {normalize_relation(e) for e in exclude_labels} if normalize else exclude_labels

    micro = compute_micro_metrics(gold_n, pred_n, exclude_labels=excl)
    macro = compute_macro_metrics(gold_n, pred_n, exclude_labels=excl)

    result = {**micro, **macro}

    # Schema-valid rate (computed on raw pred, not normalised gold-filtered)
    if allowed_labels is not None:
        result["schema_valid_rate"] = compute_schema_valid_rate(pred, allowed_labels)
    else:
        result["schema_valid_rate"] = float("nan")

    result["malformed_rate"] = compute_malformed_rate(pred)

    return result
