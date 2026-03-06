from typing import Dict, Iterable, List, Tuple


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def set_metrics(true_set: Iterable[str], pred_set: Iterable[str]) -> Dict[str, float]:
    true_set = set(true_set)
    pred_set = set(pred_set)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    return precision_recall_f1(tp, fp, fn)


def explained_metrics(
    true_labels: Dict[str, int],
    pred_scores: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for concept_id, true_label in true_labels.items():
        pred = 1 if pred_scores.get(concept_id, 0.0) >= threshold else 0
        if true_label == 1 and pred == 1:
            tp += 1
        elif true_label == 0 and pred == 1:
            fp += 1
        elif true_label == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    metrics = precision_recall_f1(tp, fp, fn)
    metrics["accuracy"] = _safe_div(tp + tn, tp + tn + fp + fn)
    return metrics


def macro_average(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = rows[0].keys()
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}


def bucket_by_frequency(
    concept_freq: Dict[str, int],
    true_labels: Dict[str, int],
    pred_scores: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    buckets = {"rare": [], "mid": [], "frequent": []}
    for concept_id, freq in concept_freq.items():
        if freq <= 1:
            buckets["rare"].append(concept_id)
        elif freq <= 4:
            buckets["mid"].append(concept_id)
        else:
            buckets["frequent"].append(concept_id)

    results = {}
    for name, ids in buckets.items():
        if not ids:
            results[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        sub_true = {cid: true_labels.get(cid, 0) for cid in ids}
        sub_pred = {cid: pred_scores.get(cid, 0.0) for cid in ids}
        results[name] = explained_metrics(sub_true, sub_pred, threshold)
    return results
