from typing import Any, Dict, Optional


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_continuity_metrics(
    comparison: Dict[str, Any], weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    summary = comparison.get("summary", {})
    concepts_a_count = summary.get("concepts_a", 0)
    shared_count = summary.get("shared", 0)
    jaccard = summary.get("jaccard", 0.0)
    transition_jaccard = summary.get("transition_jaccard", 0.0)
    topic_shift_flag = summary.get("topic_shift_flag", False)
    prereq_gap_rate = summary.get("prereq_gap_rate", 0.0)
    order_violation_rate = summary.get("order_violation_rate", 0.0)

    s_concept = (
        shared_count / max(1, concepts_a_count)
        if concepts_a_count
        else 0.0
    )
    s_concept = max(s_concept, jaccard)
    s_bridge = transition_jaccard
    if topic_shift_flag:
        s_bridge = min(s_bridge, 0.2)
    s_prereq = 1.0 - prereq_gap_rate
    s_sequence = 1.0 - order_violation_rate

    s_concept = _clamp(s_concept)
    s_bridge = _clamp(s_bridge)
    s_prereq = _clamp(s_prereq)
    s_sequence = _clamp(s_sequence)

    weights = weights or {
        "concept": 0.25,
        "bridge": 0.25,
        "prereq": 0.25,
        "sequence": 0.25,
    }
    lcs = (
        weights.get("concept", 0.25) * s_concept
        + weights.get("bridge", 0.25) * s_bridge
        + weights.get("prereq", 0.25) * s_prereq
        + weights.get("sequence", 0.25) * s_sequence
    )

    if lcs >= 0.75:
        label = "smooth"
    elif lcs >= 0.5:
        label = "moderate"
    else:
        label = "abrupt"

    return {
        "S_concept": round(s_concept, 3),
        "S_bridge": round(s_bridge, 3),
        "S_prereq": round(s_prereq, 3),
        "S_sequence": round(s_sequence, 3),
        "LCS": round(lcs, 3),
        "lcs_label": label,
    }
