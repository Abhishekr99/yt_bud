import csv
from datetime import datetime
from typing import Any, Dict


def build_row(video_a: str, video_b: str, comparison: Dict[str, Any]) -> Dict[str, Any]:
    summary = comparison.get("summary", {})
    row = {
        "videoA_id": video_a,
        "videoB_id": video_b,
        "concepts_a_count": summary.get("concepts_a", 0),
        "concepts_b_count": summary.get("concepts_b", 0),
        "shared_count": summary.get("shared", 0),
        "jaccard_full": summary.get("jaccard", 0.0),
        "transition_jaccard": summary.get("transition_jaccard", 0.0),
        "transition_embedding_similarity": summary.get(
            "transition_embedding_similarity", 0.0
        ),
        "tfidf_similarity_full": summary.get("tfidf_similarity_full", 0.0),
        "embedding_similarity_full": summary.get("embedding_similarity_full", 0.0),
        "new_in_b_count": summary.get("new_in_b_count", 0),
        "new_concept_rate": summary.get("new_concept_rate", 0.0),
        "prereq_gaps_count": summary.get("prereq_gaps_count", 0),
        "prereq_gap_rate": summary.get("prereq_gap_rate", 0.0),
        "prereq_edges_considered_b": summary.get("prereq_edges_considered_b", 0),
        "order_violations_count": summary.get("order_violations_count", 0),
        "order_violation_rate": summary.get("order_violation_rate", 0.0),
        "topic_shift_flag": summary.get("topic_shift_flag", False),
        "S_concept": summary.get("S_concept", 0.0),
        "S_bridge": summary.get("S_bridge", 0.0),
        "S_prereq": summary.get("S_prereq", 0.0),
        "S_sequence": summary.get("S_sequence", 0.0),
        "LCS": summary.get("LCS", 0.0),
        "lcs_label": summary.get("lcs_label", ""),
        "timestamp": datetime.utcnow().isoformat(),
    }
    return row


def write_row(csv_path: str, row: Dict[str, Any]) -> None:
    fieldnames = list(row.keys())
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing = next(reader, None)
            if existing:
                fieldnames = existing
    except FileNotFoundError:
        pass

    file_exists = False
    try:
        with open(csv_path, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
