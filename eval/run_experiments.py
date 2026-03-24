import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dataset.schema import CurriculumConcept, VideoItem
from dataset.utils import read_jsonl
from eval.metrics import (
    explained_confusion,
    explained_metrics,
    macro_average,
    precision_recall_f1,
    set_counts,
    set_metrics,
)
from graph_rag import ensure_video_graph, graph_enabled
from models.baselines.embedding_definition import EmbeddingDefinitionBaseline
from models.baselines.explanation_heuristic import ExplanationHeuristicBaseline
from models.baselines.llm_reasoning import LLMReasoningBaseline
from models.baselines.mention_only import MentionOnlyBaseline
from models.baselines.tfidf_definition import TFIDFDefinitionBaseline
from models.final.neo4j_model import Neo4jGraphGapModel
from utility import create_chunks, load_cached_transcript

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _load_curriculum(path: str) -> List[CurriculumConcept]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CurriculumConcept.from_dict(item) for item in payload.get("concepts", [])]


def _normalize_curriculum_id(curriculum_id: str) -> str:
    text = (curriculum_id or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _load_curricula_dir(path: str) -> Dict[str, List[CurriculumConcept]]:
    curricula_by_id: Dict[str, List[CurriculumConcept]] = {}
    normalized_index: Dict[str, List[CurriculumConcept]] = {}
    root = Path(path)
    for json_path in sorted(root.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        concepts = [
            CurriculumConcept.from_dict(item) for item in payload.get("concepts", [])
        ]
        curriculum_id = str(payload.get("curriculum_id", "")).strip()
        if curriculum_id:
            curricula_by_id[curriculum_id] = concepts
            norm_payload = _normalize_curriculum_id(curriculum_id)
            if norm_payload and norm_payload not in normalized_index:
                normalized_index[norm_payload] = concepts

        stem_id = json_path.stem.strip()
        norm_stem = _normalize_curriculum_id(stem_id)
        if norm_stem and norm_stem not in normalized_index:
            normalized_index[norm_stem] = concepts

    for key, value in normalized_index.items():
        curricula_by_id.setdefault(key, value)
    return curricula_by_id


def _load_labels(path: str) -> Dict[str, Dict[str, int]]:
    labels_by_video: Dict[str, Dict[str, int]] = {}
    for row in read_jsonl(path):
        video_id = row["video_id"]
        concept_id = row["concept_id"]
        labels_by_video.setdefault(video_id, {})[concept_id] = row.get(
            "explained_label", 0
        )
    return labels_by_video


def _missing_truth(
    explained_labels: Dict[str, int], curriculum: List[CurriculumConcept]
) -> List[str]:
    missing = set()
    explained_ids = {cid for cid, val in explained_labels.items() if val == 1}
    for concept in curriculum:
        if concept.concept_id not in explained_ids:
            continue
        for prereq in concept.prerequisites:
            if explained_labels.get(prereq, 0) != 1:
                missing.add(prereq)
    return sorted(missing)


def _load_transcript(video: VideoItem) -> str:
    if video.transcript_path and Path(video.transcript_path).exists():
        return Path(video.transcript_path).read_text(encoding="utf-8")
    cached = load_cached_transcript(video.video_id)
    return cached or ""


def _model_registry():
    return {
        "mention_only": MentionOnlyBaseline(),
        "explanation_heuristic": ExplanationHeuristicBaseline(),
        "tfidf_definition": TFIDFDefinitionBaseline(),
        "embedding_definition": EmbeddingDefinitionBaseline(),
        "llm_reasoning": LLMReasoningBaseline(),
        "neo4j_graph": Neo4jGraphGapModel(),
    }


def run_experiments(config: dict) -> dict:
    args = argparse.Namespace(**config)
    configure_logging(getattr(args, "log_level", "INFO"))
    use_progress = not bool(getattr(args, "no_progress", False))
    if use_progress and tqdm is None:
        LOGGER.warning(
            "tqdm is not installed; install it with 'pip install tqdm' to enable progress bars."
        )
        use_progress = False

    has_single_curriculum = bool(getattr(args, "curriculum", None))
    has_curricula_dir = bool(getattr(args, "curricula_dir", None))
    if has_single_curriculum and has_curricula_dir:
        raise ValueError("Use either --curriculum or --curricula-dir, not both.")
    if not has_single_curriculum and not has_curricula_dir:
        raise ValueError("One of --curriculum or --curricula-dir is required.")

    curriculum = _load_curriculum(args.curriculum) if has_single_curriculum else []
    curricula_by_id = (
        _load_curricula_dir(args.curricula_dir) if has_curricula_dir else {}
    )
    labels = _load_labels(args.labels)
    videos = [VideoItem.from_dict(item) for item in read_jsonl(args.videos)]
    chunker_config = json.loads(args.chunker_config)
    LOGGER.info(
        "Loaded videos=%s labels=%s curriculum_mode=%s",
        len(videos),
        len(labels),
        "single" if has_single_curriculum else "folder",
    )

    active_videos: List[VideoItem] = []
    curriculum_by_video: Dict[str, List[CurriculumConcept]] = {}
    skipped_missing_curriculum = 0
    normalized_match_hits = 0
    missing_curriculum_ids = set()

    for video in videos:
        active_curriculum = curriculum
        if curricula_by_id:
            active_curriculum = curricula_by_id.get(video.curriculum_id, [])
            if not active_curriculum:
                normalized_id = _normalize_curriculum_id(video.curriculum_id)
                if normalized_id:
                    active_curriculum = curricula_by_id.get(normalized_id, [])
                    if active_curriculum:
                        normalized_match_hits += 1
            if not active_curriculum:
                skipped_missing_curriculum += 1
                if video.curriculum_id:
                    missing_curriculum_ids.add(video.curriculum_id)
                continue

        curriculum_by_video[video.video_id] = active_curriculum
        active_videos.append(video)

    if curricula_by_id:
        LOGGER.info(
            "Curriculum mapping: active_videos=%s skipped_missing_curriculum=%s normalized_matches=%s",
            len(active_videos),
            skipped_missing_curriculum,
            normalized_match_hits,
        )
        if missing_curriculum_ids:
            LOGGER.warning(
                "Missing curriculum_id values: %s",
                ", ".join(sorted(missing_curriculum_ids)),
            )

    model_registry = _model_registry()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown_models = [name for name in model_names if name not in model_registry]
    if unknown_models:
        raise ValueError(f"Unknown models requested: {', '.join(unknown_models)}")

    run_id = config.get("run_id") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Run id: %s | output dir: %s", run_id, out_dir)

    if getattr(args, "build_graph", False):
        if not graph_enabled():
            raise RuntimeError("Neo4j is not configured for --build-graph.")
        LOGGER.info("Building/validating Neo4j graph for %s videos", len(active_videos))
        graph_iter = (
            tqdm(active_videos, desc="Building graph", unit="video")
            if use_progress
            else active_videos
        )
        built_graph_for = 0
        skipped_graph_missing_transcript = 0
        for video in graph_iter:
            transcript = _load_transcript(video)
            if not transcript:
                skipped_graph_missing_transcript += 1
                continue
            ensure_video_graph(
                video.video_id,
                transcript,
                video.language,
                rebuild=getattr(args, "graph_rebuild", False),
                chunker_kind=args.chunker,
                chunker_config=chunker_config,
            )
            built_graph_for += 1
        LOGGER.info(
            "Graph stage done: built_or_checked=%s skipped_no_transcript=%s",
            built_graph_for,
            skipped_graph_missing_transcript,
        )

    metrics_rows = []
    model_iter = (
        tqdm(model_names, desc="Running models", unit="model")
        if use_progress
        else model_names
    )
    for model_name in model_iter:
        LOGGER.info("Running model: %s", model_name)
        model = model_registry[model_name]
        predictions_path = out_dir / f"predictions_{model_name}.jsonl"
        pred_rows = []
        set_scores = []
        set_scores_nonempty_truth = []
        explained_scores = []
        explained_scores_nonempty_truth = []
        missing_tp_total = 0
        missing_fp_total = 0
        missing_fn_total = 0
        explained_tp_total = 0
        explained_fp_total = 0
        explained_fn_total = 0
        explained_tn_total = 0
        per_model_iter = (
            tqdm(
                active_videos,
                desc=f"{model_name} videos",
                unit="video",
                leave=False,
            )
            if use_progress
            else active_videos
        )
        processed_videos = 0
        skipped_no_transcript = 0
        for video in per_model_iter:
            active_curriculum = curriculum_by_video[video.video_id]
            transcript = _load_transcript(video)
            if not transcript:
                skipped_no_transcript += 1
                continue
            docs = create_chunks(
                transcript, chunker_kind=args.chunker, chunker_config=chunker_config
            )
            chunks = [
                {"chunk_index": doc.metadata.get("chunk_index", idx), "text": doc.page_content}
                for idx, doc in enumerate(docs)
            ]
            result = model.predict(
                video.video_id, transcript, chunks, active_curriculum
            )
            explained_truth = labels.get(video.video_id, {})
            missing_truth = _missing_truth(explained_truth, active_curriculum)
            missing_pred = [
                cid for cid, score in result.missing.items() if score >= 0.5
            ]
            set_metric = set_metrics(missing_truth, missing_pred)
            set_scores.append(set_metric)
            if missing_truth:
                set_scores_nonempty_truth.append(set_metric)

            explained_metric = explained_metrics(explained_truth, result.explained)
            explained_scores.append(explained_metric)
            if any(val == 1 for val in explained_truth.values()):
                explained_scores_nonempty_truth.append(explained_metric)

            tp_set, fp_set, fn_set = set_counts(missing_truth, missing_pred)
            missing_tp_total += tp_set
            missing_fp_total += fp_set
            missing_fn_total += fn_set

            tp_exp, fp_exp, fn_exp, tn_exp = explained_confusion(
                explained_truth, result.explained
            )
            explained_tp_total += tp_exp
            explained_fp_total += fp_exp
            explained_fn_total += fn_exp
            explained_tn_total += tn_exp
            processed_videos += 1

            pred_rows.append(
                {
                    "video_id": video.video_id,
                    "model": model_name,
                    "missing_pred": missing_pred,
                    "explained_scores": result.explained,
                }
            )

        with open(predictions_path, "w", encoding="utf-8") as handle:
            for row in pred_rows:
                handle.write(json.dumps(row) + "\n")

        set_macro = macro_average(set_scores)
        set_macro_nonempty = macro_average(set_scores_nonempty_truth)
        set_micro = precision_recall_f1(
            missing_tp_total, missing_fp_total, missing_fn_total
        )
        explained_macro = macro_average(explained_scores)
        explained_macro_nonempty = macro_average(explained_scores_nonempty_truth)
        explained_micro = precision_recall_f1(
            explained_tp_total, explained_fp_total, explained_fn_total
        )
        explained_micro["accuracy"] = (
            (explained_tp_total + explained_tn_total)
            / (
                explained_tp_total
                + explained_tn_total
                + explained_fp_total
                + explained_fn_total
            )
            if (
                explained_tp_total
                + explained_tn_total
                + explained_fp_total
                + explained_fn_total
            )
            else 0.0
        )
        metric_row = {
            "model": model_name,
            "missing_precision": round(set_macro.get("precision", 0.0), 3),
            "missing_recall": round(set_macro.get("recall", 0.0), 3),
            "missing_f1": round(set_macro.get("f1", 0.0), 3),
            "missing_precision_macro_nonempty": round(
                set_macro_nonempty.get("precision", 0.0), 3
            ),
            "missing_recall_macro_nonempty": round(
                set_macro_nonempty.get("recall", 0.0), 3
            ),
            "missing_f1_macro_nonempty": round(set_macro_nonempty.get("f1", 0.0), 3),
            "missing_precision_micro": round(set_micro.get("precision", 0.0), 3),
            "missing_recall_micro": round(set_micro.get("recall", 0.0), 3),
            "missing_f1_micro": round(set_micro.get("f1", 0.0), 3),
            "explained_precision": round(explained_macro.get("precision", 0.0), 3),
            "explained_recall": round(explained_macro.get("recall", 0.0), 3),
            "explained_f1": round(explained_macro.get("f1", 0.0), 3),
            "explained_accuracy": round(explained_macro.get("accuracy", 0.0), 3),
            "explained_precision_macro_nonempty": round(
                explained_macro_nonempty.get("precision", 0.0), 3
            ),
            "explained_recall_macro_nonempty": round(
                explained_macro_nonempty.get("recall", 0.0), 3
            ),
            "explained_f1_macro_nonempty": round(
                explained_macro_nonempty.get("f1", 0.0), 3
            ),
            "explained_accuracy_macro_nonempty": round(
                explained_macro_nonempty.get("accuracy", 0.0), 3
            ),
            "explained_precision_micro": round(explained_micro.get("precision", 0.0), 3),
            "explained_recall_micro": round(explained_micro.get("recall", 0.0), 3),
            "explained_f1_micro": round(explained_micro.get("f1", 0.0), 3),
            "explained_accuracy_micro": round(explained_micro.get("accuracy", 0.0), 3),
        }
        metrics_rows.append(metric_row)
        LOGGER.info(
            "Model done: %s | processed=%s skipped_no_transcript=%s | missing_f1=%s explained_f1=%s",
            model_name,
            processed_videos,
            skipped_no_transcript,
            metric_row["missing_f1"],
            metric_row["explained_f1"],
        )

    metrics_path = out_dir / "metrics.csv"
    if metrics_rows:
        keys = list(metrics_rows[0].keys())
        with open(metrics_path, "w", encoding="utf-8") as handle:
            handle.write(",".join(keys) + "\n")
            for row in metrics_rows:
                handle.write(",".join(str(row[key]) for key in keys) + "\n")
        LOGGER.info("Metrics written to %s", metrics_path)

    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# Experiment Summary\n\n")
        if curricula_by_id:
            handle.write(
                f"- videos_skipped_missing_curriculum: {skipped_missing_curriculum}\n"
            )
            handle.write(f"- videos_matched_normalized_id: {normalized_match_hits}\n")
            if missing_curriculum_ids:
                joined = ", ".join(sorted(missing_curriculum_ids))
                handle.write(f"- missing_curriculum_ids: {joined}\n")
            handle.write("\n")
        for row in metrics_rows:
            handle.write(
                f"- {row['model']}: missing_f1={row['missing_f1']} | "
                f"explained_f1={row['explained_f1']}\n"
            )
    LOGGER.info("Summary written to %s", summary_path)
    LOGGER.info("Run completed with %s models", len(metrics_rows))
    return {"run_id": run_id, "out_dir": str(out_dir), "metrics": metrics_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prerequisite gap experiments.")
    parser.add_argument("--videos", required=True, help="videos.jsonl path")
    curriculum_group = parser.add_mutually_exclusive_group(required=True)
    curriculum_group.add_argument("--curriculum", help="curriculum json path")
    curriculum_group.add_argument(
        "--curricula-dir",
        help="directory of curriculum json files keyed by curriculum_id",
    )
    parser.add_argument("--labels", required=True, help="labels jsonl path")
    parser.add_argument("--chunker", default="char", help="char|semantic")
    parser.add_argument("--chunker-config", default="{}", help="JSON chunker config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--models",
        default="mention_only,explanation_heuristic,tfidf_definition,embedding_definition,llm_reasoning,neo4j_graph",
        help="Comma-separated model list",
    )
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Build Neo4j graphs before running models (required for neo4j_graph).",
    )
    parser.add_argument(
        "--graph-rebuild",
        action="store_true",
        help="Force rebuild of Neo4j graphs when using --build-graph.",
    )
    args = parser.parse_args()

    run_experiments(vars(args))


if __name__ == "__main__":
    main()
