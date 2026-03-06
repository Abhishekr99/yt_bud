import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dataset.schema import CurriculumConcept, VideoItem
from dataset.utils import read_jsonl
from eval.metrics import explained_metrics, macro_average, set_metrics
from graph_rag import ensure_video_graph, graph_enabled
from models.baselines.embedding_definition import EmbeddingDefinitionBaseline
from models.baselines.explanation_heuristic import ExplanationHeuristicBaseline
from models.baselines.llm_reasoning import LLMReasoningBaseline
from models.baselines.mention_only import MentionOnlyBaseline
from models.baselines.tfidf_definition import TFIDFDefinitionBaseline
from models.final.neo4j_model import Neo4jGraphGapModel
from utility import create_chunks, load_cached_transcript


def _load_curriculum(path: str) -> List[CurriculumConcept]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CurriculumConcept.from_dict(item) for item in payload.get("concepts", [])]


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
    curriculum = _load_curriculum(args.curriculum)
    labels = _load_labels(args.labels)
    videos = [VideoItem.from_dict(item) for item in read_jsonl(args.videos)]
    chunker_config = json.loads(args.chunker_config)

    model_registry = _model_registry()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    run_id = config.get("run_id") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "build_graph", False):
        if not graph_enabled():
            raise RuntimeError("Neo4j is not configured for --build-graph.")
        for video in videos:
            transcript = _load_transcript(video)
            if not transcript:
                continue
            ensure_video_graph(
                video.video_id,
                transcript,
                video.language,
                rebuild=getattr(args, "graph_rebuild", False),
                chunker_kind=args.chunker,
                chunker_config=chunker_config,
            )

    metrics_rows = []
    for model_name in model_names:
        model = model_registry[model_name]
        predictions_path = out_dir / f"predictions_{model_name}.jsonl"
        pred_rows = []
        set_scores = []
        explained_scores = []
        for video in videos:
            transcript = _load_transcript(video)
            if not transcript:
                continue
            docs = create_chunks(
                transcript, chunker_kind=args.chunker, chunker_config=chunker_config
            )
            chunks = [
                {"chunk_index": doc.metadata.get("chunk_index", idx), "text": doc.page_content}
                for idx, doc in enumerate(docs)
            ]
            result = model.predict(video.video_id, transcript, chunks, curriculum)
            explained_truth = labels.get(video.video_id, {})
            missing_truth = _missing_truth(explained_truth, curriculum)
            missing_pred = [
                cid for cid, score in result.missing.items() if score >= 0.5
            ]
            set_scores.append(set_metrics(missing_truth, missing_pred))
            explained_scores.append(explained_metrics(explained_truth, result.explained))

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
        explained_macro = macro_average(explained_scores)
        metrics_rows.append(
            {
                "model": model_name,
                "missing_precision": round(set_macro.get("precision", 0.0), 3),
                "missing_recall": round(set_macro.get("recall", 0.0), 3),
                "missing_f1": round(set_macro.get("f1", 0.0), 3),
                "explained_precision": round(explained_macro.get("precision", 0.0), 3),
                "explained_recall": round(explained_macro.get("recall", 0.0), 3),
                "explained_f1": round(explained_macro.get("f1", 0.0), 3),
                "explained_accuracy": round(explained_macro.get("accuracy", 0.0), 3),
            }
        )

    metrics_path = out_dir / "metrics.csv"
    if metrics_rows:
        keys = list(metrics_rows[0].keys())
        with open(metrics_path, "w", encoding="utf-8") as handle:
            handle.write(",".join(keys) + "\n")
            for row in metrics_rows:
                handle.write(",".join(str(row[key]) for key in keys) + "\n")

    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# Experiment Summary\n\n")
        for row in metrics_rows:
            handle.write(
                f"- {row['model']}: missing_f1={row['missing_f1']} | "
                f"explained_f1={row['explained_f1']}\n"
            )
    return {"run_id": run_id, "out_dir": str(out_dir), "metrics": metrics_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prerequisite gap experiments.")
    parser.add_argument("--videos", required=True, help="videos.jsonl path")
    parser.add_argument("--curriculum", required=True, help="curriculum json path")
    parser.add_argument("--labels", required=True, help="labels jsonl path")
    parser.add_argument("--chunker", default="char", help="char|semantic")
    parser.add_argument("--chunker-config", default="{}", help="JSON chunker config")
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
