import argparse
import json
from pathlib import Path
from typing import List

from dataset.labeling import AgreementLabeler, HeuristicLabeler, LLMJudgeLabeler
from dataset.schema import CurriculumConcept, VideoItem
from dataset.utils import read_jsonl, write_jsonl
from utility import create_chunks, load_cached_transcript


def _load_curriculum(path: str) -> List[CurriculumConcept]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    concepts = payload.get("concepts", [])
    return [CurriculumConcept.from_dict(item) for item in concepts]


def _load_transcript(video: VideoItem) -> str:
    if video.transcript_path and Path(video.transcript_path).exists():
        return Path(video.transcript_path).read_text(encoding="utf-8")
    cached = load_cached_transcript(video.video_id)
    if cached:
        return cached
    return ""


def _labeler_from_kind(kind: str):
    if kind == "heuristic":
        return HeuristicLabeler()
    if kind == "llm":
        return LLMJudgeLabeler()
    if kind == "agreement":
        return AgreementLabeler()
    raise ValueError(f"Unknown labeler: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ground truth labels.")
    parser.add_argument("--videos", required=True, help="Path to videos.jsonl")
    parser.add_argument("--curriculum", required=True, help="Curriculum JSON path")
    parser.add_argument("--chunker", default="char", help="char|semantic")
    parser.add_argument(
        "--chunker-config",
        default="{}",
        help="JSON dict for chunker config",
    )
    parser.add_argument(
        "--labeler",
        default="heuristic",
        choices=["heuristic", "llm", "agreement"],
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL for concept coverage labels",
    )
    args = parser.parse_args()

    chunker_config = json.loads(args.chunker_config)
    labeler = _labeler_from_kind(args.labeler)
    curriculum = _load_curriculum(args.curriculum)
    videos_raw = read_jsonl(args.videos)
    videos = [VideoItem.from_dict(item) for item in videos_raw]

    all_labels = []
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
        labels = labeler.label(video, curriculum, chunks)
        all_labels.extend([label.to_dict() for label in labels])

    write_jsonl(args.out, all_labels)


if __name__ == "__main__":
    main()
