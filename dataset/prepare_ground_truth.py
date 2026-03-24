import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from dataset.labeling import AgreementLabeler, HeuristicLabeler, LLMJudgeLabeler
from dataset.schema import CurriculumConcept, VideoItem
from dataset.utils import read_jsonl, write_jsonl
from utility import create_chunks, load_cached_transcript

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _load_curriculum(path: str) -> List[CurriculumConcept]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    concepts = payload.get("concepts", [])
    return [CurriculumConcept.from_dict(item) for item in concepts]


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
        curriculum_id = str(payload.get("curriculum_id", "")).strip()
        concepts = payload.get("concepts", [])
        parsed_concepts = [CurriculumConcept.from_dict(item) for item in concepts]
        if curriculum_id:
            curricula_by_id[curriculum_id] = parsed_concepts
            norm_payload_id = _normalize_curriculum_id(curriculum_id)
            if norm_payload_id and norm_payload_id not in normalized_index:
                normalized_index[norm_payload_id] = parsed_concepts

        # Also index by filename stem so files named as <curriculum_id>.json work
        # even if payload curriculum_id is stale.
        stem_id = json_path.stem.strip()
        norm_stem_id = _normalize_curriculum_id(stem_id)
        if norm_stem_id and norm_stem_id not in normalized_index:
            normalized_index[norm_stem_id] = parsed_concepts

    # Merge normalized aliases into direct lookup map for simpler runtime matching.
    for key, value in normalized_index.items():
        curricula_by_id.setdefault(key, value)
    return curricula_by_id


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
    curriculum_group = parser.add_mutually_exclusive_group(required=True)
    curriculum_group.add_argument(
        "--curriculum", help="Single curriculum JSON path (applied to all videos)"
    )
    curriculum_group.add_argument(
        "--curricula-dir",
        help="Directory of curriculum JSON files keyed by curriculum_id",
    )
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    chunker_config = json.loads(args.chunker_config)
    labeler = _labeler_from_kind(args.labeler)
    curriculum = _load_curriculum(args.curriculum) if args.curriculum else []
    curricula_by_id = (
        _load_curricula_dir(args.curricula_dir) if args.curricula_dir else {}
    )
    videos_raw = read_jsonl(args.videos)
    videos = [VideoItem.from_dict(item) for item in videos_raw]
    LOGGER.info(
        "Loaded videos=%s | labeler=%s | curriculum_mode=%s",
        len(videos),
        args.labeler,
        "single" if args.curriculum else "folder",
    )
    if args.curriculum:
        LOGGER.info("Single curriculum concepts=%s", len(curriculum))
    if args.curricula_dir:
        LOGGER.info("Loaded curricula from dir: %s (keys=%s)", args.curricula_dir, len(curricula_by_id))

    use_progress = not args.no_progress
    if use_progress and tqdm is None:
        LOGGER.warning(
            "tqdm is not installed; install it with 'pip install tqdm' to enable progress bars."
        )
        use_progress = False

    all_labels = []
    skipped_missing_curriculum = 0
    missing_curriculum_ids = set()
    normalized_match_hits = 0
    skipped_missing_transcript = 0
    labeled_videos = 0

    iterable = tqdm(videos, desc="Labeling videos", unit="video") if use_progress else videos
    for video in iterable:
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
                if use_progress:
                    iterable.set_postfix(
                        labels=len(all_labels),
                        skipped_curr=skipped_missing_curriculum,
                        skipped_tx=skipped_missing_transcript,
                    )
                continue
        transcript = _load_transcript(video)
        if not transcript:
            skipped_missing_transcript += 1
            if use_progress:
                iterable.set_postfix(
                    labels=len(all_labels),
                    skipped_curr=skipped_missing_curriculum,
                    skipped_tx=skipped_missing_transcript,
                )
            continue
        docs = create_chunks(
            transcript, chunker_kind=args.chunker, chunker_config=chunker_config
        )
        chunks = [
            {"chunk_index": doc.metadata.get("chunk_index", idx), "text": doc.page_content}
            for idx, doc in enumerate(docs)
        ]
        labels = labeler.label(video, active_curriculum, chunks)
        all_labels.extend([label.to_dict() for label in labels])
        labeled_videos += 1
        if use_progress:
            iterable.set_postfix(
                labels=len(all_labels),
                skipped_curr=skipped_missing_curriculum,
                skipped_tx=skipped_missing_transcript,
            )

    write_jsonl(args.out, all_labels)
    LOGGER.info("Wrote %s labels to %s", len(all_labels), args.out)
    LOGGER.info(
        "Summary: labeled_videos=%s | skipped_missing_curriculum=%s | skipped_missing_transcript=%s | normalized_matches=%s",
        labeled_videos,
        skipped_missing_curriculum,
        skipped_missing_transcript,
        normalized_match_hits,
    )
    if curricula_by_id:
        print(
            f"Videos skipped due to missing curriculum file: {skipped_missing_curriculum}"
        )
        print(f"Videos matched via normalized curriculum_id: {normalized_match_hits}")
        if missing_curriculum_ids:
            print(
                "Missing curriculum_id values: "
                + ", ".join(sorted(missing_curriculum_ids))
            )


if __name__ == "__main__":
    main()
