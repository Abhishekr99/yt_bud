import argparse
import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from youtube_transcript_api import YouTubeTranscriptApi

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - depends on optional dependency
    tqdm = None


LOGGER = logging.getLogger(__name__)


def is_ip_block_error(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    markers = (
        "youtube is blocking requests from your ip",
        "requestblocked",
        "ipblocked",
        "could not retrieve a transcript for the video",
    )
    return any(marker in lowered for marker in markers)


def sanitize_header(name: str) -> str:
    return name.lstrip("\ufeff").strip()


def normalize_fieldnames(fieldnames: List[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for name in fieldnames:
        cleaned = sanitize_header(name)
        if cleaned and cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return normalized


def normalize_row_keys(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        for key in list(row.keys()):
            cleaned_key = sanitize_header(key)
            if cleaned_key == key:
                continue
            value = row.pop(key)
            if cleaned_key not in row or not row.get(cleaned_key):
                row[cleaned_key] = value


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else ""


def fetch_transcript(video_id: str, language: str) -> str:
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=[language])
    return " ".join([item.text for item in transcript])


def load_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_curriculum_id(row: Dict[str, str]) -> str:
    return row.get("curriculum_id") or row.get("curriculam_id") or ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch transcripts for new rows and build videos.jsonl."
    )
    parser.add_argument("--csv", required=True, help="Path to data.csv")
    parser.add_argument(
        "--transcripts-dir",
        default="data/transcripts",
        help="Directory to store transcript files",
    )
    parser.add_argument(
        "--videos-jsonl",
        default="data/videos/videos.jsonl",
        help="Output videos.jsonl path",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Default transcript language if column missing",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between transcript calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print summary",
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
        help="Disable progress bar output",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    csv_path = Path(args.csv)
    transcripts_dir = Path(args.transcripts_dir)
    videos_jsonl = Path(args.videos_jsonl)

    rows, fieldnames = load_csv(csv_path)
    fieldnames = normalize_fieldnames(fieldnames)
    normalize_row_keys(rows)
    LOGGER.info("Loaded %s rows from %s", len(rows), csv_path)
    if "transcript_path" not in fieldnames:
        fieldnames.append("transcript_path")
        LOGGER.info("Added missing 'transcript_path' column")

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Transcripts directory: %s", transcripts_dir)

    updated = 0
    fetched = 0
    reused_existing_files = 0
    skipped_existing_paths = 0
    skipped_missing_video_id = 0
    skipped_dry_run = 0
    fetch_errors = 0
    aborted_due_to_ip_block = False
    abort_reason = ""
    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(rows, desc="Syncing transcripts", unit="row")
    elif not args.no_progress and tqdm is None:
        LOGGER.warning(
            "tqdm is not installed; install it with 'pip install tqdm' to enable progress bar."
        )

    iterable = progress if progress is not None else rows
    for row in iterable:
        video_id = row.get("video_id", "").strip()
        if not video_id:
            video_id = extract_video_id(row.get("youtube_url", ""))
            row["video_id"] = video_id

        transcript_path = row.get("transcript_path", "").strip()
        if transcript_path:
            transcript_file = Path(transcript_path)
            if transcript_file.exists():
                skipped_existing_paths += 1
                if progress is not None:
                    progress.set_postfix(updated=updated, fetched=fetched, errors=fetch_errors)
                continue
        if not video_id:
            skipped_missing_video_id += 1
            LOGGER.debug("Skipping row because video_id could not be extracted")
            if progress is not None:
                progress.set_postfix(updated=updated, fetched=fetched, errors=fetch_errors)
            continue

        transcript_file = transcripts_dir / f"{video_id}.txt"
        if transcript_file.exists():
            row["transcript_path"] = str(transcript_file).replace("\\", "/")
            updated += 1
            reused_existing_files += 1
            LOGGER.debug("Reused existing transcript file for video_id=%s", video_id)
            if progress is not None:
                progress.set_postfix(updated=updated, fetched=fetched, errors=fetch_errors)
            continue

        if args.dry_run:
            skipped_dry_run += 1
            LOGGER.debug("Dry-run: skipping transcript fetch for video_id=%s", video_id)
            if progress is not None:
                progress.set_postfix(updated=updated, fetched=fetched, errors=fetch_errors)
            continue

        language = row.get("language", "").strip() or args.language
        try:
            transcript_text = fetch_transcript(video_id, language)
        except Exception as exc:
            fetch_errors += 1
            LOGGER.warning(
                "Failed to fetch transcript for video_id=%s (language=%s): %s",
                video_id,
                language,
                exc,
            )
            if is_ip_block_error(exc):
                aborted_due_to_ip_block = True
                abort_reason = (
                    "YouTube likely blocked this IP. Stop the run, wait before retrying, "
                    "increase --sleep, or use a different/proxied IP."
                )
                LOGGER.error(abort_reason)
                if progress is not None:
                    progress.set_postfix(
                        updated=updated, fetched=fetched, errors=fetch_errors
                    )
                break
            if progress is not None:
                progress.set_postfix(updated=updated, fetched=fetched, errors=fetch_errors)
            continue

        transcript_file.write_text(transcript_text, encoding="utf-8")
        row["transcript_path"] = str(transcript_file).replace("\\", "/")
        fetched += 1
        updated += 1
        LOGGER.info("Fetched transcript for video_id=%s", video_id)
        if progress is not None:
            progress.set_postfix(updated=updated, fetched=fetched, errors=fetch_errors)
        time.sleep(args.sleep)

    if progress is not None:
        progress.close()

    if not args.dry_run:
        write_csv(csv_path, rows, fieldnames)
        LOGGER.info("Updated CSV written to %s", csv_path)

        videos_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with videos_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                video_id = row.get("video_id", "").strip()
                if not video_id:
                    continue
                record = {
                    "video_id": video_id,
                    "domain": row.get("Domain", "") or row.get("domain", ""),
                    "module": row.get("Module", "") or row.get("module", ""),
                    "topic": row.get("Topic", "") or row.get("topic", ""),
                    "youtube_url": row.get("youtube_url", ""),
                    "language": row.get("language", "") or args.language,
                    "transcript_path": row.get("transcript_path", ""),
                    "curriculum_id": row_curriculum_id(row),
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        LOGGER.info("videos.jsonl written to %s", videos_jsonl)
    else:
        LOGGER.info("Dry-run enabled; skipping CSV and videos.jsonl writes")

    summary = (
        f"Rows updated: {updated} | transcripts fetched: {fetched} | "
        f"reused transcript files: {reused_existing_files} | "
        f"rows skipped(existing transcript_path): {skipped_existing_paths} | "
        f"rows skipped(no video_id): {skipped_missing_video_id} | "
        f"rows skipped(dry-run): {skipped_dry_run} | "
        f"fetch errors: {fetch_errors} | "
        f"aborted_due_to_ip_block: {aborted_due_to_ip_block} | "
        f"csv: {csv_path} | videos: {videos_jsonl}"
    )
    if abort_reason:
        summary += f" | reason: {abort_reason}"
    LOGGER.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
