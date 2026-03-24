import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


LOGGER = logging.getLogger(__name__)
INVALID_FILENAME_CHARS = set('<>:"/\\|?*')


@dataclass(frozen=True)
class CurriculumSeed:
    curriculum_id: str
    domain: str
    module: str
    topic: str


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_curriculum_id(curriculum_id: str) -> List[str]:
    return curriculum_id.split("_", 2) if curriculum_id else []


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_curriculum_id_for_filename(curriculum_id: str) -> str | None:
    if not curriculum_id:
        return "empty curriculum_id"
    if any(ord(ch) < 32 for ch in curriculum_id):
        return "contains control characters"
    bad_chars = sorted({ch for ch in curriculum_id if ch in INVALID_FILENAME_CHARS})
    if bad_chars:
        joined = " ".join(bad_chars)
        return f"contains invalid filename characters: {joined}"
    if curriculum_id.endswith(" ") or curriculum_id.endswith("."):
        return "cannot end with space or dot"
    return None


def load_builder() -> Callable[..., dict]:
    try:
        from dataset.curriculum_builder import build_curriculum
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependencies for curriculum generation. Install requirements and retry."
        ) from exc
    return build_curriculum


def row_to_seed(row: Dict[str, str]) -> CurriculumSeed | None:
    curriculum_id = str(row.get("curriculum_id", "")).strip()
    if not curriculum_id:
        return None

    domain = str(row.get("domain", "")).strip()
    module = str(row.get("module", "")).strip()
    topic = str(row.get("topic", "")).strip()

    # Fallback: infer missing fields from curriculum_id = domain_module_topic
    if not (domain and module and topic):
        parts = parse_curriculum_id(curriculum_id)
        if len(parts) == 3:
            domain = domain or parts[0].strip()
            module = module or parts[1].strip()
            topic = topic or parts[2].strip()

    if not (domain and module and topic):
        return None

    return CurriculumSeed(
        curriculum_id=curriculum_id,
        domain=domain,
        module=module,
        topic=topic,
    )


def collect_unique_seeds(
    rows: Iterable[Dict[str, str]], requested_ids: Set[str]
) -> List[CurriculumSeed]:
    by_id: Dict[str, CurriculumSeed] = {}
    skipped_invalid = 0
    conflicts = 0

    for row in rows:
        seed = row_to_seed(row)
        if seed is None:
            skipped_invalid += 1
            continue
        if requested_ids and seed.curriculum_id not in requested_ids:
            continue
        existing = by_id.get(seed.curriculum_id)
        if existing is None:
            by_id[seed.curriculum_id] = seed
            continue
        if (
            existing.domain != seed.domain
            or existing.module != seed.module
            or existing.topic != seed.topic
        ):
            conflicts += 1
            LOGGER.warning(
                "Conflicting row for curriculum_id=%s. Keeping first: "
                "(%s, %s, %s), ignoring: (%s, %s, %s)",
                seed.curriculum_id,
                existing.domain,
                existing.module,
                existing.topic,
                seed.domain,
                seed.module,
                seed.topic,
            )

    if skipped_invalid:
        LOGGER.warning("Skipped %s rows with missing/invalid curriculum metadata", skipped_invalid)
    if conflicts:
        LOGGER.warning("Detected %s curriculum_id conflicts", conflicts)

    return sorted(by_id.values(), key=lambda item: item.curriculum_id.lower())


def output_path_for_seed(out_dir: Path, seed: CurriculumSeed) -> Path:
    return out_dir / f"{seed.curriculum_id}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate curriculum JSON files from videos.jsonl rows."
    )
    parser.add_argument(
        "--videos",
        default="data/videos/videos.jsonl",
        help="Path to videos.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="data/curricula",
        help="Directory where curriculum JSON files will be written",
    )
    parser.add_argument(
        "--curriculum-id",
        action="append",
        default=[],
        help="Generate only this curriculum_id (repeatable)",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=40,
        help="Max concepts per generated curriculum",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing curriculum JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
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

    videos_path = Path(args.videos)
    out_dir = Path(args.out_dir)
    requested_ids = {item.strip() for item in args.curriculum_id if item.strip()}

    rows = read_jsonl(videos_path)
    seeds = collect_unique_seeds(rows, requested_ids=requested_ids)
    LOGGER.info("Loaded %s rows and identified %s unique curriculum seeds", len(rows), len(seeds))

    filename_owner: Dict[str, str] = {}
    filename_issues = 0
    for seed in seeds:
        file_name = f"{seed.curriculum_id}.json"
        validation_error = validate_curriculum_id_for_filename(seed.curriculum_id)
        if validation_error:
            filename_issues += 1
            LOGGER.error(
                "Invalid curriculum_id for filename '%s': %s",
                seed.curriculum_id,
                validation_error,
            )
            continue
        key = file_name.lower()
        existing_id = filename_owner.get(key)
        if existing_id and existing_id != seed.curriculum_id:
            filename_issues += 1
            LOGGER.error(
                "Filename collision (case-insensitive): '%s' and '%s' both map to '%s'",
                existing_id,
                seed.curriculum_id,
                file_name,
            )
        else:
            filename_owner[key] = seed.curriculum_id
    if filename_issues:
        LOGGER.error(
            "Found %s filename issue(s). Fix curriculum_id values before generating files.",
            filename_issues,
        )
        raise SystemExit(1)

    if requested_ids:
        found_ids = {seed.curriculum_id for seed in seeds}
        missing = sorted(requested_ids - found_ids)
        for curriculum_id in missing:
            LOGGER.warning("Requested curriculum_id not found: %s", curriculum_id)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    builder = None
    if not args.dry_run:
        try:
            builder = load_builder()
        except RuntimeError as exc:
            LOGGER.error("%s", exc)
            raise SystemExit(1) from exc

    created = 0
    skipped_existing = 0
    failed = 0

    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(seeds, desc="Generating curricula", unit="curriculum")
    elif not args.no_progress and tqdm is None:
        LOGGER.warning(
            "tqdm is not installed; install it with 'pip install tqdm' to enable progress bar."
        )

    iterable = progress if progress is not None else seeds
    for seed in iterable:
        out_path = output_path_for_seed(out_dir, seed)
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            LOGGER.debug("Skipping existing: %s", out_path)
            if progress is not None:
                progress.set_postfix(created=created, skipped=skipped_existing, failed=failed)
            continue

        LOGGER.info(
            "Generating curriculum_id=%s (%s | %s | %s)",
            seed.curriculum_id,
            seed.domain,
            seed.module,
            seed.topic,
        )

        if args.dry_run:
            created += 1
            if progress is not None:
                progress.set_postfix(created=created, skipped=skipped_existing, failed=failed)
            continue

        try:
            curriculum = builder(
                domain=seed.domain,
                module=seed.module,
                topic=seed.topic,
                curriculum_id=seed.curriculum_id,
                max_concepts=args.max_concepts,
            )
            out_path.write_text(json.dumps(curriculum, indent=2, ensure_ascii=False), encoding="utf-8")
            created += 1
        except Exception as exc:
            failed += 1
            LOGGER.error("Failed to generate %s: %s", seed.curriculum_id, exc)

        if progress is not None:
            progress.set_postfix(created=created, skipped=skipped_existing, failed=failed)

    if progress is not None:
        progress.close()

    summary = (
        f"Curricula processed: {len(seeds)} | created: {created} | "
        f"skipped(existing): {skipped_existing} | failed: {failed} | out_dir: {out_dir}"
    )
    LOGGER.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
