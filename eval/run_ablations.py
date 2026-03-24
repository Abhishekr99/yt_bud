import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from eval.run_experiments import run_experiments

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation grid.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--out", default="results", help="Output root")
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
    args = parser.parse_args()
    configure_logging(args.log_level)
    use_progress = not args.no_progress
    if use_progress and tqdm is None:
        LOGGER.warning(
            "tqdm is not installed; install it with 'pip install tqdm' to enable progress bars."
        )
        use_progress = False

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = payload.get("base", {})
    grid = payload.get("grid", [])
    LOGGER.info(
        "Loaded ablation config: base_keys=%s grid_entries=%s",
        len(base),
        len(grid),
    )

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Ablation run id: %s | output dir: %s", run_id, out_dir)

    rows = []
    iterable = (
        tqdm(list(enumerate(grid)), desc="Ablation runs", unit="run")
        if use_progress
        else enumerate(grid)
    )
    for idx, entry in iterable:
        LOGGER.info("Running ablation_%s with overrides: %s", idx, entry)
        config = dict(base)
        config.update(entry)
        config["out"] = str(out_dir)
        config["run_id"] = f"ablation_{idx}"
        config["chunker_config"] = json.dumps(config.get("chunker_config", {}))
        config.setdefault("log_level", args.log_level)
        if args.no_progress:
            config["no_progress"] = True

        result = run_experiments(config)
        LOGGER.info(
            "Completed %s | models=%s",
            config["run_id"],
            len(result.get("metrics", [])),
        )
        for metric in result["metrics"]:
            row = {"run_id": config["run_id"], "chunker": config.get("chunker")}
            row.update(entry.get("ablation_flags", {}))
            row.update(metric)
            rows.append(row)
        if use_progress:
            iterable.set_postfix(rows=len(rows))

    csv_path = out_dir / "ablations.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write(",".join(keys) + "\n")
            for row in rows:
                handle.write(",".join(str(row.get(key, "")) for key in keys) + "\n")
        LOGGER.info("Ablation summary written to %s (rows=%s)", csv_path, len(rows))
    else:
        LOGGER.warning("No ablation rows were produced. Check config grid/models.")


if __name__ == "__main__":
    main()
