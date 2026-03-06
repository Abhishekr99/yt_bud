import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from eval.run_experiments import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation grid.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--out", default="results", help="Output root")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = payload.get("base", {})
    grid = payload.get("grid", [])

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, entry in enumerate(grid):
        config = dict(base)
        config.update(entry)
        config["out"] = str(out_dir)
        config["run_id"] = f"ablation_{idx}"
        config["chunker_config"] = json.dumps(config.get("chunker_config", {}))
        result = run_experiments(config)
        for metric in result["metrics"]:
            row = {"run_id": config["run_id"], "chunker": config.get("chunker")}
            row.update(entry.get("ablation_flags", {}))
            row.update(metric)
            rows.append(row)

    csv_path = out_dir / "ablations.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write(",".join(keys) + "\n")
            for row in rows:
                handle.write(",".join(str(row.get(key, "")) for key in keys) + "\n")


if __name__ == "__main__":
    main()
