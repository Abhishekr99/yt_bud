import argparse
import json
from pathlib import Path

from dataset.labeling import export_disagreements
from dataset.schema import ConceptCoverageLabel


def main() -> None:
    parser = argparse.ArgumentParser(description="Export disagreement labels to CSV.")
    parser.add_argument("--labels", required=True, help="Labels JSONL path")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    rows = []
    with open(args.labels, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(ConceptCoverageLabel.from_dict(json.loads(line)))

    export_disagreements(rows, args.out)


if __name__ == "__main__":
    main()
