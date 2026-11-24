"""Summarize the frozen-order incremental path on the held-out final test."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


PATH = (
    ("inc_cnn_seq", "Sequence CNN"),
    ("inc_cnn_structure", "+ RNAfold channels"),
    ("inc_rnafm", "+ RNA-FM"),
    ("inc_gnn_backbone", "+ backbone GNN"),
    ("no_cell", "+ pair messages"),
    ("full", "+ cell context"),
)
METRICS = ("AUROC", "AP", "MCC", "local_AUROC")
CELLS = ("HeLa", "H9", "macro")


def metric_path(root, arm, seed):
    if arm == "no_cell":
        return root / "ablations/no_cell/final" / f"seed{seed}" / "metrics.json"
    if arm == "full":
        return root / "model/full/final" / f"seed{seed}" / "metrics.json"
    return root / "incremental" / arm / "final" / f"seed{seed}" / "metrics.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/final_evaluation")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/final_evaluation/incremental_analysis"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    values = {}
    for arm, _ in PATH:
        for seed in range(5):
            source_path = metric_path(args.root, arm, seed)
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            data = json.loads(source_path.read_text())
            for cell in CELLS:
                source = data["macro"] if cell == "macro" else data["cells"][cell]["metrics"]
                for metric in METRICS:
                    values[arm, seed, cell, metric] = float(source[metric])

    summary = []
    paired = []
    for arm, label in PATH:
        for cell in CELLS:
            for metric in METRICS:
                x = np.array([values[arm, seed, cell, metric] for seed in range(5)])
                summary.append(
                    dict(
                        arm=arm,
                        label=label,
                        cell=cell,
                        metric=metric,
                        n_seeds=5,
                        mean=float(x.mean()),
                        population_sd=float(x.std(ddof=0)),
                    )
                )
    for (previous, _), (arm, label) in zip(PATH, PATH[1:]):
        for cell in CELLS:
            for metric in METRICS:
                delta = np.array(
                    [
                        values[arm, seed, cell, metric]
                        - values[previous, seed, cell, metric]
                        for seed in range(5)
                    ]
                )
                paired.append(
                    dict(
                        previous=previous,
                        arm=arm,
                        addition=label,
                        cell=cell,
                        metric=metric,
                        n_pairs=5,
                        mean_delta=float(delta.mean()),
                        population_sd_delta=float(delta.std(ddof=0)),
                    )
                )

    for filename, rows in (("summary.csv", summary), ("paired_steps.csv", paired)):
        with (args.output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    protocol = {
        "analysis": "fixed-order incremental component analysis",
        "partition": "final",
        "held_out_role": "test",
        "seeds": [0, 1, 2, 3, 4],
        "path": [{"arm": arm, "label": label} for arm, label in PATH],
        "selection": "epoch selected on S; threshold selected on C; test used only for reporting",
        "interpretation": "Adjacent differences are conditional on the frozen order, not order-independent or Shapley contributions.",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
