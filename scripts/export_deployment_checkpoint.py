#!/usr/bin/env python3
"""Export the final-test-blind development-selected deployment checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one full-model seed by its best S-split macro AUROC and export "
            "a checkpoint without cohort identifiers."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/final_evaluation/model/full/final"),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/GlycoContext_dev_selected_seed4.pt"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selector_summary(run: dict) -> dict:
    selectors = [fit for fit in run["fits"] if fit["stage"] == "selector"]
    if len(selectors) != 1:
        raise ValueError("Expected exactly one selector fit")
    selector = selectors[0]
    history = selector["history"]
    if not history or any("S_macro_AUROC" not in row for row in history):
        raise ValueError("Selector history lacks S_macro_AUROC")
    best = max(history, key=lambda row: (float(row["S_macro_AUROC"]), -int(row["epoch"])))
    if int(selector["selected_epoch"]) != int(best["epoch"]):
        raise ValueError("Stored selected epoch is not the best S-split epoch")
    return {
        "seed": int(run["seed"]),
        "selected_epoch": int(best["epoch"]),
        "S_macro_AUROC": float(best["S_macro_AUROC"]),
    }


def main() -> None:
    args = arguments()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be unique")
    candidates: list[dict] = []
    runs: dict[int, dict] = {}
    for seed in args.seeds:
        run_path = args.runs_root / f"seed{seed}" / "training.json"
        run = json.loads(run_path.read_text())
        if str(run.get("status", "")).upper() != "COMPLETE" or run.get("ablation") != "full":
            raise ValueError(f"Incomplete or non-full training record: {run_path}")
        if int(run["seed"]) != seed:
            raise ValueError(f"Seed mismatch: {run_path}")
        runs[seed] = run
        candidates.append(selector_summary(run))

    # Seed is used only as a deterministic tie-breaker. No final-test metric is read.
    selected = max(candidates, key=lambda row: (row["S_macro_AUROC"], -row["seed"]))
    seed = selected["seed"]
    run = runs[seed]
    source = args.runs_root / f"seed{seed}" / "model.pt"
    state = torch.load(source, map_location="cpu", weights_only=True)
    if int(state["seed"]) != seed or state.get("ablation") != "full":
        raise ValueError("Source checkpoint metadata mismatch")
    if int(state["epochs"]) != selected["selected_epoch"]:
        raise ValueError("Source checkpoint does not use the S-selected epoch count")
    if state["thresholds"] != run["thresholds"]:
        raise ValueError("Source checkpoint and training-record thresholds differ")

    exported = {
        "state_dict": state["state_dict"],
        "seed": seed,
        "ablation": "full",
        "epochs": int(state["epochs"]),
        "stage": "refit_E_plus_S",
        "cell_names": ["HeLa", "H9"],
        "thresholds": {str(key): float(value) for key, value in state["thresholds"].items()},
        "architecture": state["architecture"],
        "deployment_selection": {
            "criterion": "maximum selector S equal-cell macro AUROC across seeds; lower seed breaks exact ties",
            "selected_seed": seed,
            "selector_S_macro_AUROC": selected["S_macro_AUROC"],
            "selector_selected_epoch": selected["selected_epoch"],
            "candidate_summaries": sorted(candidates, key=lambda row: row["seed"]),
            "refit": "fresh initialization trained on E+S for the selected epoch count",
            "thresholds": "per-cell MCC thresholds selected on C",
            "final_test_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.save(exported, args.output)
    print(json.dumps(exported["deployment_selection"], indent=2))
    print(f"wrote={args.output}")
    print(f"sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
