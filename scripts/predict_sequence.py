#!/usr/bin/env python3
"""Scan every nucleotide in one RNA with the released GlycoContext checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glycocontext.inference import (  # noqa: E402
    RNAFMEncoder,
    build_scan_features,
    fold_for_inference,
    load_deployment_checkpoint,
    normalize_sequence,
    read_single_fasta,
    resolve_device,
    score_tracks,
)
from glycocontext.io import read_json, sequence_key, sha256, write_json  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a GlycoContext score at every nucleotide of one putative glycoRNA."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help="RNA sequence; DNA T is normalized to U")
    source.add_argument("--fasta", type=Path, help="Single-record FASTA file")
    parser.add_argument("--rna-id", help="Output identifier (defaults to FASTA ID or query_rna)")
    parser.add_argument("--cell", choices=["HeLa", "H9"], help="Omit to emit both HeLa and H9 tracks")
    parser.add_argument("--candidate-bases", default="ACGU", help="Bases to scan; default: ACGU")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("weights/GlycoContext_dev_selected_seed4.pt"),
    )
    parser.add_argument(
        "--rnafm-checkpoint",
        type=Path,
        default=Path(os.environ["RNAFM_CHECKPOINT"]) if os.getenv("RNAFM_CHECKPOINT") else None,
        help="Official RNA-FM checkpoint; alternatively set RNAFM_CHECKPOINT",
    )
    parser.add_argument("--rnafm-repo", type=Path, default=Path("third_party/RNA-FM"))
    parser.add_argument(
        "--preparation-config", type=Path, default=Path("configs/preparation.json")
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rnafm-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.rnafm_checkpoint is None:
        raise ValueError("Provide --rnafm-checkpoint or set RNAFM_CHECKPOINT")
    if args.fasta:
        fasta_id, sequence = read_single_fasta(args.fasta)
        rna_id = args.rna_id or fasta_id
    else:
        sequence = normalize_sequence(args.sequence)
        rna_id = args.rna_id or "query_rna"
    if not rna_id or any(value in rna_id for value in "\r\n\t,"):
        raise ValueError("RNA identifier must be a nonempty single CSV-safe token")

    candidate_bases = "".join(dict.fromkeys(args.candidate_bases.upper().replace("T", "U")))
    if not candidate_bases or set(candidate_bases) - set("ACGU"):
        raise ValueError("--candidate-bases must be a nonempty subset of A/C/G/U/T")
    centers = np.array(
        [index for index, base in enumerate(sequence) if base in candidate_bases], dtype=np.int64
    )
    if not len(centers):
        raise ValueError("No candidate nucleotide remains after --candidate-bases filtering")

    output = args.output.resolve()
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing prediction or metadata file")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    preparation = read_json(args.preparation_config)
    maximum = int(preparation["max_RNA_nt"])
    if len(sequence) > maximum:
        raise ValueError(
            f"RNA has {len(sequence)} nt; the validated RNA-FM limit is {maximum} nt"
        )
    structure = fold_for_inference(sequence)
    encoder = RNAFMEncoder(
        args.rnafm_repo, args.rnafm_checkpoint, preparation, device
    )
    views = encoder.site_views(sequence, centers, args.rnafm_batch_size)
    features = build_scan_features(sequence, centers, structure, views)
    model, checkpoint = load_deployment_checkpoint(args.checkpoint, device)
    cells = [args.cell] if args.cell else ["HeLa", "H9"]
    rows = score_tracks(model, checkpoint, features, cells, device, args.batch_size)
    for row in rows:
        row["rna_id"] = rna_id
        row["nucleotide"] = sequence[row["position1"] - 1]

    fields = [
        "rna_id", "cell", "position1", "nucleotide", "probability", "threshold",
        "prediction", "logit", "cnn_logit", "gnn_logit", "rnafm_logit",
    ]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)

    positives = {
        cell: sum(row["prediction"] for row in rows if row["cell"] == cell)
        for cell in cells
    }
    metadata = {
        "schema": "GlycoContext-sequence-scan-v1",
        "rna_id": rna_id,
        "sequence_sha256": sequence_key(sequence),
        "length_nt": len(sequence),
        "candidate_bases": candidate_bases,
        "candidate_positions": len(centers),
        "cell_tracks": cells,
        "positive_calls": positives,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_seed": int(checkpoint["seed"]),
        "checkpoint_epochs": int(checkpoint["epochs"]),
        "checkpoint_selection": checkpoint.get("deployment_selection"),
        "thresholds": {cell: float(checkpoint["thresholds"][str(index)]) for index, cell in enumerate(["HeLa", "H9"])},
        "rnafm_checkpoint_sha256": encoder.checkpoint_sha256,
        "rnafm_layer": encoder.layer,
        "rnafold_version": structure["RNA_version"],
        "rnafold_temperature_C": structure["temperature_C"],
        "mfe_kcal_per_mol": structure["mfe_kcal_per_mol"],
        "ensemble_free_energy_kcal_per_mol": structure["ensemble_free_energy"],
        "device": device,
        "interpretation": "Model scores for the operational enriched-only versus input-only label; not chemical confirmation of glycosylation.",
    }
    write_json(metadata_path, metadata)
    print(f"Wrote {len(rows)} site-cell rows to {output}")
    for cell in cells:
        print(f"{cell}: {positives[cell]} / {len(centers)} positions at the checkpoint's C-selected threshold")


if __name__ == "__main__":
    main()
