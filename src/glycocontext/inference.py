"""Sequence-wide inference with frozen RNAfold and RNA-FM representations."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from .features import cnn_window, fold, graph_arrays
from .io import sequence_key, sha256
from .model import GlycoContext


CELL_TO_ID = {"HeLa": 0, "H9": 1}
ID_TO_CELL = {value: key for key, value in CELL_TO_ID.items()}


def normalize_sequence(sequence: str) -> str:
    """Normalize an RNA string while rejecting ambiguous novel input."""
    value = "".join(sequence.split()).upper().replace("T", "U")
    if not value:
        raise ValueError("RNA sequence is empty")
    invalid = sorted(set(value) - set("ACGU"))
    if invalid:
        raise ValueError("Novel inference accepts only A/C/G/U/T; invalid symbols: " + "".join(invalid))
    return value


def read_single_fasta(path: Path) -> tuple[str, str]:
    """Read exactly one FASTA record and return its identifier and sequence."""
    identifier = None
    pieces: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                raise ValueError("Inference FASTA must contain exactly one RNA record")
            identifier = line[1:].split()[0]
            if not identifier:
                raise ValueError("FASTA identifier is empty")
        else:
            if identifier is None:
                raise ValueError("FASTA sequence encountered before its header")
            pieces.append(line)
    if identifier is None:
        raise ValueError("No FASTA record found")
    return identifier, normalize_sequence("".join(pieces))


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


class RNAFMEncoder:
    """Exact layer-12 RNA-FM encoder used by the training feature pipeline."""

    def __init__(self, repository: Path, checkpoint: Path, config: dict, device: str):
        repository, checkpoint = Path(repository), Path(checkpoint)
        if not repository.exists():
            raise FileNotFoundError(f"RNA-FM repository not found: {repository}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"RNA-FM checkpoint not found: {checkpoint}")
        observed = sha256(checkpoint)
        if observed != config["checkpoint_sha256"]:
            raise ValueError(f"RNA-FM checkpoint SHA256 mismatch: {observed}")
        sys.path.insert(0, str(repository.resolve()))
        import fm

        with torch.serialization.safe_globals([Namespace]):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model, alphabet = fm.pretrained.load_model_and_alphabet_core(
            "rna-fm", payload, theme="rna"
        )
        del payload
        self.model = model.float().requires_grad_(False).eval().to(device)
        self.converter = alphabet.get_batch_converter()
        self.alphabet = alphabet
        self.offset = int(alphabet.prepend_bos)
        self.device = device
        self.dimension = int(config["dimension"])
        self.layer = int(config["layer"])
        self.max_nt = int(config["max_RNA_nt"])
        self.checkpoint_sha256 = observed

    @torch.inference_mode()
    def encode(self, sequences: list[str]) -> list[np.ndarray]:
        if not sequences:
            return []
        if max(map(len, sequences)) > self.max_nt:
            raise ValueError(f"RNA-FM input exceeds {self.max_nt} nt")
        _, strings, tokens = self.converter([(str(i), value) for i, value in enumerate(sequences)])
        if strings != sequences:
            raise ValueError("RNA-FM tokenizer changed an input sequence")
        for row, value in enumerate(sequences):
            expected = [self.alphabet.get_idx(base) for base in value]
            if tokens[row, self.offset:self.offset + len(value)].tolist() != expected:
                raise ValueError("RNA-FM token/base alignment failed")
        output = self.model(
            tokens.to(self.device), repr_layers=[self.layer], return_contacts=False
        )["representations"][self.layer].float()
        result = [
            output[row, self.offset:self.offset + len(value)].cpu().numpy().copy()
            for row, value in enumerate(sequences)
        ]
        if any(
            array.shape != (len(value), self.dimension) or not np.isfinite(array).all()
            for value, array in zip(sequences, result)
        ):
            raise ValueError("Invalid RNA-FM representation")
        return result

    def site_views(self, sequence: str, centers: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Return native/51-nt/101-nt candidate-token representations."""
        if len(sequence) > self.max_nt:
            raise ValueError(
                f"RNA has {len(sequence)} nt; the validated RNA-FM limit is {self.max_nt} nt"
            )
        if batch_size < 1:
            raise ValueError("RNA-FM batch size must be positive")
        centers = np.asarray(centers, dtype=np.int64)
        if centers.ndim != 1 or np.any(centers < 0) or np.any(centers >= len(sequence)):
            raise ValueError("Candidate centers are outside the RNA")

        native = self.encode([sequence])[0]
        views = np.full((len(centers), 3, self.dimension), np.nan, dtype=np.float32)
        views[:, 0] = native[centers]
        fragments: dict[str, str] = {}
        bindings: dict[str, list[tuple[int, int, int]]] = {}
        for row, center in enumerate(centers.tolist()):
            for column, width in ((1, 51), (2, 101)):
                left = max(0, center - width // 2)
                right = min(len(sequence), center + width // 2 + 1)
                if left == 0 and right == len(sequence):
                    views[row, column] = native[center]
                else:
                    fragment = sequence[left:right]
                    key = sequence_key(fragment)
                    fragments[key] = fragment
                    bindings.setdefault(key, []).append((row, column, center - left))

        order = sorted(fragments, key=lambda key: (-len(fragments[key]), key))
        for start in range(0, len(order), batch_size):
            keys = order[start:start + batch_size]
            batch = [fragments[key] for key in keys]
            for key, representation in zip(keys, self.encode(batch)):
                for row, column, offset in bindings[key]:
                    views[row, column] = representation[offset]
        if not np.isfinite(views).all():
            raise ValueError("Incomplete RNA-FM site views")
        return views


def build_scan_features(
    sequence: str,
    centers: np.ndarray,
    structure_record: dict,
    views: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the exact CNN and graph tensors used by the fixed model."""
    centers = np.asarray(centers, dtype=np.int64)
    if structure_record.get("RNA_version") != "2.7.2":
        raise ValueError("Sequence-wide inference requires ViennaRNA 2.7.2")
    values = graph_arrays(
        sequence,
        structure_record["structure"],
        structure_record["probabilities"],
    )
    nodes, neighbours, masks, edges, confidence, partner = values
    cnn = np.stack([cnn_window(sequence, int(center), partner) for center in centers])
    views = np.asarray(views, dtype=np.float32)
    if cnn.shape != (len(centers), 8, 51) or views.shape != (len(centers), 3, 640):
        raise ValueError("Sequence-wide feature shape mismatch")
    return {
        "cnn": cnn,
        "views": views,
        "nodes": nodes,
        "neighbours": neighbours,
        "masks": masks,
        "edges": edges,
        "confidence": confidence,
        "centers": centers,
    }


def load_deployment_checkpoint(path: Path, device: str) -> tuple[GlycoContext, dict]:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"state_dict", "thresholds", "seed", "epochs", "cell_names", "architecture"}
    if not required <= set(state):
        raise ValueError("Deployment checkpoint is missing required fields")
    if state.get("ablation", "full") != "full":
        raise ValueError("Sequence-wide inference requires a full-model checkpoint")
    if state["cell_names"] != ["HeLa", "H9"] or set(state["thresholds"]) != {"0", "1"}:
        raise ValueError("Checkpoint cell/threshold metadata mismatch")
    model = GlycoContext("full").to(device).eval()
    model.load_state_dict(state["state_dict"], strict=True)
    return model, state


@torch.inference_mode()
def score_tracks(
    model: GlycoContext,
    checkpoint: dict,
    features: dict[str, np.ndarray],
    cells: list[str],
    device: str,
    batch_size: int = 64,
) -> list[dict]:
    """Score every supplied center for one or both supported cell tracks."""
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    unknown = sorted(set(cells) - set(CELL_TO_ID))
    if unknown:
        raise ValueError("Unsupported cell tracks: " + ", ".join(unknown))

    nodes = torch.from_numpy(features["nodes"]).to(device)
    neighbours = torch.from_numpy(features["neighbours"]).long().to(device)
    masks = torch.from_numpy(features["masks"]).to(device)
    edges = torch.from_numpy(features["edges"]).to(device)
    confidence = torch.from_numpy(features["confidence"]).to(device)
    centers = features["centers"]
    rows: list[dict] = []

    for cell_name in cells:
        cell_id = CELL_TO_ID[cell_name]
        threshold = float(checkpoint["thresholds"][str(cell_id)])
        for start in range(0, len(centers), batch_size):
            stop = min(start + batch_size, len(centers))
            size = stop - start
            x = torch.from_numpy(features["cnn"][start:stop]).to(device)
            views = torch.from_numpy(features["views"][start:stop]).to(device)
            cell = torch.full((size,), cell_id, dtype=torch.long, device=device)
            graph = (
                nodes.unsqueeze(0).expand(size, -1, -1),
                neighbours.unsqueeze(0).expand(size, -1, -1),
                masks.unsqueeze(0).expand(size, -1, -1),
                torch.from_numpy(centers[start:stop]).long().to(device),
                edges.unsqueeze(0).expand(size, -1, -1, -1),
                views,
            )
            output = model(
                x,
                cell,
                graph,
                confidence.unsqueeze(0).expand(size, -1),
                return_components=True,
            )
            residual = output["logits"] - output["cnn"] - output["gnn"] - output["rnafm"]
            if float(residual.abs().max().cpu()) > 1e-5:
                raise RuntimeError("Branch logits do not reconstruct the full-model logit")
            probability = torch.sigmoid(output["logits"])
            for offset in range(size):
                rows.append({
                    "cell": cell_name,
                    "position1": int(centers[start + offset]) + 1,
                    "probability": float(probability[offset].cpu()),
                    "threshold": threshold,
                    "prediction": int(float(probability[offset].cpu()) >= threshold),
                    "logit": float(output["logits"][offset].cpu()),
                    "cnn_logit": float(output["cnn"][offset].cpu()),
                    "gnn_logit": float(output["gnn"][offset].cpu()),
                    "rnafm_logit": float(output["rnafm"][offset].cpu()),
                })
    return rows


def fold_for_inference(sequence: str) -> dict:
    """Fold one complete RNA once; all candidate sites reuse this whole-RNA fold."""
    return fold(sequence)
