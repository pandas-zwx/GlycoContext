# Competing baselines: definitions and execution

This repository provides two baselines: a **GlyinsRNA-style random forest reconstructed from the published description** and **RNA-FM embeddings plus metadata with LightGBM**. This document defines the methods, commands, and comparability boundaries; it does not provide scores, predictions, or trained classifiers. Implementations are in `src/glycocontext/baselines.py`, and fixed specifications are in `configs/baselines.json`.

## Shared experimental protocol

- Use the current set of 11,479 HeLa/H9 candidates and `splits/*.json`; do not randomly resplit windows. See `splits/README.md` for grouping rules.
- `E` is used for initial fitting, `S` for model selection, `E+S` for final refitting, `C` only for per-cell MCC threshold selection, and `test` only for evaluation.
- RF hyperparameters and ensemble weights are selected using five-fold group-aware out-of-fold predictions within E. The RF does not use S for epoch selection; S is subsequently included in the final fit.
- LightGBM is fitted on E, uses S to select its configuration and number of boosting rounds, and is then refitted on E+S for the selected number of rounds.
- Seeds 0, 1, 2, 3, and 4 are refitted independently. Inner RF selection uses fixed seed 0, and the five final training seeds share the same selection rule; favorable seeds are never selected post hoc.
- Thresholds on C are searched from 0.01 to 0.99 in steps of 0.01 to maximize MCC. Ties are resolved first by proximity to 0.5 and then by choosing the smaller threshold.
- AUROC, AP, MCC, Brier score, and log loss are reported per cell, together with an equal-weight HeLa/H9 macro average. AP means average precision and is not described as trapezoidal PR area.
- Every model uses the same prediction-CSV interface and `scripts/evaluate.py`. Thresholds are not refitted during inference.

## 1. GlyinsRNA-style random forest

Reference: *GlyinsRNA: a webserver for predicting glycosylation sites on small RNAs*, <https://doi.org/10.1080/15476286.2021.1982574>.

This baseline is **reconstructed from the feature descriptions in the paper**. It is not a component-by-component reproduction of the authors' official source code, original split, exact hyperparameters, or pretrained model. The current data come from the curated GlycoRNAdb cohort. Although the two studies share the GSE136967 source, their samples, filters, and grouping protocols differ. Do not subtract the published score from the present score and interpret the difference directly as an algorithmic improvement.

Three random forests are trained independently for each cell, and their positive-class probabilities are combined with nonnegative weights:

| Encoding | Input and dimension | Implementation |
|---|---|---|
| PSDSP | 50 adjacent dinucleotides in a 51-nt window; 50 dimensions | Position-specific positive frequency minus negative frequency, computed only from the current fitting set |
| One-hot | 51 nt; 204 dimensions | Four A/C/G/U channels; out-of-range positions and N are all zero; no structure channels |
| KNF | Central 17 nt; 336 dimensions | Frequencies of 16, 64, and 256 possible 2/3/4-mers, each normalized by the number of valid k-mers |

Padding at RNA termini never extends beyond the database sequence, and PSDSP skips dinucleotides containing padding or unknown bases. PSDSP is supervised: for every RF fitting sample, frequencies are computed while **excluding its entire similarity group**. Validation and prediction samples use frequencies derived only from the fitting set. Computing PSDSP from full-cohort labels before splitting is prohibited.

Each RF uses 200 trees, `max_features=sqrt`, `class_weight=None`, and two threads; candidate minimum leaf sizes are 1 and 5. Within E, each cell is divided into five folds using a fixed SHA256 ordering of groups to produce out-of-fold probabilities for the three encodings. Ensemble weights are selected to maximize out-of-fold AUROC over a nonnegative sum-to-one grid with step size 0.1, plus the exact equal-weight vector. Weight ties favor proximity to equal weighting and then the lexicographically larger weight vector; leaf-size ties favor the smaller value. C and test labels are never used for supervised encoding, RF fitting, or ensemble-weight selection.

```bash
python scripts/train_baseline.py --method rf --split splits/final.json --seed 0 \
  --output runs/baselines/rf/final/seed0
python scripts/predict_baseline.py --checkpoint runs/baselines/rf/final/seed0/model.joblib \
  --split splits/final.json --output runs/baselines/rf/final/seed0/predictions.csv
python scripts/evaluate.py --predictions runs/baselines/rf/final/seed0/predictions.csv \
  --output runs/baselines/rf/final/seed0/metrics.json
```

The random forest requires no GPU, RNAfold installation, or RNA-FM feature cache.

## 2. RNA-FM embeddings plus metadata with LightGBM

This model is an additional strong non-neural control. It is **neither the RF from the original GlyinsRNA paper nor a RiNALMo fusion model**. RNA-FM is frozen, and the input has 2,564 dimensions:

- Target-nucleotide vectors from the full database RNA and the 51-nt and 101-nt contexts: 3 x 640 dimensions.
- The mean of all nucleotide vectors from the full database RNA: 640 dimensions. BOS/EOS are excluded; this is not an average of target-site vectors.
- `log1p(RNA length)`, relative position `index0/max(length-1,1)`, and two HeLa/H9 one-hot features: 4 dimensions.

Because this baseline explicitly receives a whole-RNA mean, RNA length, and relative position that are not explicit inputs to the main model, it is a **strong competing baseline rather than a perfectly input-matched structural ablation**. It does not use RNAfold, GNN predictions, experimental signals, sequencing read counts, or RiNALMo.

Six configurations are considered: `raw/token_LN x num_leaves=7/15/31`. `token_LN` standardizes each 640-dimensional vector independently by its feature mean and variance with epsilon 1e-5; it does not fit a dataset-wide scaler. LightGBM 4.6.0 uses a binary objective, learning rate 0.03, minimum 50 observations per leaf, L2 regularization 10, `max_bin=63`, feature fraction 0.8, and at most 800 boosting rounds. Early stopping uses the equal-weight cell AUROC on S with patience 40. Configuration ties select the first item in the fixed traversal order. At every fit, positive observations are reweighted by the cell-specific negative-to-positive ratio computed from the current fitting set, never from C or test. The selected number of rounds is refitted on E+S, and C is used only for thresholds. Training uses two CPU threads and does not train RNA-FM.

### Environment

The RF can use scikit-learn from the existing `diffusion_gpu` environment. LightGBM additionally requires version 4.6.0:

```bash
python -m pip install -e '.[baselines]'
```

Using this optional dependency in a separate environment is recommended to avoid changing the main-model training environment. Native LightGBM libraries can differ slightly across platforms, so bitwise cross-platform identity is not guaranteed.

### Precomputed inputs

The three target-context vectors are read from `features/rnafm_views.npy` and shared with the main model. Whole-RNA means are read from `features_baselines/rnafm_native_mean.npy` and aligned by RNA hash through `sequence_keys.json`. Both directories have SHA256 manifests and contain **sequence-derived features, not predictions or trained classifiers**; both are excluded from Git by default. The IMI handoff directory contains these caches. Rebuild whole-RNA means with:

```bash
python scripts/prepare_baseline_features.py --data data --output features_baselines_rebuilt \
  --rnafm-repo third_party/RNA-FM --checkpoint /path/to/RNA-FM_pretrained.pth
```

The preparation procedure fixes the official RNA-FM weights, layer 12, FP32, and per-RNA batch size 1, using the same weight SHA256 and 1,022-nt length limit as the main model. Nucleotide vectors are accumulated in float64 before the mean is stored as float32. See "Feature reconstruction" in the root README for the complete feature-preparation procedure. New data require both the three target-context representations and the whole-RNA means, passed through their corresponding directories:

```bash
python scripts/train_baseline.py --method rnafm_lgbm --split splits/final.json --seed 0 \
  --output runs/baselines/rnafm_lgbm/final/seed0
python scripts/predict_baseline.py --checkpoint runs/baselines/rnafm_lgbm/final/seed0/model.joblib \
  --split splits/final.json --output runs/baselines/rnafm_lgbm/final/seed0/predictions.csv
python scripts/evaluate.py --predictions runs/baselines/rnafm_lgbm/final/seed0/predictions.csv \
  --output runs/baselines/rnafm_lgbm/final/seed0/metrics.json
```

For inference on new data, omit `--split` and pass `--data /path/new_data --features /path/new_features --mean-features /path/new_means`. Unknown cells, missing features, misaligned IDs, and incorrect weight or cache digests raise explicit errors; missing labels are not converted into negatives. Load `joblib` files only when they were created locally or obtained from a trusted source, because untrusted pickle files can execute code.

## Finite run queue and validation

```bash
python -m unittest discover -s tests -v
python scripts/run_baselines.py --methods rf rnafm_lgbm --seeds 0 1 2 3 4 \
  --partitions dev0 dev1 dev2 final --output runs/baselines
```

The queue trains, saves, predicts, and invokes the shared evaluation entry point sequentially. It stops rather than overwriting existing outputs. Use `train_baseline.py` directly when training without evaluation. Both commands should run on allocated CPU resources, not consume a login node for full experiments. Use the neural training entry points in the root README for the strongest main model and its ablations; RF or tree models cannot substitute for retrained neural ablations.
