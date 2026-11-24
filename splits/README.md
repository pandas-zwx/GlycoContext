# Fixed splits and training protocol

## Group before splitting

Full RNA sequences are converted to uppercase with T replaced by U, and exact sequence duplicates are merged by SHA256. Union-find connected components are then constructed from the following relationships:

1. Normalized global edit similarity of at least 0.8: `1-Levenshtein(A,B)/max(len(A),len(B))`. Distances are computed using edlib in global NW mode and include substitutions, insertions, and deletions. This is not the CD-HIT definition of alignment identity.
2. A shared valid, nonempty, correctly formatted RNA source ID or Ensembl gene ID. Missing-value placeholders such as `.` must not merge unrelated RNAs.
3. Previously frozen protected components remain intact and are not split when candidates are expanded.
4. Reference-validated exon intervals overlap on the same chromosome and strand, while established conservative groups are retained.
5. An identical 51-nt candidate window and `valid_mask` is present, even if full-RNA global similarity is below 0.8.

Connectivity is transitive: if A connects to B and B connects to C, A, B, and C are assigned together even when A and C do not directly reach 0.8. All sites and all HeLa/H9 records from the same RNA remain in the same partition. Groups are not constructed from y or model predictions. These rules control the specified forms of duplication but do not claim to remove every remote homology relationship or all cross-study confounding.

## Counts in final.json

| Role | Records | Purpose |
|---|---:|---|
| E | 4,775 | Fit selector-model parameters |
| S | 2,176 | Select the training epoch only |
| C | 1,416 | Select per-cell MCC thresholds only |
| test | 3,112 | Fixed evaluation partition |
| E+S | 6,951 | Reinitialize and refit model parameters after epoch selection |

The total is 11,479 records. E/S/C form the development set of 8,367 records, 383 RNAs, and 243 groups. The evaluation set contains 116 RNAs and 70 groups. The same RNA may appear in both cells, so per-cell RNA counts cannot be added directly. Proportions arise from assigning complete groups and are **not a forced random 70/15/15 split of sites**.

The 243 development groups are ordered by a fixed SHA256 procedure with the prespecified salt `20261101`: 49 groups are assigned to C, 49 to S, and 145 to E. The authoritative assignments are the candidate-ID lists in the JSON files. Do not resample the splits or retry them to improve class proportions or scores.

## dev0/1/2.json

These files use only the 8,367 development records and preserve the original three outer group folds. Within each outer training portion, fixed inner groups are assigned as follows: inner2/3/4 to E, inner0 to S, and inner1 to C. The held-out outer fold is the `test` role in that file. Therefore, `test` in these JSON files denotes the corresponding outer validation fold and is not the 3,112-record evaluation partition in `final.json`. Multiple training seeds represent different initializations, not new biological replicates or new data splits.

## Complete procedure for each arm and seed

1. Reset the Python, NumPy, and PyTorch/CUDA random states; create a fresh model and AdamW optimizer.
2. Train on E only for at most 20 epochs. At every epoch, calculate AUROC separately for HeLa and H9 on S and take their equal-weight mean. Update the best epoch only on a strict improvement; ties retain the earlier epoch. Do not use the last epoch, AP, or test for this choice.
3. Reinitialize a fresh model and optimizer with the same seed. Train on E+S for the number of epochs selected in step 2; do not reuse selector weights.
4. Predict C with the refitted model and select an MCC threshold separately within each cell. Search 0.01 to 0.99 in steps of 0.01. MCC ties favor the threshold closest to 0.5 and then the smaller value.
5. Freeze the model and thresholds before predicting the designated evaluation partition. Never reselect thresholds from evaluation labels.

AdamW uses learning rate 0.001 and weight decay 0.0001. Training batches contain 128 candidates, gradients are clipped at 5, and prediction batches contain 64 candidates. Training records are shuffled independently within each cell before batch events from the two cells are interleaved. The positive BCE weight is the cell-specific negative-to-positive ratio in the actual fitting partition. There is no RNA-matching ranking auxiliary objective, additional background branch, length or probability calibration, or RNA-FM fine-tuning.

For reported benchmark performance, all five seeds, 0/1/2/3/4, are retained and no seed is selected using evaluation-partition performance. Report each seed and cell before taking equal-weight averages. The separate deployment checkpoint is selected only from development information, as documented in `weights/README.md`. AP is average precision and must not be silently replaced by trapezoidal PR-AUC. Comparisons among RF, LightGBM, and neural models must use the same split manifests.

SHA256 digests for every fixed split are recorded in `manifest.json`. `scripts/check_data.py` verifies that groups and exact RNA sequences do not cross roles.
