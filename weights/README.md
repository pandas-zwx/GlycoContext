# Released deployment checkpoint

`GlycoContext_dev_selected_seed4.pt` is the single checkpoint released for
sequence-wide inference. It was selected only from development information:

1. for each prespecified seed (0--4), training on E used S to select the epoch;
2. seed 4 had the highest best-epoch equal-cell macro AUROC on S;
3. that seed was freshly refit on E+S for five epochs; and
4. per-cell classification thresholds were selected on C.

The held-out final test was not used to select the seed, epoch, model parameters,
or thresholds. The checkpoint stores no cohort record identifiers. See
`manifest.json` for its checksum and auditable selection summary. Official
RNA-FM weights are a separate upstream dependency and are not redistributed.
