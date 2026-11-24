# Configuration and ablation definitions

`train.json` defines the reference training protocol. `preparation.json` fixes the pretrained weights, representation layer, sequence-length limit, and RNAfold version. `ablations.json` defines ablations of the current main model; it is not a list of experimental results. `baselines.json` defines the fixed candidate configurations for the RF and RNA-FM-only LightGBM baselines; see the [baseline documentation](../baselines/README.md) for full training specifications.

| Arm | Question | Exact definition |
|---|---|---|
| full | Main model | Add the three branch logits and retain cell-conditioned readouts |
| no_cnn | Are local CNN features useful? | Set the CNN feature score to zero while retaining its cell-specific intercept |
| no_gnn | Is the explicit graph branch complementary? | Set the GNN score to zero while retaining the CNN structure channels and RNA-FM |
| no_rnafm | Is frozen pretrained context useful? | Set the RNA-FM branch and its cell interaction to zero |
| no_rnafold | Is explicit RNAfold information useful? | Zero the CNN pairing channels and GNN pairing messages while retaining backbone messages |
| no_cell | Is explicit cell identity useful? | Use one shared CNN intercept and remove the RNA-FM cell-interaction term |
| no_cnn_structure | Do RNAfold channels contribute through the CNN? | Zero only the CNN pairing indicator and signed-span channels while retaining GNN pairing messages |
| no_gnn_pair | Do long-range RNAfold graph messages contribute? | Remove only GNN pairing messages while retaining CNN structure channels and backbone messages |

Every ablation must be retrained from scratch and must select its own threshold from its own C predictions. It must not reuse the full model's threshold or disable a branch only at inference time. All arms retain the same registered parameter inventory and initialization, but zeroing components changes effective capacity; these are not strictly capacity-matched experiments. `no_rnafold` does not remove all possible structural knowledge because RNA-FM may encode implicit structural information. `no_cnn_structure` and `no_gnn_pair` isolate the two routes by which explicit RNAfold information enters the CNN and GNN, respectively; both retain RNA-FM. `no_cell` removes only explicit identity input, while the training batches and positive-class weighting protocol remain unchanged.

Any additional CNN-only, RNA-FM-only, or parameter-matched control should be introduced as a separate configuration rather than redefining the existing arms. Claims about dynamic routing or a mixture of experts would require a corresponding method and strong static controls; the current fixed equal-weight RNA-context readout must not be called a mixture of experts.

`incremental.json` separately defines the fixed-order incremental component analysis on the development sets:

| Step | Newly added effective information or branch |
|---|---|
| inc_cnn_seq | Local sequence CNN without cell conditioning or explicit structure channels |
| inc_cnn_structure | The two explicit RNAfold channels in the CNN |
| inc_rnafm | Frozen RNA-FM branch with three contexts |
| inc_gnn_backbone | GNN branch containing adjacent backbone edges only |
| no_cell | Probability-weighted RNAfold base-pair edges; reuses an existing result |
| full | Cell-conditioned readouts; reuses an existing result |

This path is permitted only on `dev0/1/2`. Differences between adjacent stages are interpreted only as conditional marginal contributions under the prespecified order. The analysis complements rather than replaces leave-one-component-out ablation, and its differences are not described as order-independent contributions or Shapley values.

The training script accepts an explicit `--max-epochs` override for small technical checks and records that override in the run specification. Such a run is not equivalent to the standard maximum-20-epoch protocol. This directory contains no historical scores, predictions, or ablation conclusions.
