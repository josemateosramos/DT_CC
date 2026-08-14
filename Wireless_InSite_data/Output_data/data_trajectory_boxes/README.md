# `data_trajectory_boxes`

Measured CSI along the UE trajectory in the scenario with **additional wooden boxes
in the large room**, shown in Fig. 6. The boxes measure 1 m along all the axes. The
rest of the environment, the AP positions and the UE height (1.5 m) are those of the
nominal scenario of Fig. 2.

| File | Shape | dtype | Content |
|---|---|---|---|
| `H_all.npy` | `(14606, 32, 52)` | `complex64` | Estimated CSI matrices `H^(n)` of Sec. II. |

> `H_all.npy` is not tracked in the repository (~194 MB, over GitHub's 100 MB
> limit). Download it with `./download_data.sh` from the repository root, or
> individually as the `H_all_boxes.npy` asset of the
> [`data-v1` release](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1).

Same format as `data_trajectory_standard/H_all.npy`: 32 rows = `A = 8` APs times
`K = 4` antennas per AP (rows `4a:4a+4` belong to AP `a`), 52 columns = `S = 52`
subcarriers, noiseless, normalized by the transmit power per antenna. Row `n`
corresponds to row `n` of `UE_pos_trajectory.npy` and `timestamps_trajectory.npy`
one level up.

Generated with `preprocessing_simulations/read_csi_to_H.py`.

## Where it appears in the paper

- **Table V, *Wooden boxes* column**: this is a *testing-only* dataset. The NN is
  trained on `data_trajectory_standard/` and evaluated here, which is the second type
  of distribution shift of Sec. IV-D2 (change in the environmental model after
  deployment). The DT used during training stays matched to the nominal scenario.
- **Fig. 6**: the scenario that this data corresponds to.
