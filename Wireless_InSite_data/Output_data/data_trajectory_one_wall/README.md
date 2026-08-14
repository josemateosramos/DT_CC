# `data_trajectory_one_wall`

CSI along the UE trajectory in the scenario with **one additional wooden wall**, the
wall numbered 4 in the top horizontal corridor of Fig. 5. Everything else (floor
plan, materials, AP positions, UE height of 1.5 m) is that of the nominal scenario
of Fig. 2.

| File | Shape | dtype | Content |
|---|---|---|---|
| `H_all.npy` | `(14606, 32, 52)` | `complex64` | CSI matrices along the trajectory. |

> `H_all.npy` is not tracked in the repository (~194 MB, over GitHub's 100 MB
> limit). Download it with `./download_data.sh` from the repository root, or
> individually as the `H_all_one_wall.npy` asset of the
> [`data-v1` release](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1).

Same format as `data_trajectory_standard/H_all.npy`: 32 rows = `A = 8` APs times
`K = 4` antennas per AP (rows `4a:4a+4` belong to AP `a`), 52 columns = `S = 52`
subcarriers, noiseless, normalized by the transmit power per antenna. Row `n`
corresponds to row `n` of `UE_pos_trajectory.npy` and `timestamps_trajectory.npy`
one level up.

Generated with `preprocessing_simulations/read_csi_to_H.py`.

## Where it appears in the paper

This dataset is used in two different roles.

- **Table V, *Wooden wall* column** (as *real* data): a testing-only dataset. The NN
  is trained on `data_trajectory_standard/` and evaluated here, i.e., the wall
  appears in the environment after deployment. **Fig. 8** shows the corresponding
  estimated trajectories, where the positions in the corridor are pulled towards the
  APs because the added wall attenuates their signal.
- **Table IV, *One additional wooden wall* column** (as *synthetic DT* data): the
  same simulation also plays the role of the CSI that the mismatched DT of
  `data_dt_one_wall/` synthesizes at the trajectory positions, which is the labeled
  training set of the DT-based CSI fingerprinting and DT-based *Power* fingerprinting
  baselines of Sec. IV-D1. Those baselines are trained on this synthetic CSI and
  tested on the measured `data_trajectory_standard/`.
