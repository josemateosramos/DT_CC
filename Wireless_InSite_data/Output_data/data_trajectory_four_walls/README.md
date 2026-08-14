# `data_trajectory_four_walls`

CSI along the UE trajectory in the scenario with the **four additional walls**
numbered 1 to 4 in Fig. 5. This is the most severe of the modeling mismatches
considered in Sec. IV-D1. The floor plan, the materials, the AP positions and the UE
height (1.5 m) are otherwise those of Fig. 2.

| File | Shape | dtype | Content |
|---|---|---|---|
| `H_all.npy` | `(14606, 32, 52)` | `complex64` | CSI matrices along the trajectory. |

> `H_all.npy` is not tracked in the repository (~194 MB, over GitHub's 100 MB
> limit). Download it with `./download_data.sh` from the repository root, or
> individually as the `H_all_four_walls.npy` asset of the
> [`data-v1` release](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1).

Same format as `data_trajectory_standard/H_all.npy`: 32 rows = `A = 8` APs times
`K = 4` antennas per AP (rows `4a:4a+4` belong to AP `a`), 52 columns = `S = 52`
subcarriers, noiseless, normalized by the transmit power per antenna. Row `n`
corresponds to row `n` of `UE_pos_trajectory.npy` and `timestamps_trajectory.npy`
one level up.

Generated with `preprocessing_simulations/read_csi_to_H.py`.

## Where it appears in the paper

- **Table IV, *Four additional walls* column**: this is the CSI that the mismatched
  DT of `data_dt_four_walls/` synthesizes at the trajectory positions. Together with
  the true positions it forms the labeled training set of the DT-based CSI
  fingerprinting and DT-based *Power* fingerprinting baselines of Sec. IV-D1, which
  are then tested on the measured CSI of `data_trajectory_standard/`. The proposed
  approach in the same column does not use this file: it only needs the *Power*
  feature stored in `data_dt_four_walls/`.
- **Fig. 5**: the scenario that this data corresponds to.
