# `data_trajectory_wood`

CSI along the UE trajectory in the scenario where the **walls and the floor are made
of wood** instead of the materials of Fig. 2 (default floor material, reinforced
concrete, brick and metal). The floor plan, the AP positions and the UE height
(1.5 m) are unchanged.

| File | Shape | dtype | Content |
|---|---|---|---|
| `H_all.npy` | `(14606, 32, 52)` | `complex64` | CSI matrices along the trajectory. |

> `H_all.npy` is not tracked in the repository (~194 MB, over GitHub's 100 MB
> limit). Download it with `./download_data.sh` from the repository root, or
> individually as the `H_all_wood.npy` asset of the
> [`data-v1` release](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1).

Same format as `data_trajectory_standard/H_all.npy`: 32 rows = `A = 8` APs times
`K = 4` antennas per AP (rows `4a:4a+4` belong to AP `a`), 52 columns = `S = 52`
subcarriers, noiseless, normalized by the transmit power per antenna. Row `n`
corresponds to row `n` of `UE_pos_trajectory.npy` and `timestamps_trajectory.npy`
one level up.

Generated with `preprocessing_simulations/read_csi_to_H.py`.

## Where it appears in the paper

- **Table IV, *Material mismatches* column**: this is the CSI that the mismatched DT
  of `data_dt_wood/` synthesizes at the trajectory positions. Together with the true
  positions it forms the labeled training set of the DT-based CSI fingerprinting and
  DT-based *Power* fingerprinting baselines of Sec. IV-D1, which are then tested on
  the measured CSI of `data_trajectory_standard/`. The proposed approach in the same
  column does not use this file: it only needs the *Power* feature stored in
  `data_dt_wood/`.
