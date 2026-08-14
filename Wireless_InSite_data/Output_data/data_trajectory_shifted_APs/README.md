# `data_trajectory_shifted_APs`

CSI along the UE trajectory in the scenario where **all the APs are shifted by half a
wavelength (6.25 cm) in the positive direction of the X-axis** with respect to their
true positions in Fig. 2. The floor plan, the materials and the UE height (1.5 m) are
unchanged. This is the mildest of the modeling mismatches of Sec. IV-D1, and the
counterpart of the DT data in `data_dt_shifted_APs/`.

| File | Shape | dtype | Content |
|---|---|---|---|
| `H_all.npy` | `(14606, 32, 52)` | `complex64` | CSI matrices along the trajectory. |

> `H_all.npy` is not tracked in the repository (~194 MB, over GitHub's 100 MB
> limit). Download it with `./download_data.sh` from the repository root, or
> individually as the `H_all_shifted_APs.npy` asset of the
> [`data-v1` release](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1).

Same format as `data_trajectory_standard/H_all.npy`: 32 rows = `A = 8` APs times
`K = 4` antennas per AP (rows `4a:4a+4` belong to AP `a`), 52 columns = `S = 52`
subcarriers, noiseless, normalized by the transmit power per antenna. Row `n`
corresponds to row `n` of `UE_pos_trajectory.npy` and `timestamps_trajectory.npy`
one level up.

Generated with `preprocessing_simulations/read_csi_to_H.py`.

## Where it appears in the paper

- **Table IV, *Shifted AP positions* column**: this is the CSI that the mismatched DT
  of `data_dt_shifted_APs/` synthesizes at the trajectory positions. Together with the
  true positions it forms the labeled training set of the DT-based CSI fingerprinting
  and DT-based *Power* fingerprinting baselines of Sec. IV-D1, which are then tested
  on the measured CSI of `data_trajectory_standard/`. The proposed approach in the
  same column does not use this file: it only needs the *Power* feature stored in
  `data_dt_shifted_APs/`.
