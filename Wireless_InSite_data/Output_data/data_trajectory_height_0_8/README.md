# `data_trajectory_height_0_8`

Measured CSI along the UE trajectory when the **UE height is lowered from 1.5 m to
0.8 m**. The floor plan, the materials and the AP positions are the ones of the
nominal scenario of Fig. 2; only the height of the receiver changes.

| File | Shape | dtype | Content |
|---|---|---|---|
| `H_all.npy` | `(14606, 32, 52)` | `complex64` | Estimated CSI matrices `H^(n)` of Sec. II. |

> `H_all.npy` is not tracked in the repository (~194 MB, over GitHub's 100 MB
> limit). Download it with `./download_data.sh` from the repository root, or
> individually as the `H_all_height_0_8.npy` asset of the
> [`data-v1` release](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1).

Same format as `data_trajectory_standard/H_all.npy`: 32 rows = `A = 8` APs times
`K = 4` antennas per AP (rows `4a:4a+4` belong to AP `a`), 52 columns = `S = 52`
subcarriers, noiseless, normalized by the transmit power per antenna. Row `n`
corresponds to row `n` of `UE_pos_trajectory.npy` and `timestamps_trajectory.npy`
one level up; the horizontal `(x, y)` positions are unchanged, only the height (which
is not stored) differs.

Generated with `preprocessing_simulations/read_csi_to_H.py`.

## Where it appears in the paper

- **Table V, *Decreased UE height* column**: this is a *testing-only* dataset. The NN
  is trained on `data_trajectory_standard/` at 1.5 m and evaluated here, which is the
  first type of distribution shift of Sec. IV-D2 (change in the UE position).
- **Fig. 7**: the estimated trajectories for the *Power* feature and for CSI
  fingerprinting under this height change, which show the pathloss effect that pushes
  the estimated positions away from the APs.
