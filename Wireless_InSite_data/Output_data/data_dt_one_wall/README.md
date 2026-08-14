# `data_dt_one_wall`

Digital-twin data for a **mismatched DT that contains one additional wooden wall**
that does not exist in the real scenario: the wall numbered 4 in Fig. 5, in the top
horizontal corridor. The floor plan, the materials, the AP positions and the grid of
predefined positions (`Delta = 0.5 m`, `P = 713`) are otherwise those of
`data_dt_0_5_spacing/`, so `pos_matrix.npy` is identical to the one of the matched
DT.

| File | Shape | dtype | `--processing` | Content |
|---|---|---|---|---|
| `pos_matrix.npy` | `(713, 2)` | `float64` | — | Predefined DT positions `X`, as `(x, y)` in metres. |
| `Power_matrix.npy` | `(713, 8)` | `float64` | `power` | *Power* feature of Sec. III-C1 (`D = 1` per AP). |

Only the *Power* feature is stored, because Table IV reports the proposed approach
under DT mismatches with the *Power* feature alone. Running any other
`--processing` option with this folder therefore requires generating the missing
features first with `preprocessing_simulations/compute_ls_feat_from_H.py` from the
CSI simulated by this DT at the grid points.

The conventions are the ones of `data_dt_0_5_spacing/README.md`: row `p` is the same
predefined position in both files, the `A = 8` APs are ordered along the columns of
`Power_matrix.npy`, the power is in watts averaged over the `K = 4` antennas of each
AP and is 0 where no ray reaches the AP, the grid height is 1.5 m, and no noise is
added.

## Where it appears in the paper

- **Table IV, *One additional wooden wall* column**: this DT replaces
  `data_dt_0_5_spacing/` in the *Power* row, while the measured data stays
  `data_trajectory_standard/`. The DT-based fingerprinting baselines of the same
  column instead use the CSI synthesized by this DT along the trajectory, stored in
  `data_trajectory_one_wall/`.
- **Fig. 5**: the additional wall is wall number 4 of that figure.

Do not confuse this folder with `data_trajectory_one_wall/`, which contains the same
geometry but is used in Table V as a *real* environment that changes after
deployment.
