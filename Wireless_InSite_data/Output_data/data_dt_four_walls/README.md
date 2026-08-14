# `data_dt_four_walls`

Digital-twin data for a **mismatched DT that contains four additional walls** that do
not exist in the real scenario, numbered 1 to 4 in Fig. 5. This is the most severe of
the modeling mismatches of Sec. IV-D1. The floor plan, the materials, the AP
positions and the grid of predefined positions (`Delta = 0.5 m`, `P = 713`) are
otherwise those of `data_dt_0_5_spacing/`, so `pos_matrix.npy` is identical to the
one of the matched DT.

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

- **Table IV, *Four additional walls* column**: this DT replaces
  `data_dt_0_5_spacing/` in the *Power* row, while the measured data stays
  `data_trajectory_standard/`. The DT-based fingerprinting baselines of the same
  column instead use the CSI synthesized by this DT along the trajectory, stored in
  `data_trajectory_four_walls/`. This is the case where the proposed approach
  outperforms DT-based CSI fingerprinting, showing the robustness of the large-scale
  features under severe mismatches.
- **Fig. 5**: the scenario that this DT corresponds to.
