# `data_dt_0_5_spacing`

Digital-twin data for a DT that **matches** the real scenario, with a grid spacing of
`Delta = 0.5 m` between predefined positions. This is the default DT of the paper:
it is the grid drawn as beige points in Fig. 2 and the value of Table I
(`P = 713` DT positions), and it is the DT used in every experiment except the grid
spacing study of Table III.

It holds the predefined positions `X` of Sec. II and the large-scale features `V` of
Sec. III-C, which enter the DT-aided loss of (5).

| File | Shape | dtype | `--processing` | Content |
|---|---|---|---|---|
| `pos_matrix.npy` | `(713, 2)` | `float64` | — | Predefined DT positions `X`, as `(x, y)` in metres. |
| `Power_matrix.npy` | `(713, 8)` | `float64` | `power` | *Power* feature of Sec. III-C1 (`D = 1` per AP). |
| `angle_profiles.npy` | `(713, 32)` | `float32` | `angle` | *Angle-power profile* (APP) of Sec. III-C2 (`D = K = 4` per AP). |
| `delay_profiles.npy` | `(713, 416)` | `float32` | `delay` | *Delay-power profile* (DPP) of Sec. III-C3 (`D = S = 52` per AP). |
| `cov_profiles_abs_fft.npy` | `(713, 128)` | `float32` | `cov_abs_fft` | *Covariance* feature of Sec. III-C4 (`D = K^2 = 16` per AP). |
| `tdp.npy` | `(713, 416)` | `float32` | `tdp` | *Truncated delay profile* (TDP) of Sec. III-C5 (`D = K*C = 52` per AP). |

Notes:

- Row `p` of every file is the same predefined position, and all the feature files
  concatenate the `A = 8` APs along the columns in the AP ordering described in the
  parent `README.md`, i.e., `v^(p) = [v^(1,p); ...; v^(A,p)]`.
- `Power_matrix.npy` is the received power in **watts** (linear scale), averaged over
  the `K = 4` antenna elements of each AP. It is 0 for the (AP, position) pairs that
  no ray reaches, which is the situation of Proposition 2.
- The grid height is 1.5 m (Table I), so only `(x, y)` are stored.
- The features are noiseless: no AWGN is ever added to the DT.

Produced by `preprocessing_simulations/grid_positions.py` (positions),
`read_power.py` (power) and `compute_ls_feat_from_H.py` (the remaining features, from
the CSI simulated at the grid points).

## Where it appears in the paper

- **Table II and Fig. 3e–3i**: all the *Proposed* rows, one per feature file.
- **Table III, `Delta = 0.5 m` column**: the reference spacing.
- **Fig. 4**: the *Power* curve of the proposed approach.
- **Table IV, *No mismatch* column**, and **Table V** (the DT is matched in all its
  columns).
