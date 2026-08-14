# `data_dt_1_5_spacing`

Digital-twin data for a DT that matches the real scenario, with a **grid spacing of
`Delta = 1.5 m`** between predefined positions, the coarsest grid of the study of
Table III. It covers the same floor plan as the default DT, which results in
`P = 98` predefined positions instead of the 713 of `data_dt_0_5_spacing/`.

| File | Shape | dtype | `--processing` | Content |
|---|---|---|---|---|
| `pos_matrix.npy` | `(98, 2)` | `float64` | — | Predefined DT positions `X`, as `(x, y)` in metres. |
| `Power_matrix.npy` | `(98, 8)` | `float64` | `power` | *Power* feature of Sec. III-C1 (`D = 1` per AP). |
| `angle_profiles.npy` | `(98, 32)` | `float32` | `angle` | *Angle-power profile* (APP) of Sec. III-C2 (`D = K = 4` per AP). |
| `delay_profiles.npy` | `(98, 416)` | `float32` | `delay` | *Delay-power profile* (DPP) of Sec. III-C3 (`D = S = 52` per AP). |
| `cov_profiles_abs_fft.npy` | `(98, 128)` | `float32` | `cov_abs_fft` | *Covariance* feature of Sec. III-C4 (`D = K^2 = 16` per AP). |
| `tdp.npy` | `(98, 416)` | `float32` | `tdp` | *Truncated delay profile* (TDP) of Sec. III-C5 (`D = K*C = 52` per AP). |

The conventions are the ones of `data_dt_0_5_spacing/README.md`: row `p` is the same
predefined position in every file, the `A = 8` APs are concatenated along the
columns, the power is in watts averaged over the `K = 4` antennas of each AP, the
grid height is 1.5 m, and no noise is added.

Note that `P` changes the output dimension of the last layer of the MLP, since the
NN outputs the probability mass function `p` over the `P` predefined positions. It
also bounds the positioning error, since the estimated positions are confined to the
convex hull of these 98 points.

## Where it appears in the paper

- **Table III, `Delta = 1.5 m` column**, for the *Power*, *APP* and *DPP* features.
  This column supports the conclusion of Sec. IV-C1 that the *Power* feature is
  robust to a sparser DT grid, which reduces the DT complexity without a substantial
  performance loss.
