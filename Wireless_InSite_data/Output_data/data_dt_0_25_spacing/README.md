# `data_dt_0_25_spacing`

Digital-twin data for a DT that matches the real scenario, with a **grid spacing of
`Delta = 0.25 m`** between predefined positions, i.e., the finest grid of the study
of Table III. It covers the same floor plan as the default DT, which results in
`P = 2775` predefined positions instead of the 713 of `data_dt_0_5_spacing/`.

| File | Shape | dtype | `--processing` | Content |
|---|---|---|---|---|
| `pos_matrix.npy` | `(2775, 2)` | `float64` | — | Predefined DT positions `X`, as `(x, y)` in metres. |
| `Power_matrix.npy` | `(2775, 8)` | `float64` | `power` | *Power* feature of Sec. III-C1 (`D = 1` per AP). |
| `angle_profiles.npy` | `(2775, 32)` | `float32` | `angle` | *Angle-power profile* (APP) of Sec. III-C2 (`D = K = 4` per AP). |
| `delay_profiles.npy` | `(2775, 416)` | `float32` | `delay` | *Delay-power profile* (DPP) of Sec. III-C3 (`D = S = 52` per AP). |
| `cov_profiles_abs_fft.npy` | `(2775, 128)` | `float32` | `cov_abs_fft` | *Covariance* feature of Sec. III-C4 (`D = K^2 = 16` per AP). |
| `tdp.npy` | `(2775, 416)` | `float32` | `tdp` | *Truncated delay profile* (TDP) of Sec. III-C5 (`D = K*C = 52` per AP). |

The conventions are the ones of `data_dt_0_5_spacing/README.md`: row `p` is the same
predefined position in every file, the `A = 8` APs are concatenated along the
columns, the power is in watts averaged over the `K = 4` antennas of each AP, the
grid height is 1.5 m, and no noise is added.

Note that `P` changes the output dimension of the last layer of the MLP, since the
NN outputs the probability mass function `p` over the `P` predefined positions.

## Where it appears in the paper

- **Table III, `Delta = 0.25 m` column**, for the *Power*, *APP* and *DPP* features.
  This is the spacing that shows the performance drop and the large standard
  deviation across NN initializations discussed in Sec. IV-C1.
