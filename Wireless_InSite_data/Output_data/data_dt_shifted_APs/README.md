# `data_dt_shifted_APs`

Digital-twin data for a **mismatched DT in which all the APs are shifted by half a
wavelength (6.25 cm) in the positive direction of the X-axis** with respect to their
true positions in Fig. 2. This is the mildest of the modeling mismatches of
Sec. IV-D1. The floor plan, the materials and the grid of predefined positions
(`Delta = 0.5 m`, `P = 713`) are those of `data_dt_0_5_spacing/`; only the AP
locations used to simulate the features differ, so `pos_matrix.npy` is identical to
the one of the matched DT.

| File | Shape | dtype | `--processing` | Content |
|---|---|---|---|---|
| `pos_matrix.npy` | `(713, 2)` | `float64` | — | Predefined DT positions `X`, as `(x, y)` in metres. |
| `Power_matrix.npy` | `(713, 8)` | `float64` | `power` | *Power* feature of Sec. III-C1 (`D = 1` per AP). |
| `angle_profiles.npy` | `(713, 32)` | `float32` | `angle` | *Angle-power profile* (APP) of Sec. III-C2 (`D = K = 4` per AP). |
| `delay_profiles.npy` | `(713, 416)` | `float32` | `delay` | *Delay-power profile* (DPP) of Sec. III-C3 (`D = S = 52` per AP). |
| `cov_profiles_abs_fft.npy` | `(713, 128)` | `float32` | `cov_abs_fft` | *Covariance* feature of Sec. III-C4 (`D = K^2 = 16` per AP). |
| `tdp.npy` | `(713, 416)` | `float32` | `tdp` | *Truncated delay profile* (TDP) of Sec. III-C5 (`D = K*C = 52` per AP). |

The conventions are the ones of `data_dt_0_5_spacing/README.md`: row `p` is the same
predefined position in every file, the `A = 8` APs are concatenated along the
columns, the power is in watts averaged over the `K = 4` antennas of each AP, the
grid height is 1.5 m, and no noise is added.

## Where it appears in the paper

- **Table IV, *Shifted AP positions* column**: this DT replaces
  `data_dt_0_5_spacing/` while the measured data stays `data_trajectory_standard/`.
  The displacement barely changes the large-scale *Power* feature, so the proposed
  approach is nearly unaffected, whereas it substantially alters the phase of the
  synthesized CSI and therefore degrades DT-based CSI fingerprinting.
