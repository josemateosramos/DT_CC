# `data_trajectory_standard`

Measured CSI along the UE trajectory in the **nominal indoor scenario** of Fig. 2,
i.e., the real environment without any modification. This is the default measured
dataset `{H^(n), t^(n)}` of Sec. II and the one used in all the experiments unless a
distribution shift is being evaluated.

| File | Shape | dtype | Content |
|---|---|---|---|
| `H_all.npy` | `(14606, 32, 52)` | `complex64` | Estimated CSI matrices `H^(n)` of Sec. II. |

> `H_all.npy` is not tracked in the repository (~194 MB, over GitHub's 100 MB
> limit). Download it with `./download_data.sh` from the repository root, or
> individually as the `H_all_standard.npy` asset of the
> [`data-v1` release](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1).

`H_all[n]` is the aggregated CSI matrix `H^(n) = [H^(1,n); ...; H^(A,n)]` at timestamp
`t^(n)`: the 32 rows are the `A = 8` APs times the `K = 4` antenna elements per AP
(rows `4a:4a+4` belong to AP `a`, in the ordering described in the parent
`README.md`), and the 52 columns are the `S = 52` OFDM subcarriers of a 20 MHz band
at 2.4 GHz. The channel is normalized by the transmit power per antenna, so it is a
channel response and not a received signal, and it is noiseless: the AWGN for the
25 dB SNR of Table I is added in `lib/simulation_parameters.py`.

Row `n` corresponds to row `n` of `UE_pos_trajectory.npy` and
`timestamps_trajectory.npy` one level up. The 80/20 train/test split of Sec. IV-B is
applied to these 14606 samples.

Generated from the Wireless InSite CIR files with
`preprocessing_simulations/read_csi_to_H.py`.

## Where it appears in the paper

- **Table II and Fig. 3**: measured data for the proposed method and all baselines.
- **Table III**: measured data, combined with each `data_dt_*_spacing/` DT.
- **Fig. 4**: measured data for the semi-supervised sweep.
- **Table IV**: measured data (the mismatch is on the DT side); it also provides the
  synthetic training CSI of the DT-based fingerprinting baselines in the
  *No mismatch* column, where the DT matches the real scenario.
- **Table V**: the *No mismatch* column, and the training data for all the other
  columns.
