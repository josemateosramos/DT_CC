# Output data

Ray-tracing data produced with Remcom's Wireless InSite for the indoor scenario of
Fig. 2 of the paper, post-processed with the scripts in `preprocessing_simulations/`.
Every folder in here contains its own `README.md` describing its files and the
table/figure of the paper it belongs to.

> **The `H_all.npy` files are not tracked in this repository.** Each one is ~194 MB
> and exceeds GitHub's 100 MB file limit, so they are distributed as assets of the
> [`data-v1` release](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1).
> Everything else in here is tracked as usual.
>
> Run `./download_data.sh` from the repository root to place all seven of them
> (~1.4 GB) in their `data_trajectory_*/` folders. To fetch a single scenario
> instead, use the [GitHub CLI](https://cli.github.com):
>
> ```bash
> gh release download data-v1 --repo josemateosramos/DT_CC \
>   --pattern "H_all_<scenario>.npy" \
>   --output Wireless_InSite_data/Output_data/data_trajectory_<scenario>/H_all.npy
> ```
>
> The release assets are named `H_all_<scenario>.npy` because release assets need
> distinct names; each one must end up as `H_all.npy` inside its own folder.

## Shared files of the UE trajectory

The two files at this level are shared by **all** `data_trajectory_*` folders, since
the UE follows the same trajectory in every scenario and only the propagation
environment changes:

| File | Shape | dtype | Content |
|---|---|---|---|
| `UE_pos_trajectory.npy` | `(14606, 2)` | `float64` | Ground-truth UE positions `x^(n)` of Sec. II, as `(x, y)` in metres. |
| `timestamps_trajectory.npy` | `(14606, 1)` | `float64` | Timestamps `t^(n)` of Sec. II, in seconds. |

Details:

- `UE_pos_trajectory.npy` are the true positions of the trajectory shown in Fig. 3a.
  They cover all the rooms of the floor plan, `x` in `[0.09, 9.33] m` and `y` in
  `[0.17, 26.17] m`. Consecutive positions are 2.5 cm apart (Table I), and the UE
  height is constant and therefore not stored (1.5 m, or 0.8 m for
  `data_trajectory_height_0_8/`).
- `timestamps_trajectory.npy` starts at 0 s and ends at 365.125 s with a constant
  step of 25 ms, i.e., the UE moves at 1 m/s. The timestamps are only used to build
  the set of triplets `T` in (2) through the coherence times `Tc = Tf = 2 s`, so what
  matters is the time difference between samples and not the absolute origin.
- Row `n` of both files corresponds to row `n` of `H_all.npy` in every
  `data_trajectory_*` folder. `N = 14606` is the number of UE positions of Table I.

## Folder overview

Measured (real-scenario) CSI along the UE trajectory:

| Folder | Scenario | Used in |
|---|---|---|
| `data_trajectory_standard/` | Nominal scenario of Fig. 2 | Table II, Fig. 3, Table III, Fig. 4, Table IV, Table V |
| `data_trajectory_height_0_8/` | UE height lowered to 0.8 m | Table V, Fig. 7 |
| `data_trajectory_boxes/` | Wooden boxes added in the large room (Fig. 6) | Table V |
| `data_trajectory_one_wall/` | One additional wooden wall (wall 4 of Fig. 5) | Table V, Fig. 8; Table IV |
| `data_trajectory_wood/` | Wall and floor materials changed to wood | Table IV |
| `data_trajectory_four_walls/` | Four additional walls (Fig. 5) | Table IV |

DT data (predefined positions `X` and large-scale features `V`):

| Folder | DT | Used in |
|---|---|---|
| `data_dt_0_5_spacing/` | Nominal DT, grid spacing `Delta = 0.5 m` (Fig. 2) | Table II, Fig. 3, Table III, Fig. 4, Table IV, Table V |
| `data_dt_0_25_spacing/` | `Delta = 0.25 m` | Table III |
| `data_dt_1_spacing/` | `Delta = 1 m` | Table III |
| `data_dt_1_5_spacing/` | `Delta = 1.5 m` | Table III |
| `data_dt_shifted_APs/` | APs shifted by half a wavelength (6.25 cm) along +X | Table IV |
| `data_dt_wood/` | Wall and floor materials changed to wood | Table IV |
| `data_dt_one_wall/` | One additional wooden wall (wall 4 of Fig. 5) | Table IV |
| `data_dt_four_walls/` | Four additional walls (Fig. 5) | Table IV |

The `data_trajectory_wood/`, `data_trajectory_four_walls/` and
`data_trajectory_one_wall/` folders play a double role: they also contain the CSI
synthesized by the *mismatched DT* along the trajectory positions, which is what
trains the DT-based CSI fingerprinting and DT-based *Power* fingerprinting baselines
of Table IV.

## Conventions common to all folders

- **AP ordering.** There are `A = 8` APs with `K = 4` antenna elements each
  (Table I). The APs are sorted by their `(txSet, txPt)` pair in Wireless InSite, and
  the same ordering is used everywhere: column `a` of `Power_matrix.npy`, rows
  `4a:4a+4` of `H_all.npy`, and the `a`th block of the concatenated large-scale
  feature vectors all refer to the same AP.
- **Positions.** All coordinates are in metres in the coordinate system of the
  Wireless InSite scenario, and are two-dimensional (`Dx = 2`) because the AP height
  (2 m), the UE height (1.5 m) and the DT grid height (1.5 m) are fixed.
- **No noise.** The CSI and the features are noiseless. The AWGN that yields the
  25 dB SNR of Table I is added at runtime in `lib/simulation_parameters.py`, and it
  is never added to the DT.

## Using the data

`lib/simulation_parameters.py` reads the two shared files above from `data_folder`,
which points to this directory, and then one `data_trajectory_*` folder and one
`data_dt_*` folder through `data_trajectory_folder` and `data_dt_folder`. The
defaults are `data_trajectory_standard/` and `data_dt_0_5_spacing/`, and the paths
are relative to the repository root, from where the scripts are run as modules.

The distribution shift tests of Table V load a second trajectory folder through
`load_folder` in `methods/test_dropout.py` and `methods/test_supervised.py`, which
defaults to `data_trajectory_height_0_8/`.
