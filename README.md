# Positioning via Digital-Twin-Aided Channel Charting with Large-Scale CSI Features

Code of the paper *"Positioning via Digital-Twin-Aided Channel Charting with Large-Scale CSI Features"*, in IEEE Trans. Wireless Commun.

Channel charting (CC) is a self-supervised positioning technique whose estimated positions lie in an arbitrary coordinate system. This repository implements a framework that produces CC positions in **true spatial coordinates** with the aid of a digital twin (DT), by matching large-scale CSI features extracted from the measured CSI and from the DT with a cosine-similarity loss, which is then combined with a conventional CC loss.

## Repository structure

```
lib/
  simulation_parameters.py   Common setup: CLI arguments, data loading, noise variance
                             for the target SNR, and train/test split. Defaults follow Table I
  functions.py               Dataset, losses, training/testing loops and evaluation metrics
  neural_networks.py         MLP with softmax output and dropout
methods/                     Training and testing scripts (one per method, see below)
preprocessing_simulations/   Wireless InSite output -> CSI matrix H, DT grid, large-scale features
Wireless_InSite_data/
  Output_data/               Ray-tracing data used in the simulations
  Simulation_files/          Wireless InSite projects that generated that data
```

## Requirements

Python 3.12 with `torch`, `numpy`, `pandas`, `scikit-learn`.

## Usage

Before running anything, download the CSI trajectory files, which are distributed separately from the repository (see [Data](#data)):

```bash
./download_data.sh
```

All scripts are run as modules from the repository root, so that `lib` is importable:

```bash
mkdir -p models results
python -m methods.train_dropout --processing power --seed 1
```

`lib/simulation_parameters.py` is imported by every script and holds all the shared settings. It reads the data from `Wireless_InSite_data/Output_data/`, relative to the repository root, through three paths:

- `data_folder`: root folder of the data. It holds the true positions and the timestamps of the trajectory (`UE_pos_trajectory.npy`, `timestamps_trajectory.npy`), which are shared by all the scenarios.
- `data_trajectory_folder`: folder with the measured CSI of the UE trajectory (`H_all.npy`). Defaults to `data_trajectory_standard/`.
- `data_dt_folder`: folder with the DT data (`pos_matrix.npy`, `Power_matrix.npy`, `angle_profiles.npy`, `delay_profiles.npy`, `tdp.npy`, `cov_profiles_abs_fft.npy`). Defaults to `data_dt_0_5_spacing/`.

Change the last two to select a different scenario or DT. Each data folder contains a `README.md` that describes its files and the table or figure of the paper it corresponds to.

Other parameters (SNR, number of epochs, grid areas, etc.) are edited directly in that file. The remaining hyper-parameters are command-line arguments:

| Argument | Default | Meaning |
|---|---|---|
| `-n, --neurons` | 256 | Neurons per hidden layer |
| `-l, --layers` | 4 | Number of hidden layers |
| `-r, --lr` | 1e-3 | Learning rate |
| `-t, --triplet` | 1.0 | Weight of the CC triplet loss |
| `-f, --feature_lambda` | 10 | Weight of the DT-aided feature loss |
| `-s, --seed` | 1 | Random seed |
| `-m, --Mtidx` | 0.9 | Margin `Mt` of the triplet loss |
| `-d, --dropout` | 0.15 | Dropout rate |
| `-p, --processing` | `power` | Large-scale feature: `power`, `angle`, `delay`, `cov_abs_fft`, `tdp` |

Trained models are saved in `models/` and results in `results/`, with a file name built from the hyper-parameters above.

## Methods

| Script | Method |
|---|---|
| `methods/train_dropout.py` | Proposed DT-aided CC approach (Algorithm 1) |
| `methods/train_supervised.py` | CSI fingerprinting baseline |
| `methods/train_supervised_power.py` | Power fingerprinting baseline (received power per AP as NN input) |
| `methods/train_cc_affine_transform.py` | CC-only training followed by the optimal affine transformation |
| `methods/train_dropout_supervised.py` | Semi-supervised baseline (CC + supervised loss, fraction of labels set by `sl_ratio`) |
| `methods/test_dropout.py`, `methods/test_supervised.py`, `methods/test_supervised_power.py` | Inference with an already trained model |

### Reproducing the results of the paper

- **Table II and Fig. 3**: `train_dropout` (proposed), `train_supervised` (fingerprinting), `train_cc_affine_transform` (affine transformation).
- **Table III**: `train_dropout` with the different DT datasets (grid spacing `Delta`).
- **Fig. 4**: `train_dropout_supervised` for the CSI fingerprinting curve, sweeping `sl_ratio`.
- **Table IV**: `train_dropout` with the different datasets, `train_supervised` (fingerprinting) and `train_supervised_power` (Power fingerprinting).
- **Table V** (distribution shift): train with `train_dropout` / `train_supervised`, then test with `test_dropout` / `test_supervised`. If the testing dataset must contain all the points (dense trajectory), define the new testing dataset inside the `test_*` script (see the `load_folder` lines). If it must contain the same points as the original testing dataset, it is enough to change the folder name in `lib/simulation_parameters.py` and use `test_loader` without defining a new testing dataset.

The scripts to plot the results will be uploaded as soon as possible.

## Data

`Wireless_InSite_data/Output_data/` contains the ray-tracing data needed to run the simulations:

- `data_trajectory_*/`: measured CSI along the UE trajectory (`H_all.npy`) for each scenario (standard, wood, one wall, four walls, boxes, height 0.8 m, shifted APs). The ground-truth positions and timestamps of the trajectory are shared by all of them (`UE_pos_trajectory.npy`, `timestamps_trajectory.npy`).
- `data_dt_*/`: DT data, i.e., the grid positions and the large-scale features, for the different grid spacings, AP placements and material/geometry mismatches.

### Downloading the CSI trajectory files

The CSI trajectory files (`data_trajectory_*/H_all.npy`, `(14606, 32, 52)` `complex64`, ~194 MB each) are **not tracked in this repository**, since they exceed GitHub's 100 MB file limit. They are distributed as assets of the [`data-v1` release](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1). Every other data file is tracked as usual, so a fresh clone only needs this one extra step:

```bash
git clone https://github.com/josemateosramos/DT_CC.git
cd cc
./download_data.sh
```

The script downloads the seven files (~1.4 GB in total) into their `data_trajectory_*/` folders, which is where `lib/simulation_parameters.py` expects them. Files that are already present are skipped, so it can be re-run safely after an interrupted download.

To fetch only one scenario (~194 MB) instead of all seven, which is enough to reproduce everything that does not involve a distribution shift, download the default one with the [GitHub CLI](https://cli.github.com):

```bash
gh release download data-v1 --repo josemateosramos/DT_CC \
  --pattern "H_all_standard.npy" \
  --output Wireless_InSite_data/Output_data/data_trajectory_standard/H_all.npy
```

The assets can also be downloaded by hand from the [release page](https://github.com/josemateosramos/DT_CC/releases/tag/data-v1). Note that each asset is named `H_all_<scenario>.npy` there, because all seven files share the name `H_all.npy` and release assets need distinct names; a manually downloaded file must therefore be renamed to `H_all.npy` and placed in the matching `data_trajectory_<scenario>/` folder. The script handles this mapping automatically.

The scripts in `preprocessing_simulations/` generate these files from the raw Wireless InSite output: `read_csi_to_H.py` (CIR to CSI matrix `H`), `read_power.py` (received power per AP), `grid_positions.py` (DT grid of Fig. 2) and `compute_ls_feat_from_H.py` (angle-power, delay-power, covariance and truncated delay profiles).

### Wireless InSite project files

`Wireless_InSite_data/Simulation_files/` contains the ray-tracing projects that generated all the data above, so that the simulations can be reproduced or modified. They were built and run with **Remcom Wireless InSite® version 3.4.4.13**. Wireless InSite is a commercial product of [Remcom](https://www.remcom.com), which is not affiliated with this work, and a valid license is needed to open and run these projects.

Each folder is a self-contained project (`.setup`, `.txrx`, the `.flp` floor plan, the `.vw` view and the X3D geometry file; `Scenario boxes/` additionally ships the `Box*.object` files):

| Folder | Scenario |
|---|---|
| `Standard scenario/` | Nominal scenario of Fig. 2. It also contains the variants built on top of it: the UE trajectory at h = 0.8 m, the APs shifted by half a wavelength, and the DT grids with the different spacings |
| `Scenario all wood/` | Nominal scenario with all the walls made out of wood |
| `Scenario boxes/` | Nominal scenario with the boxes added in the largest room (Fig. 6) |
| `Scenario four walls/` | Nominal scenario with the four additional walls of Fig. 5 |
| `Scenario one wall/` | Nominal scenario with only one of those walls (wall 4 of Fig. 5) |

**The APs and the DT grid points are distributed in sets.** The APs are split into four point sets (`AP_Set_1_1` … `AP_Set_1_4`) and the DT grid into six grid sets, one per area of the floor plan (`UE_Set_1` … `UE_Set_6`). All the sets of the configuration being simulated must be **active at the same time**; otherwise part of the APs or of the DT grid is missing from the output. Their ordering is what fixes the AP ordering used throughout the data (see `Wireless_InSite_data/Output_data/README.md`).

**Select what to simulate.** Every project contains both the DT grid points and the UE trajectory, and only one of the two should be active in a given run:

- the `UE_Set_*` grids (0.5 m spacing, h = 1.5 m) produce the DT data of the `data_dt_*` folders;
- the `UE_trajectory` trajectory (h = 1.5 m, 2.5 cm point spacing at 1 m/s; named `First_trajectory` in `Scenario boxes/`) produces the measured CSI of the `data_trajectory_*` folders.

## Citation

If you use this code, please cite:

> J. M. Mateos-Ramos, F. Zumegen, H. Wymeersch, C. Häger, and C. Studer, "Positioning via digital-twin-aided channel charting with large-scale CSI features," *IEEE Trans. Wireless Commun.*, early access, Aug. 2026
