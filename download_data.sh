#!/usr/bin/env bash
# Downloads the CSI trajectory files (H_all.npy) that are distributed as GitHub
# release assets instead of being tracked in the repository, since each one
# exceeds GitHub's 100 MB file limit.
#
# Run from the repository root:
#   ./download_data.sh
#
# Each file is ~194 MB, ~1.4 GB in total. Files that are already present are
# skipped, so the script can be re-run after an interrupted download.
#
# While the repository is private, the assets are not publicly reachable and the
# download needs an authenticated GitHub CLI (https://cli.github.com):
#   gh auth login
# Once the repository is public, plain curl is enough and the GitHub CLI is
# optional.

set -euo pipefail

REPO="josemateosramos/DT_CC"
TAG="data-v1"
BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
DEST_ROOT="Wireless_InSite_data/Output_data"

SCENARIOS=(standard wood one_wall four_walls boxes height_0_8 shifted_APs)

if [ ! -d "${DEST_ROOT}" ]; then
    echo "Error: run this script from the repository root (${DEST_ROOT} not found)." >&2
    exit 1
fi

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    USE_GH=1
else
    USE_GH=0
    echo "GitHub CLI not available or not logged in, falling back to curl."
    echo "If the repository is still private, the download will fail: run 'gh auth login' first."
fi

for scenario in "${SCENARIOS[@]}"; do
    dest="${DEST_ROOT}/data_trajectory_${scenario}/H_all.npy"
    if [ -f "${dest}" ]; then
        echo "Skipping ${scenario}: ${dest} already exists."
        continue
    fi
    mkdir -p "$(dirname "${dest}")"
    echo "Downloading ${scenario} to ${dest} ..."
    if [ "${USE_GH}" -eq 1 ]; then
        gh release download "${TAG}" --repo "${REPO}" \
            --pattern "H_all_${scenario}.npy" --output "${dest}.part" --clobber
    else
        curl -fL --progress-bar -o "${dest}.part" "${BASE_URL}/H_all_${scenario}.npy"
    fi
    mv "${dest}.part" "${dest}"
done

echo "Done. All CSI trajectory files are in ${DEST_ROOT}/data_trajectory_*/."
