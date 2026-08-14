'''
File to read the CIR from Wireless InSite and compute the CSI
matrix H for all positions. The CIR files should be under the
same folder.

Transmitter layout in Wireless InSite:
the APs are distributed into 4 transmitter sets (txSet) with 2 APs
(txPt) per set, i.e., 4*2 = 8 APs in total, and each AP is a uniform
linear array with 4 antenna elements (txEl). The UE is the receiver
(rxSet, rxPt) and has a single antenna element (rxEl), so the rxEl
dimension is squeezed at the end of the file.

Note that n_AP below counts the number of transmitter *sets* (4) and
not the number of APs (8). The APs are ordered by sorting the
(txSet, txPt) pairs, which is the same ordering used to build the
columns of the power matrix in read_power.py. Each AP occupies
n_Tx_ant consecutive rows of H_all, so that
H_all[:, a*n_Tx_ant:(a+1)*n_Tx_ant, :] is the CSI of the ath AP and
H_all has n_APs*n_Tx_ant = 8*4 = 32 rows, matching the aggregated CSI
matrix H = [H^(1); ...; H^(A)] used during training.
'''
import numpy as np
import numpy.fft as fft
import pandas as pd
import os
import re
from collections import defaultdict

# Parameters
class Params:
    bandwidth = 20e6  # Hz
    subcarriers = 52
    transmit_power_dBm = 23  # dBm

par = Params()

#Paths must end with a slash. The H matrix is saved in one of the
#Wireless_InSite_data/Output_data/data_trajectory_* folders
folder_cir = 'path_to_cir_folder/'
save_folder_h_matrix = 'path_to_save_data/'

# Precompute pilot grid for efficiency
pilot_grid = np.arange(par.subcarriers) / par.subcarriers

# Step 1: Identify max rxPt per rxSet and track unique (txSet, txPt)
rxSet_rxPt_counts = defaultdict(set)  # Dictionary to track unique rxPt per rxSet
txSet_txPt_pairs = set()  # Store unique (txSet, txPt) pairs
txEl_set = set()    #Store unique txEl numbers
rxEl_set = set()    #Store unique rxEl numbers

file_pattern = re.compile(r".*\.txSet(\d{3})\.txPt(\d{3})\.rxSet(\d{3})\.rxPt(\d{3,5})\.txEl(\d{3})\.rxEl(\d{3})\.")  

for filename in os.listdir(folder_cir):
    match = file_pattern.search(filename)
    if match:
        txSet = int(match.group(1))  # Extract txSet
        txPt = int(match.group(2))   # Extract txPt
        rxSet = int(match.group(3))  # Extract rxSet
        rxPt = int(match.group(4))   # Extract rxPt
        txEl = int(match.group(5))   # Extract unique Tx antenna elements (to know how many there are, assuming all APs have the same number)
        rxEl = int(match.group(6))   # Extract unique Rx antenna elements (again assuming all Rx have the same number of antennas)

        rxSet_rxPt_counts[rxSet].add(rxPt)  # Track unique rxPt per rxSet
        txSet_txPt_pairs.add((txSet, txPt))  # Store unique (txSet, txPt) pairs
        txEl_set.add(txEl)
        rxEl_set.add(rxEl)

# Sort (txSet, txPt) pairs to maintain order
sorted_txSet_txPt_pairs = sorted(txSet_txPt_pairs)

# Assign unique ordered index for each (txSet, txPt)
txSet_txPt_mapping = {pair: idx for idx, pair in enumerate(sorted_txSet_txPt_pairs)}

#Get the number of AP sets (4) and the number of Tx antennas per AP (4)
n_AP = len(np.unique([e[0] for e in txSet_txPt_pairs]))
n_Tx_ant = len(txEl_set)
n_Rx_ant = len(rxEl_set)
transmit_power_per_antenna = 10 ** (par.transmit_power_dBm / 10) * 10 ** (-3) / n_Tx_ant

# Compute U_total as sum of max rxPt per rxSet
U_total = sum(len(rxPt_set) for rxPt_set in rxSet_rxPt_counts.values())

# Assign a unique index to each (rxSet, rxPt) pair
user_index_mapping = {}
current_idx = 0
for rxSet in sorted(rxSet_rxPt_counts.keys()):  # Ensure rxSet is processed in order
    for rxPt in sorted(rxSet_rxPt_counts[rxSet]):
        user_index_mapping[(rxSet, rxPt)] = current_idx
        current_idx += 1

# Step 2: Preallocate the full matrix
H_all = np.zeros((U_total, len(sorted_txSet_txPt_pairs) * n_Tx_ant, n_Rx_ant, par.subcarriers), dtype=np.complex64)

# Step 3: Process each file (only one loop)
for filename in os.listdir(folder_cir):
    match = file_pattern.search(filename)
    if match:  # Skip files that don't match the pattern
        # Extract parameters from the filename
        txSet = int(match.group(1))
        txPt = int(match.group(2))
        rxSet = int(match.group(3))
        rxPt = int(match.group(4))
        txEl = int(match.group(5))
        rxEl = int(match.group(6))

        if ((rxSet, rxPt) in user_index_mapping) and ((txSet, txPt) in txSet_txPt_mapping):
            u_idx = user_index_mapping[(rxSet, rxPt)]  # Correct user index
            #Compute index directly. The stride is the number of antennas per AP,
            #which here equals n_AP because there are 4 sets and 4 antennas per AP
            idx = txSet_txPt_mapping[(txSet, txPt)] * n_AP + (txEl - 1)

            filepath = os.path.join(folder_cir, filename)
            
            try:
                tmp = pd.read_csv(filepath, skiprows=2, usecols=[2, 3, 4]).values
            except FileNotFoundError:
                tmp = np.array([[0, 0, 0]])  # Default if file is missing

            # Compute H directly
            power = np.sqrt(tmp[:, 0] / transmit_power_per_antenna)
            phase = np.exp(1j * tmp[:, 1])
            delay = tmp[:, 2]

            # if (delay==-1).sum()>0:
            #     print('Stop')

            H_all[u_idx, idx, rxEl - 1, :] = np.sum((power * phase)[:, None] * np.exp(-2j * np.pi * par.bandwidth * delay[:, None] * pilot_grid), axis=0)

if n_Rx_ant == 1:
    # H_all.squeeze(axis=-2)
    H_all = np.squeeze(H_all, axis=-2)

# Save the full matrix
np.save(save_folder_h_matrix + 'H_all.npy', H_all)

print('H matrix saved successfully')
