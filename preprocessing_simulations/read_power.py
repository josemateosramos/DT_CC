'''
File to read the received power at each RX antenna element
from Wireless InSite and compute the average power per position.
The power files should be under the same folder.

As in read_csi_to_H.py, the APs are distributed into 4 transmitter
sets (txSet) with 2 APs (txPt) per set, i.e., 4*2 = 8 APs in total,
and each AP has 4 antenna elements (txEl). The output power matrix
has one row per UE position and one column per AP, where the columns
follow the sorted (txSet, txPt) pairs. This is the same AP ordering
used to build the rows of H_all in read_csi_to_H.py, so that column a
of the power matrix and the ath block of H_all refer to the same AP.
The power of each AP is averaged over its antenna elements.
'''
import os
import numpy as np
import pandas as pd
import re
from collections import defaultdict

# Directory containing CSV files
#Paths must end with a slash. The power matrix is saved in one of the
#Wireless_InSite_data/Output_data/data_dt_* folders
data_folder = 'path_to_folder_with_pow_files/'
save_folder = 'path_to_save_results/'

# Regular expression to extract txSet, txPt, and rxSet from filenames
filename_pattern = re.compile(
    r"power\.txSet(\d+)\.txPt(\d+)\.rxSet(\d+)\.txEl(\d+)\.rxEl001\.inst\d+\.csv"
)

# Data storage
power_data = defaultdict(lambda: defaultdict(list))  # {rxSet: { (txSet, txPt): [powers] }}
tx_set_pt_pairs = set()  # Stores (txSet, txPt) unique pairs
rx_sets = set()
tx_el_set = set()
rx_points_set = []  #rx_points_set cannot be a set because there could be 2 elements with the same value (inserting 0 at the beginning is useful)

# Read and process each CSV file
for filename in os.listdir(data_folder):
    match = filename_pattern.match(filename)        ######I wanna change the name of the 'match' variable
    if not match:
        raise ValueError('There is a file in the folder that does not match the file format.') 

    tx_set, tx_pt, rx_set, tx_el = map(int, match.groups())  # Convert to integers
    
    file_path = os.path.join(data_folder, filename)

    with open(file_path, "r") as f:
        lines = f.readlines()           # Metadata lines are at the beginning starting with #

    # Find the first non-metadata line
    data_start = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    
    # Load the CSV data
    df = pd.read_csv(file_path, skiprows=data_start, header=None, usecols=[1])  #Each data row is organized as: index, RX power [W], phase [rad], path loss [dB], path gain [dB]
    received_powers = df.iloc[:, 0].values  # Extract power in W

    if len(power_data[rx_set][(tx_set, tx_pt)]) == 0:
        power_data[rx_set][(tx_set, tx_pt)] = np.zeros_like(received_powers)
    # Store power values and sum over antenna elements
    power_data[rx_set][(tx_set, tx_pt)] += np.array(received_powers)

    if rx_set not in rx_sets:   #It can happen that 2 different RX sets have the same number of points
        rx_points_set.append(len(received_powers))

    tx_set_pt_pairs.add((tx_set, tx_pt))
    rx_sets.add(rx_set)
    tx_el_set.add(tx_el)

#Take number of antenna elements of APs, assuming that all APs have the same number of antenna elements
numAntennas = len(tx_el_set)

# Sort TX sets and RX sets for consistent indexing
tx_set_pt_list = sorted(tx_set_pt_pairs)  # Sorted list of (txSet, txPt) pairs
rx_list = sorted(rx_sets)

# #Insert 0 at the beginning of the rx_points_list
# rx_points_list.insert(0,0)

# Create the matrix with zeros (default power is 0 W for missing sets)
total_rx_points = sum(rx_points_set)
power_matrix = np.zeros((total_rx_points, len(tx_set_pt_list)))

# Fill the matrix with averaged power values
prev_sum = 0        
for i, rx_set in enumerate(rx_list):
    for j, (tx_set, tx_pt) in enumerate(tx_set_pt_list):
        if (tx_set, tx_pt) in power_data[rx_set]:
            current_len = len(power_data[rx_set][(tx_set, tx_pt)])
            power_matrix[prev_sum:prev_sum + current_len, j] = power_data[rx_set][(tx_set, tx_pt)] / numAntennas  
    prev_sum += current_len    

np.save(save_folder + 'Power_matrix.npy', power_matrix)