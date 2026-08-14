'''
Common setup of all the methods: it parses the simulation parameters, loads the
measured CSI and the DT data, computes the noise variance for the target SNR and
splits the dataset into training and testing sets. The default values correspond
to the simulation parameters of Table I.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.fft as fft
import argparse

from lib.functions import *
from lib.neural_networks import *

######## Simulation Parameters selected by user ########
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--neurons", type=int, default=256,
                    help="Integer to change the number of neurons per layer.")
parser.add_argument("-l", "--layers", type=int, default=4,
                    help="Integer to change the number of hidden layers.")
parser.add_argument("-r", "--lr", type=float, default=1e-3,
                    help="Learning rate of the training optimizer.")
parser.add_argument("-t", "--triplet", type=float, default=1.0,
                    help="Index to change the weight of the triplet loss.")
parser.add_argument("-f", "--feature_lambda", type=float, default=10,
                    help="Index to change the weight of the feature loss.")
parser.add_argument("-s", "--seed", type=int, default=1,
                    help="Index to select the random seed.")
parser.add_argument("-m", "--Mtidx", type=float, default=0.9,
                    help="Value of the hyper-parameter Mt in the triplet loss.")
parser.add_argument("-d", "--dropout", type=float, default=0.15,
                    help="Dropout rate of the NN.")
parser.add_argument("-p", "--processing", type=str, default='power', 
                    choices=['power', 'angle', 'delay', 'cov_abs_fft', 'tdp'],
                    help="String indicating which processing of the CSI and loss function to use.")
args = parser.parse_args()

#Fix seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
#Try to make results reproducible
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
#To use torch.use_deterministic_algorithms(True), we need the following sentences
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" #This will increase memory, more info: https://docs.nvidia.com/cuda/cublas/#results-reproducibility

# Sample Inputs
C = 13    # Number of subcarriers to keep after IFFT
Tc = 2    # Threshold for close/far timestamps
Mt = args.Mtidx
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data. The paths are relative to the repository root, from where the scripts are run
data_folder = 'Wireless_InSite_data/Output_data/'                   # Root folder with all the ray-tracing data
data_trajectory_folder = data_folder + 'data_trajectory_standard/'  # Measured CSI along the UE trajectory
data_dt_folder = data_folder + 'data_dt_0_5_spacing/'               # DT positions and large-scale features
#The true positions and the timestamps are shared by all the data_trajectory_* folders,
#since the UE follows the same trajectory in every scenario, so they are stored in data_folder
UE_positions = torch.tensor(np.load(data_folder + 'UE_pos_trajectory.npy'), dtype=torch.float32).to(device)  # Nx2 positions
timestamps = torch.tensor(np.load(data_folder + 'timestamps_trajectory.npy'), dtype=torch.float32).to(device)  # Nx1 timestamps
CSI_matrices = torch.tensor(np.load(data_trajectory_folder + 'H_all.npy'), dtype=torch.cfloat).to(device)  # (N, 32, 52) complex CSI
grid_positions = torch.tensor(np.load(data_dt_folder + 'pos_matrix.npy'), dtype=torch.float32).to(device)  # Nx2
received_powers = torch.tensor(np.load(data_dt_folder + 'Power_matrix.npy'), dtype=torch.float32).to(device)  # NxA
angle_profiles = torch.tensor(np.load(data_dt_folder + 'angle_profiles.npy'), dtype=torch.float32).to(device)  # Nx32
delay_profiles = torch.tensor(np.load(data_dt_folder + 'delay_profiles.npy'), dtype=torch.float32).to(device)  # Nx416
trunc_delay_prof = torch.tensor(np.load(data_dt_folder + 'tdp.npy'), dtype=torch.float32).to(device)  # Nx416
cov_profiles_fft_abs = torch.tensor(np.load(data_dt_folder + 'cov_profiles_abs_fft.npy'), dtype=torch.float32).to(device)

num_samples_total = UE_positions.shape[0]       #Number of samples
n_grid = grid_positions.shape[0]
n_APs  = received_powers.shape[1]

#Select the SNR and compute the noise std to yield that SNR
snr_db = 25.0
n_subcarriers = CSI_matrices.shape[-1]
n_antennas = int(CSI_matrices.shape[-2]/n_APs)
#Process the CSI to obtain the delay-domain representation
CSI_ifft = IFFTTruncated(CSI_matrices, C)
#Compute the variance of the noise corresponding to the target SNR
power_per_ap = torch.linalg.vector_norm(CSI_ifft.reshape(CSI_ifft.shape[0],n_APs,C*n_antennas),dim=-1)**2
noise_var = power_per_ap.max()/(C*n_antennas*10**(snr_db/10.0))

# Create dataset with raw CSI (positions are only used for testing)
dataset = ChannelChartingDataset(CSI_matrices, UE_positions, timestamps)

# Define the square areas used to split the testing dataset (only for debugging, not shown on the paper)
# We define each area by the bottom-left and top-right corners of a rectangle
coord_areas_min = [
    [0,0],              #Area 1
    [0,6.5],            #Area 2
    [3,6.5],            #Area 3
    [3,9],              #Area 4
    [3,13],             #Area 5
    [7,16]              #Area 6
]
coord_areas_max = [
    [9.5,6.5],          #Area 1
    [3,16],             #Area 2
    [9.5,9],            #Area 3
    [9.5,13],           #Area 4
    [9.5,16],           #Area 5
    [9.5,26.5]          #Area 6
]

# Split the dataset into training and testing sets
train_size = int(0.8 * num_samples_total)
#Do the random split manually since I will need the indices later on to split the test set into more areas
temp_randperm = torch.randperm(num_samples_total, device=device)
train_indices = temp_randperm[:train_size]
test_indices = temp_randperm[train_size:]
# DataLoaders
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Compute testing datasets for each area
list_dataloaders = computeTestAreas(dataset, test_indices, UE_positions, coord_areas_min, coord_areas_max)

# Print dataset sizes
print(f"Training dataset size: {len(train_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")
print(f"Testing dataset size (inside all areas): {[len(data_loader.dataset) for data_loader in list_dataloaders]}")

num_epochs = 5000
learning_rate = args.lr
layers = [args.neurons] * args.layers
layers.insert(0,C*32)       #Fix first layer dimension
dropout_rate = args.dropout

lambda_triplet       = args.triplet
lambda_feature       = args.feature_lambda
test_epoch_list = np.linspace(0,num_epochs-1,10).astype(int)    # List of epochs to test during training

save_prob_map = False   #True to save probability maps for all testing positions (bigger saved file)

processing = args.processing  #String that indicates which processing and loss function to use during training
