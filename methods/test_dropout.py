# -*- coding: utf-8 -*-
'''
Inference with a model trained with train_dropout.py on a testing dataset that
differs from the training one. This reproduces the distribution shift tests of
Sec. IV-D2, where the CSI of the new scenario is loaded from load_folder.
'''
from lib.simulation_parameters import *

# Define Network and optimizer
model = MLPSoftmaxOutputDropout(layers, n_grid, dropout_rate).to(device)

criterion = TripletLoss(margin=Mt)

#Create file_name to load and print it to debug
load_name = 'train_dropout_' + str(np.round(dropout_rate, 2)) + '_n_' + str(args.neurons) + '_l_' + str(args.layers)
load_name += '_Mt_' + str(np.round(Mt,1))
load_name += '_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs)
load_name += '_snr_' + str(snr_db) + '_dB'
load_name += '_triplet_' + str(lambda_triplet) 
load_name += '_feature_' + '{:.0e}'.format(lambda_feature)   
load_name += '_loss_' + processing
load_name += '_realization_' + str(args.seed)
print(load_name)

# Testing outside area specified in simulation_parameters.py#Select used features depending on the chosen input
if processing == 'power':
    grid_features = torch.clone(received_powers)
elif processing == 'angle':
    grid_features = torch.clone(angle_profiles)
elif processing == 'delay':
    grid_features = torch.clone(delay_profiles)
elif processing == 'tdp':
    grid_features = torch.clone(trunc_delay_prof)
elif processing == 'cov_abs_fft':
    grid_features = torch.clone(cov_profiles_fft_abs)

#Load trained model
checkpoint = torch.load('models/' + load_name)
model.load_state_dict(checkpoint['model'])
model.to(device)

# Load the new testing dataset
# Comment out these 4 lines and use test_loader as input of 
#testNN if the same testing points as the default testing want to be used
load_folder = data_folder + 'data_trajectory_height_0_8/'
CSI_matrices = torch.tensor(np.load(load_folder + 'H_all.npy'), dtype=torch.cfloat).to(device)  # (N, 32, 52) complex CSI
# Create dataset with raw CSI (positions are only used for testing)
dataset_test = ChannelChartingDataset(CSI_matrices, UE_positions, timestamps)
test_loader_new = DataLoader(dataset_test, batch_size=len(test_dataset), shuffle=False)

predicted_positions, actual_positions, probabilities_map, \
    pos_loss_test, triplet_loss_test, feature_loss_test, \
        tw_test, ct_test, ks_test, rd_test, perc_95_test = \
    testNN(test_loader_new, model, Tc, noise_var, criterion, n_APs, grid_positions, grid_features,
           columns=C, processing=processing, save_prob=save_prob_map)


# Save results
np.savez('results/load_' + load_name, 
         predicted_positions=predicted_positions, actual_positions=actual_positions,
         position_mse=pos_loss_test, triplet_test=triplet_loss_test, power_mse=feature_loss_test, 
         probabilities_map = probabilities_map, \
         tw_test = tw_test, ct_test = ct_test, ks_test = ks_test, rd_test = rd_test, perc_95_test=perc_95_test, 
         )

print(f"Testing Completed")
