# -*- coding: utf-8 -*-
'''
Inference with the CSI fingerprinting baseline trained with train_supervised.py
on a testing dataset that differs from the training one, following the
distribution shift tests of Sec. IV-D2.
'''
from lib.simulation_parameters import *

# Define Network and optimizer
model = MLPSoftmaxOutputDropout(layers, n_grid, dropout_rate).to(device)

criterion = TripletLoss(margin=Mt)

#Create file_name and print it to debug
save_name = 'train_supervised_' + str(np.round(dropout_rate, 2)) + '_n_' + str(args.neurons) + '_l_' + str(args.layers)
save_name += '_Mt_' + str(np.round(Mt,1))
save_name += '_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs)
save_name += '_snr_' + str(snr_db) + '_dB'
save_name += '_realization_' + str(args.seed)
print(save_name, flush=True)

#Load trained model
checkpoint = torch.load('models/' + save_name)
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


# Testing in the area that has not been removed from training in simulation_parameters.py
grid_features = torch.clone(received_powers)
predicted_positions, actual_positions, probabilities_map, \
    pos_loss_test, triplet_loss_test, feature_loss_test, \
        tw_test, ct_test, ks_test, rd_test, perc_95_test = \
    testNN(test_loader_new, model, Tc, noise_var, criterion, n_APs, grid_positions, grid_features,
           columns=C, processing=processing, save_prob=save_prob_map)

# Save results
np.savez('results/' + save_name, 
         predicted_positions=predicted_positions, actual_positions=actual_positions,
         position_mse=pos_loss_test, triplet_test=triplet_loss_test, power_mse=feature_loss_test, 
         probabilities_map = probabilities_map, \
         tw_test = tw_test, ct_test = ct_test, ks_test = ks_test, rd_test = rd_test, perc_95_test=perc_95_test)

print(f"Testing Completed. Predicted positions shape: {predicted_positions.shape}")