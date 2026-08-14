# -*- coding: utf-8 -*-
'''
Inference with the DT-based Power fingerprinting baseline trained with
train_supervised_power.py, which uses the received power per AP as input of the
NN. The testing dataset is the one defined in simulation_parameters.py.
'''
from lib.simulation_parameters import *

# Define Network and optimizer
layers[0] = 8
model = MLPSoftmaxOutputDropout(layers, n_grid, dropout_rate).to(device)

criterion = TripletLoss(margin=Mt)

#Create file_name and print it to debug
save_name = 'train_supervised_power_' + str(np.round(dropout_rate, 2)) + '_n_' + str(args.neurons) + '_l_' + str(args.layers)
save_name += '_Mt_' + str(np.round(Mt,1))
save_name += '_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs)
save_name += '_snr_' + str(snr_db) + '_dB'
save_name += '_realization_' + str(args.seed)
print(save_name, flush=True)

#Load trained model
checkpoint = torch.load('models/' + save_name)
model.load_state_dict(checkpoint['model'])
model.to(device)

# Testing in the area that has not been removed from training in simulation_parameters.py
grid_features = torch.clone(received_powers)
predicted_positions, actual_positions, probabilities_map, \
    pos_loss_test, triplet_loss_test, feature_loss_test, \
        tw_test, ct_test, ks_test, rd_test, perc_95_test = \
    testNN(test_loader, model, Tc, noise_var, criterion, n_APs, grid_positions, grid_features,
           columns=C, processing=processing, save_prob=save_prob_map, input_nn='power')

# Save results
np.savez('results/test_' + save_name, 
         predicted_positions=predicted_positions, actual_positions=actual_positions,
         position_mse=pos_loss_test, triplet_test=triplet_loss_test, power_mse=feature_loss_test, 
         probabilities_map = probabilities_map, \
         tw_test = tw_test, ct_test = ct_test, ks_test = ks_test, rd_test = rd_test, perc_95_test=perc_95_test)

print(f"Testing Completed. Predicted positions shape: {predicted_positions.shape}")