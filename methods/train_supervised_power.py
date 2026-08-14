# -*- coding: utf-8 -*-
'''
Training of the DT-based Power fingerprinting baseline of Sec. IV-D1, which
minimizes the supervised loss of (26) using the received power per AP as input
of the NN instead of the CSI feature of (1). The first layer is therefore
resized to the number of APs.
'''
from lib.simulation_parameters import *

# Define Network and optimizer
layers[0] = 8
model = MLPSoftmaxOutputDropout(layers, n_grid, dropout_rate).to(device)

criterion = TripletLoss(margin=Mt)
mse_loss_fn  = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Create file_name and print it to debug
save_name = 'train_supervised_power_' + str(np.round(dropout_rate, 2)) + '_n_' + str(args.neurons) + '_l_' + str(args.layers)
save_name += '_Mt_' + str(np.round(Mt,1))
save_name += '_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs)
save_name += '_snr_' + str(snr_db) + '_dB'
save_name += '_realization_' + str(args.seed)
print(save_name)

#Train NN
#Select used features depending on the chosen input (only used for validation during training)
if processing == 'power':
    grid_features = torch.clone(received_powers)
loss_it, dist_loss_eval, triplet_loss_eval = \
            trainNNSupervised(num_epochs, train_loader, Tc, noise_var, model, optimizer, mse_loss_fn,
                              criterion, n_APs, grid_positions, grid_features,
                              test_epoch_list, test_loader, input_nn='power')

print("Training Completed.")

#Save trained model
torch.save({
    'model': model.state_dict(),
}, 'models/' + save_name)

# Testing in the area that has not been removed from training in simulation_parameters.py
predicted_positions, actual_positions, probabilities_map, \
    pos_loss_test, triplet_loss_test, feature_loss_test, \
        tw_test, ct_test, ks_test, rd_test, perc_95_test = \
    testNN(test_loader, model, Tc, noise_var, criterion, n_APs, grid_positions, grid_features,
           columns=C, processing=processing, save_prob=save_prob_map, input_nn='power')

# Save results
np.savez('results/' + save_name, 
         loss=loss_it, 
         predicted_positions=predicted_positions, actual_positions=actual_positions,
         dist_loss_eval=dist_loss_eval, triplet_loss_eval=triplet_loss_eval,
         test_epoch_list=test_epoch_list,
         position_mse=pos_loss_test, triplet_test=triplet_loss_test, power_mse=feature_loss_test, 
         probabilities_map = probabilities_map, \
         tw_test = tw_test, ct_test = ct_test, ks_test = ks_test, rd_test = rd_test, perc_95_test=perc_95_test
         )

print(f"Testing Completed. Predicted positions shape: {predicted_positions.shape}")
print(f"Mean Position MSE: {pos_loss_test:.6f}")
print(f"Percentile 95 Position MSE: {perc_95_test:.6f}")
print(f"Mean Triplet MSE: {triplet_loss_test:.6f}")
print(f"Mean Power MSE: {feature_loss_test:.2e}")
print(f"Trustworthiness: {tw_test:.6f}")
print(f"Continuity: {ct_test:.6f}")
print(f"Kruskal stress: {ks_test:.6f}")
print(f"Rasjki distance: {rd_test:.6f}")
