# -*- coding: utf-8 -*-
'''
Training of the CSI fingerprinting baseline of Sec. IV-A3, which minimizes the
supervised loss of (26) with labeled data. The NN uses a grid of points from the
floor plan and estimates the probability that each point in the grid is the true
position, as in the proposed approach.
'''
from lib.simulation_parameters import *

# Define Network and optimizer
model = MLPSoftmaxOutputDropout(layers, n_grid, dropout_rate).to(device)

criterion = TripletLoss(margin=Mt)
mse_loss_fn  = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Create file_name and print it to debug
save_name = 'train_supervised_' + str(np.round(dropout_rate, 2)) + '_n_' + str(args.neurons) + '_l_' + str(args.layers)
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
                              test_epoch_list, test_loader)

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
           columns=C, processing=processing, save_prob=save_prob_map)

#Testing in smaller areas of the floor plan (including the area removed from the training dataset)
pred_pos_list, actual_pos_list, prob_map_list = [], [], []
pos_loss_list, triplet_loss_list, feature_loss_list = [], [], []
tw_list, ct_list, ks_list, rd_list, perc_95_list = [], [], [], [], []
n_point_area = []   #We need the number of points in each testing area to keep track when loading the results
for i in range(len(list_dataloaders)):
    predicted_pos_temp, actual_pos_temp, probabilities_map_temp, \
        pos_loss_temp, triplet_loss_temp, feature_loss_temp, \
        tw_temp, ct_temp, ks_temp, rd_temp, perc_95_temp = \
        testNN(list_dataloaders[i], model, Tc, noise_var, criterion, n_APs, grid_positions, grid_features,
            columns=C, processing=processing, save_prob=save_prob_map)
    pred_pos_list.append(predicted_pos_temp)
    actual_pos_list.append(actual_pos_temp)
    prob_map_list.append(probabilities_map_temp)
    pos_loss_list.append(pos_loss_temp)
    triplet_loss_list.append(triplet_loss_temp)
    feature_loss_list.append(feature_loss_temp)
    tw_list.append(tw_temp); ct_list.append(ct_temp)
    ks_list.append(ks_temp); rd_list.append(rd_temp)
    perc_95_list.append(perc_95_temp)
    n_point_area.append(predicted_pos_temp.shape[0])

#Stack variables to save vectors with consistent sizes (we can't save a list of arrays with different sizes)
pred_pos_list = np.vstack(pred_pos_list)
actual_pos_list = np.vstack(actual_pos_list)
prob_map_list = np.vstack(prob_map_list)

# Save results
np.savez('results/' + save_name, 
         loss=loss_it, 
         predicted_positions=predicted_positions, actual_positions=actual_positions,
         dist_loss_eval=dist_loss_eval, triplet_loss_eval=triplet_loss_eval,
         test_epoch_list=test_epoch_list,
         position_mse=pos_loss_test, triplet_test=triplet_loss_test, power_mse=feature_loss_test, 
         probabilities_map = probabilities_map, \
         tw_test = tw_test, ct_test = ct_test, ks_test = ks_test, rd_test = rd_test, perc_95_test=perc_95_test, \
         pred_pos_list=pred_pos_list, actual_pos_list=actual_pos_list, \
         prob_map_list = prob_map_list, pos_loss_list=pos_loss_list, \
         triplet_loss_list=triplet_loss_list, feature_loss_list=feature_loss_list, \
         tw_list=tw_list, ct_list=ct_list, ks_list=ks_list, rd_list=rd_list, \
         perc_95_list=perc_95_list, n_point_area=n_point_area)

print(f"Testing Completed. Predicted positions shape: {predicted_positions.shape}")
print(f"Mean Position MSE: {pos_loss_test:.6f}")
print(f"Percentile 95 Position MSE: {perc_95_test:.6f}")
print(f"Mean Triplet MSE: {triplet_loss_test:.6f}")
print(f"Mean Power MSE: {feature_loss_test:.2e}")
print(f"Trustworthiness: {tw_test:.6f}")
print(f"Continuity: {ct_test:.6f}")
print(f"Kruskal stress: {ks_test:.6f}")
print(f"Rasjki distance: {rd_test:.6f}")
