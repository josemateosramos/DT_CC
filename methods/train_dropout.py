# -*- coding: utf-8 -*-
'''
Training of the proposed DT-aided CC approach following Algorithm 1, i.e., with
the loss of (15) that combines the timestamp-based triplet loss of (3) and the
DT-aided loss of (5). The output of the NN is a probability vector indicating how
likely a position in a pre-defined grid is the true UE position. The large-scale
feature used in the DT-aided loss is selected with the --processing argument.
'''
from lib.simulation_parameters import *

# Define Network and optimizer
model = MLPSoftmaxOutputDropout(layers, n_grid, dropout_rate).to(device)

criterion = TripletLoss(margin=Mt)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Create file_name and print it to debug
save_name = 'train_dropout_' + str(np.round(dropout_rate, 2)) + '_n_' + str(args.neurons) + '_l_' + str(args.layers)
save_name += '_Mt_' + str(np.round(Mt,1))
save_name += '_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs)
save_name += '_snr_' + str(snr_db) + '_dB'
save_name += '_triplet_' + str(lambda_triplet) 
save_name += '_feature_' + '{:.0e}'.format(lambda_feature)   
save_name += '_loss_' + processing
save_name += '_realization_' + str(args.seed)
print(save_name)

#Train NN
#Select used features depending on the chosen input
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

loss_triplet_it, loss_feature_it, mde_it, \
    pos_loss_eval, triplet_loss_eval, feature_loss_eval, \
         mde_train_eval = \
            trainNN(num_epochs, train_loader, Tc, noise_var, model, optimizer, criterion,
            n_APs, grid_positions, grid_features, lambda_triplet, lambda_feature,
            test_epoch_list, test_loader, columns=C, processing=processing)

print("Training Completed.")

#Save trained model
torch.save({
    'model': model.state_dict(),
}, 'models/' + save_name)

# Testing with the training data (check overfitting)
pred_pos_train, real_pos_train, _, \
    pos_loss_test_train, _, _, \
        _, _, _, _, perc_95_test_train = \
    testNN(train_loader, model, Tc, noise_var, criterion, n_APs, grid_positions, grid_features,
           columns=C, processing=processing, save_prob=save_prob_map)

# Testing outside area specified in simulation_parameters.py
predicted_positions, actual_positions, probabilities_map, \
    pos_loss_test, triplet_loss_test, feature_loss_test, \
        tw_test, ct_test, ks_test, rd_test, perc_95_test = \
    testNN(test_loader, model, Tc, noise_var, criterion, n_APs, grid_positions, grid_features,
           columns=C, processing=processing, save_prob=save_prob_map)

'''
The block commented out below tests the performance of the trained model separately
in each of the areas of the floor plan defined by coord_areas_min/coord_areas_max in
simulation_parameters.py (including the area removed from the training dataset).
To obtain that behaviour, uncomment the lines below AND the corresponding lines at
the end of the np.savez call, so that the per-area results are stored as well.
Note that the last active argument of np.savez (perc_95_test=perc_95_test) closes
the call, so its closing parenthesis has to be replaced by a comma when the
per-area lines are uncommented.
'''
# pred_pos_list, actual_pos_list, prob_map_list = [], [], []
# pos_loss_list, triplet_loss_list, feature_loss_list = [], [], []
# tw_list, ct_list, ks_list, rd_list, perc_95_list = [], [], [], [], []
# n_point_area = []   #We need the number of points in each testing area to keep track when loading the results
# for i in range(len(list_dataloaders)):
#     predicted_pos_temp, actual_pos_temp, probabilities_map_temp, \
#         pos_loss_temp, triplet_loss_temp, feature_loss_temp, \
#         tw_temp, ct_temp, ks_temp, rd_temp, perc_95_temp = \
#         testNN(list_dataloaders[i], model, Tc, noise_var, criterion, n_APs, grid_positions, grid_features,
#             columns=C, processing=processing, save_prob=save_prob_map)
#     pred_pos_list.append(predicted_pos_temp)
#     actual_pos_list.append(actual_pos_temp)
#     prob_map_list.append(probabilities_map_temp)
#     pos_loss_list.append(pos_loss_temp)
#     triplet_loss_list.append(triplet_loss_temp)
#     feature_loss_list.append(feature_loss_temp)
#     tw_list.append(tw_temp); ct_list.append(ct_temp)
#     ks_list.append(ks_temp); rd_list.append(rd_temp)
#     perc_95_list.append(perc_95_temp)
#     n_point_area.append(predicted_pos_temp.shape[0])

# #Stack variables to save vectors with consistent sizes (we can't save a list of arrays with different sizes)
# pred_pos_list = np.vstack(pred_pos_list)
# actual_pos_list = np.vstack(actual_pos_list)
# prob_map_list = np.vstack(prob_map_list)

# Save results
np.savez('results/' + save_name, 
         loss_triplet=loss_triplet_it, loss_power=loss_feature_it, mde_it=mde_it, 
         pos_loss_eval=pos_loss_eval, triplet_loss_eval=triplet_loss_eval,
         pow_loss_eval=feature_loss_eval, mde_train_eval = mde_train_eval,
         pred_pos_train = pred_pos_train, real_pos_train = real_pos_train,
         pos_loss_test_train = pos_loss_test_train, perc_95_test_train = perc_95_test_train,
         predicted_positions=predicted_positions, actual_positions=actual_positions,
         test_epoch_list=test_epoch_list,
         position_mse=pos_loss_test, triplet_test=triplet_loss_test, power_mse=feature_loss_test, 
         probabilities_map = probabilities_map, \
         tw_test = tw_test, ct_test = ct_test, ks_test = ks_test, rd_test = rd_test, perc_95_test=perc_95_test)
        #  pred_pos_list=pred_pos_list, actual_pos_list=actual_pos_list, 
        #  prob_map_list = prob_map_list, pos_loss_list=pos_loss_list, 
        #  triplet_loss_list=triplet_loss_list, feature_loss_list=feature_loss_list, 
        #  tw_list=tw_list, ct_list=ct_list, ks_list=ks_list, rd_list=rd_list,
        #  perc_95_list=perc_95_list, n_point_area=n_point_area)

print(f"Testing Completed")
