# -*- coding: utf-8 -*-
'''
Training of the semi-supervised baseline of Sec. IV-C2, which combines the CC
triplet loss of (3) and the supervised loss of (26) following (27). The fraction
of labeled samples used in the supervised loss is set by sl_ratio, which allows
to reproduce the fingerprinting curve of Fig. 4.
'''
from lib.simulation_parameters import *

# Define Network and optimizer
model = MLPSoftmaxOutputDropout(layers, n_grid, dropout_rate).to(device)

criterion_triplet = TripletLoss(margin=Mt)
mse_loss_fn  = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#SL ratio to decide how many samples to use for SL
sl_ratio = 0.8

#We need not to shuffle the training dataset so that SL takes always the same samples
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

#Create file_name and print it to debug
save_name = 'train_dropout_sl_' + str(np.round(dropout_rate, 2)) + '_n_' + str(args.neurons) + '_l_' + str(args.layers)
save_name += '_Mt_' + str(np.round(Mt,1))
save_name += '_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs)
save_name += '_snr_' + str(snr_db) + '_dB'
save_name += '_triplet_' + str(lambda_triplet) 
save_name += '_sl_' + '{:.0e}'.format(lambda_feature) 
save_name += '_sl_ratio_' + str(sl_ratio)
save_name += '_realization_' + str(args.seed)
print(save_name, flush=True)

#Train NN
#Select used features depending on the chosen input
if processing == 'power':
    grid_features = torch.clone(received_powers)

loss_triplet_it, loss_sl_it, mde_it, mean_pos_mse_eval, mean_triplet_eval, mde_train_eval, train_sl_pos = \
    trainNNCCAndSupervised(num_epochs, train_loader, Tc, noise_var, model, optimizer, criterion_triplet, mse_loss_fn,
            n_APs, grid_positions, grid_features, lambda_triplet, lambda_feature, #Reuse the lambda_feature input for SL here
            test_epoch_list, test_loader, columns=C, sl_ratio=sl_ratio, processing=processing,
            device=device)

print("Training Completed.")

#Save trained model
torch.save({
    'model': model.state_dict(),
}, 'models/' + save_name)

# Testing with the training data (check overfitting)
pred_pos_train, real_pos_train, _, \
    pos_loss_test_train, _, _, \
        _, _, _, _, perc_95_test_train = \
    testNN(train_loader, model, Tc, noise_var, criterion_triplet, n_APs, grid_positions, grid_features,
           columns=C, processing=processing, save_prob=save_prob_map)

# Testing outside area specified in simulation_parameters.py
predicted_positions, actual_positions, probabilities_map, \
    pos_loss_test, triplet_loss_test, feature_loss_test, \
        tw_test, ct_test, ks_test, rd_test, perc_95_test = \
    testNN(test_loader, model, Tc, noise_var, criterion_triplet, n_APs, grid_positions, grid_features,
           columns=C, processing=processing, save_prob=save_prob_map)

# Save results
np.savez('results/' + save_name, 
         loss_triplet=loss_triplet_it, loss_sl=loss_sl_it, mde_it=mde_it, 
         pos_loss_eval=mean_pos_mse_eval, triplet_loss_eval=mean_triplet_eval,
         mde_train_eval = mde_train_eval,
         pred_pos_train = pred_pos_train, real_pos_train = real_pos_train,
         pos_loss_test_train = pos_loss_test_train, perc_95_test_train = perc_95_test_train,
         predicted_positions=predicted_positions, actual_positions=actual_positions,
         test_epoch_list=test_epoch_list,
         position_mse=pos_loss_test, triplet_test=triplet_loss_test, power_mse=feature_loss_test, 
         probabilities_map = probabilities_map, \
         tw_test = tw_test, ct_test = ct_test, ks_test = ks_test, rd_test = rd_test, perc_95_test=perc_95_test, \
            train_sl_pos = train_sl_pos.cpu().detach().numpy())

print(f"Testing Completed")
