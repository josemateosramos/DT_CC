# -*- coding: utf-8 -*-
'''
File to train a model only with CC loss and
then use all training samples as ground-truth
positions to find the best affine transformation.
This is the affine transformation baseline of Sec. IV-A2, i.e., the loss of (15)
with lambda_CC = 1 and lambda_DT = 0, followed by the least-squares problem of (25).
'''
from lib.simulation_parameters import *

# Define Network and optimizer
model = MLPSoftmaxOutputDropout(layers, n_grid, dropout_rate).to(device)

criterion = TripletLoss(margin=Mt)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Create file_name and print it to debug
save_name = 'train_cc_' + str(np.round(dropout_rate, 2)) + '_n_' + str(args.neurons) + '_l_' + str(args.layers)
save_name += '_Mt_' + str(np.round(Mt,1))
save_name += '_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs)
save_name += '_snr_' + str(snr_db) + '_dB'
save_name += '_triplet_' + str(lambda_triplet) 
save_name += '_realization_' + str(args.seed)
print(save_name)

#Train NN
grid_features = torch.clone(received_powers)    #This doesn't matter
loss_triplet_it, loss_feature_it, mde_it, \
    pos_loss_eval, triplet_loss_eval, feature_loss_eval, \
         mde_train_eval = \
            trainNN(num_epochs, train_loader, Tc, noise_var, model, optimizer, criterion,
            n_APs, grid_positions, grid_features, lambda_triplet, 0.0,
            test_epoch_list, test_loader, columns=C, processing=processing)

print("Training Completed.")

#Save trained model
torch.save({
    'model': model.state_dict(),
}, 'models/' + save_name)

#Compute affine transformation using the training data to solve the LS problem
opt_matrix = computeAffineTransform(train_loader, model, noise_var, grid_positions, columns=C, device=device)

# Testing outside area specified in simulation_parameters.py
predicted_positions, actual_positions, pos_loss_test, tw_test, ct_test, ks_test, rd_test, perc_95_test = \
    testAffineTransform(opt_matrix, test_loader, model, noise_var, grid_positions, columns=C, device=device)

# Save results
np.savez('results/' + save_name, 
         loss_triplet=loss_triplet_it, mde_it=mde_it, 
         pos_loss_eval=pos_loss_eval, triplet_loss_eval=triplet_loss_eval,
         pow_loss_eval=feature_loss_eval, mde_train_eval = mde_train_eval,
         predicted_positions=predicted_positions, actual_positions=actual_positions,
         test_epoch_list=test_epoch_list,
         position_mse=pos_loss_test,  \
         tw_test = tw_test, ct_test = ct_test, ks_test = ks_test, rd_test = rd_test, perc_95_test=perc_95_test, 
         opt_matrix=opt_matrix.cpu().detach().numpy())

print(f"Testing Completed")
