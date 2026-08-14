'''
Library with the dataset, loss, training, testing and evaluation-metric
functions of the DT-aided channel charting (CC) framework.
The equation numbers in the docstrings refer to the paper.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.fft as fft
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import trustworthiness

'''Classes'''
# Custom Dataset Class
class ChannelChartingDataset(Dataset):
    '''
    Dataset that holds the raw CSI matrices H^(n), the true UE positions
    x^(n) and the timestamps t^(n) of the collected measurements.
    The positions are only used for evaluation, never for training.
    Inputs
    ----------
    raw_csi: complex tensor of shape (N, A*K, S)
        Aggregated CSI matrices of all APs.
    positions: float tensor of shape (N, D_x)
        True UE positions.
    timestamps: float tensor of shape (N, 1)
        Timestamp of each CSI measurement.
    '''
    def __init__(self, raw_csi, positions, timestamps):
        self.raw_csi = raw_csi
        self.positions = positions
        self.timestamps = timestamps

    def __len__(self):
        return len(self.raw_csi)

    def __getitem__(self, idx):
        return self.raw_csi[idx], self.positions[idx], self.timestamps[idx]

class TripletLoss(nn.Module):
    '''
    Timestamp-based triplet loss of the CC loss in (3), where margin is
    the distance threshold Mt that enforces the anchor to be at least Mt
    closer to the positive sample than to the negative one.
    '''
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative):
        '''
        Inputs are the estimated positions of the anchor, positive and
        negative samples, each of shape (batch_size, D_x).
        '''
        d_ap = torch.norm(anchor - positive, dim=1)  # Distance between anchor & positive
        d_an = torch.norm(anchor - negative, dim=1)  # Distance between anchor & negative
        loss = self.relu(d_ap - d_an + self.margin)  # Triplet loss formula
        return loss.mean()
    
''' 
===============
Functions
===============
'''
def computeIndicesInsideArea(UE_positions, coord_min, coord_max):
    '''
    Function that computes the indices of the positions that belong to the
    rectangle defined by the two given coordinates.
    Inputs
    ----------
    UE_positions: float tensor of shape (N, 2)
        Positions to check.
    coord_min, coord_max: list of 2 floats
        Bottom-left and top-right corners of the rectangle.
    '''
    #Compute a mask indicating which indices belong to the rectangle
    mask_inside = (UE_positions[:, 0] >= coord_min[0]) & (UE_positions[:, 0] <= coord_max[0]) & \
                    (UE_positions[:, 1] >= coord_min[1]) & (UE_positions[:, 1] <= coord_max[1])

    return mask_inside.nonzero(as_tuple=True)[0]



def computeTestAreas(dataset, test_dataset_indices, UE_pos, coor_areas_min, coor_areas_max):
    '''
    Function that computes the testing areas for different coordinates
    Inputs
    ----------
    dataset: ChannelChartingDataset with all the collected samples.
    test_dataset_indices: indices of dataset that belong to the testing set.
    UE_pos: float tensor of shape (N, 2) with the true positions of dataset.
    coor_areas_min, coor_areas_max: lists with the bottom-left and top-right
        corners of each rectangular area.
    Returns one DataLoader per non-empty area, each one containing the
    testing samples that lie inside that area.
    '''
    list_dataloaders = []
    for i in range(len(coor_areas_min)):
        indices_in = computeIndicesInsideArea(UE_pos[test_dataset_indices], coor_areas_min[i], coor_areas_max[i])
        if len(indices_in) > 0:
            # Map back to global indices
            indices_area = test_dataset_indices[indices_in]
            test_set = torch.utils.data.Subset(dataset, indices_area)
            test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
            list_dataloaders.append(test_dataloader)
    return list_dataloaders

def IFFTTruncated(CSI, C):
    '''
    Function that computes the truncated delay-domain CSI [H F_S^H]_{:,1:C}
    of (1), i.e., the first C delay taps of the CSI matrix.
    '''
    CSI_ifft = fft.ifft(CSI, dim=2, norm='ortho')  # Take IFFT along subcarrier axis
    CSI_truncated = CSI_ifft[:, :, :C]  # Keep first C columns
    return CSI_truncated

def preprocess_csi(CSI, C):
    """Computes the CSI feature vector f = |h|/||h|| of (1), used as input of
    the positioning function. Applies IFFT along rows, keeps first C columns,
    vectorizes and normalizes."""
    CSI_truncated = IFFTTruncated(CSI, C)
    CSI_vectorized = CSI_truncated.reshape(CSI.shape[0], -1)  # Flatten to vector
    norms = torch.norm(CSI_vectorized, dim=1, keepdim=True)
    return torch.abs(CSI_vectorized) / norms  # Normalize by its norm

def sample_triplets(batch, Tc):
    """Vectorized function to sample triplets efficiently while handling invalid cases.
    Builds the set of triplets of (2) taking every sample of the batch as an
    anchor, and drawing uniformly at random one close (positive) sample with
    |t-t'| < Tc and one far (negative) sample with |t-t'| >= Tc. We set Tf = Tc,
    so that every sample of the batch is either close or far to the anchor.
    Returns the anchors, positives, negatives and a boolean mask with the
    anchors that had both a positive and a negative sample available, or
    None if no valid triplet exists in the batch."""
    csi, timestamps = batch  # Extract tensors
    batch_size = csi.shape[0]

    # Compute pairwise absolute timestamp differences (B x B)
    time_diffs = torch.abs(timestamps - timestamps.transpose(0, 1))

    # Create masks for close (Tc) and far samples
    close_mask = (time_diffs < Tc) & (~torch.eye(batch_size, dtype=torch.bool, device=timestamps.device))
    far_mask = time_diffs >= Tc

    # Identify valid rows (those with at least one valid close and far sample)
    valid_close = close_mask.any(dim=1)  # Boolean mask for rows with valid positive samples
    valid_far = far_mask.any(dim=1)  # Boolean mask for rows with valid negative samples
    valid_rows = valid_close & valid_far  # Only keep rows with both positive and negative samples

    if not valid_rows.any():  # If no valid triplets exist, return None
        return None

    # Filter tensors based on valid rows
    csi_valid = csi[valid_rows]
    close_mask_valid = close_mask[valid_rows]
    far_mask_valid = far_mask[valid_rows]

    # Compute normalized probabilities only for valid rows
    close_probs = close_mask_valid.float()
    close_probs /= close_probs.sum(dim=1, keepdim=True)  # Normalize

    far_probs = far_mask_valid.float()
    far_probs /= far_probs.sum(dim=1, keepdim=True)  # Normalize

    # Sample indices for positives and negatives
    positive_indices = torch.multinomial(close_probs, num_samples=1).squeeze(1)  # (valid_count,)
    negative_indices = torch.multinomial(far_probs, num_samples=1).squeeze(1)  # (valid_count,)

    # Extract triplet samples
    anchor_samples = csi_valid
    positive_samples = csi[positive_indices]
    negative_samples = csi[negative_indices]

    return anchor_samples, positive_samples, negative_samples, valid_rows

def addNoiseCSINonZero(csi, noise_var, device='cuda'):
    '''
    Function that adds gaussian noise to a given CSI matrix in the non-zero elements.
    The zero elements correspond to APs that do not receive the signal of the UE
    and are kept noiseless, so that they remain identified as non-received.
    '''
    device = csi.device
    noise_realization = torch.sqrt(noise_var/2.0)*(torch.randn(csi.shape, dtype=torch.cfloat, device=device) +
                                                1j*torch.randn(csi.shape, dtype=torch.cfloat, device=device))
    csi_noise = csi + noise_realization * (csi != 0)
    return csi_noise

######################
# Training functions #
######################
def trainNN(num_epochs, train_loader, Tc, noise_var, model, optimizer, criterion,
            n_APs, grid_positions, grid_features, lambda_triplet, lambda_feature,
            test_epoch_list, val_loader, columns=13, processing='power',
            device='cuda'):
    '''
    Function to perform training of the NN following Algorithm 1, i.e., it
    minimizes the loss of (15) that combines the CC triplet loss of (3) and
    the DT-aided loss of (5).
    - noise_var: variance of the noise to add to the CSI
    - grid_positions: positions of the grid points considered in the DT. Float tensor of shape (n_grid,2)
    - grid_features: features computed for the grid of points of the digital twin (power, angle_profile, etc.).
    Tensor of shape (n_grid, n_features). n_features depends on the considered features.
    - lambda_triplet, lambda_feature: weights lambda_CC and lambda_DT of (15)
    - test_epoch_list: epochs at which the model is evaluated on val_loader
    - processing: large-scale feature h(.) of Sec. III-C
        + 'power': pathloss from the DT and estimated power from the CSI
        + 'angle': angle-power profile from the CSI for the DT and the measurements
        + 'delay': delay-power profile from the CSI for the DT and the measurements
        + 'tdp': truncated delay profile (TDP), i.e., the preprocessed CSI used as input of the NN
        + 'cov_abs_fft': abs of H@H.transpose().conj(), where H is the beamspace CSI matrix
    The feature loss applies an exponential to the cosine similarity.
    Returns the per-epoch triplet loss, feature loss and MDE of the training
    set, together with the metrics evaluated at the epochs in test_epoch_list.
    '''
    #List to save training results
    loss_triplet_it, loss_feature_it, mde_it = [], [], []
    #List to save testing results
    mean_pos_mse_eval, mean_triplet_eval, mean_feature_mse_eval = [], [], []
    #List to save testing results with the training set (to check bias)
    mde_train_eval = []

    for epoch in range(num_epochs):
        model.train()               #This is done here instead of outside the loop because we can test within the training loop
        total_loss_triplet = 0.0
        total_loss_feature = 0.0
        total_loss_mde     = 0.0

        for (i,batch) in enumerate(train_loader):
            csi_raw, true_pos_train, timestamps = batch  # Extract both preprocessed and raw CSI

            #Add noise to the CSI matrix (those elements that are no zero)
            csi_raw = addNoiseCSINonZero(csi_raw, noise_var, device=device)
            #Preprocess the CSI to obtain the inputs to the NN
            csi_preprocessed = preprocess_csi(csi_raw, columns)

            #Extract data from csi_raw, it'll be useful to process the csi_raw later on
            #csi_raw has size (batch_size, n_APs*n_antennas, n_subcarriers)
            batch_size = csi_raw.shape[0]    
            n_antennas = int(csi_raw.shape[1]/n_APs)   
            n_subcarriers = csi_raw.shape[2]

            triplets = sample_triplets((csi_preprocessed, timestamps), Tc)  # Get valid triplets

            if (triplets is not None) or (lambda_triplet == 0):  # Skip if no valid triplets found
                anchors, positives, negatives, valid_rows = triplets

                optimizer.zero_grad()

                # Pass through the model
                probabilities_anchor = model(anchors)  # (batch_size, N)
                probabilities_positive = model(positives)
                probabilities_negative = model(negatives)

                # Compute mean estimated positions
                mean_pos_anchor = torch.sum(probabilities_anchor.unsqueeze(2) * grid_positions, dim=1)  # (batch_size, 2)
                mean_pos_positive = torch.sum(probabilities_positive.unsqueeze(2) * grid_positions, dim=1)
                mean_pos_negative = torch.sum(probabilities_negative.unsqueeze(2) * grid_positions, dim=1)

                # Compute Triplet Loss
                triplet_loss = criterion(mean_pos_anchor, mean_pos_positive, mean_pos_negative)

                # Compute a weighted average of the features from the grid points in the DT
                mean_feature_estimate_anchor = (probabilities_anchor.unsqueeze(1) @ grid_features).squeeze(1) # (batch_size, num_features)

                # Process raw CSI according to the selected processing method in the input
                if processing == 'power':
                    csi_truncated = IFFTTruncated(csi_raw, columns)
                    features_from_csi_anchor = torch.norm(csi_truncated.reshape(csi_raw.shape[0], n_APs, -1), dim=-1)**2  #(b_size, n_APs)
                elif processing == 'angle':
                    angle_prof_from_csi_anchor = torch.sum(torch.abs(fft.fft(csi_raw.reshape(batch_size,n_APs,n_antennas,n_subcarriers),dim=-2, norm='ortho'))**2, dim=-1)  #(b_size, n_APs, n_antennas)
                    features_from_csi_anchor = angle_prof_from_csi_anchor.reshape(batch_size,-1)    #Shape: (b_size, n_AP*n_antennas)
                elif processing == 'delay':
                    delay_prof_from_csi_anchor = torch.sum(torch.abs(fft.ifft(csi_raw.reshape(batch_size,n_APs,n_antennas,n_subcarriers),dim=-1, norm='ortho'))**2, dim=-2)  #(b_size, n_APs, n_subcarriers)
                    features_from_csi_anchor = delay_prof_from_csi_anchor.reshape(batch_size,-1)    #Shape: (b_size, n_AP*n_subcarriers)
                elif processing == 'tdp':
                    tdp_from_csi_anchor = csi_preprocessed #(b_size, n_APs*n_antennas*columns)
                    features_from_csi_anchor = tdp_from_csi_anchor.reshape(batch_size,-1)    #Shape: (b_size, n_APs*n_antennas*columns)
                elif processing == 'cov_abs_fft':
                    csi_raw_rsh = csi_raw.reshape(batch_size, n_APs, n_antennas, n_subcarriers)
                    csi_raw_rsh_fft = fft.fft(csi_raw_rsh, dim=-2, norm='ortho')
                    cov_per_AP = csi_raw_rsh_fft @ csi_raw_rsh_fft.transpose(-1,-2).conj()    #Shape: (b_size, n_APs, n_antennas, n_antennas)
                    features_from_csi_anchor = torch.abs(cov_per_AP.reshape(batch_size,-1))           #Shape: (b_size, n_APs*n_antennas**2)

                # Take only valid rows to compare it to anchors
                features_from_csi_anchor = features_from_csi_anchor[valid_rows]
                

                # Apply an exponential to the cosine similarity between the estimated and the measured features
                feature_loss_per_batch = torch.exp(-(torch.abs(mean_feature_estimate_anchor.unsqueeze(2).transpose(1,2).conj() @ features_from_csi_anchor.unsqueeze(2))**2).flatten()/
                                                   (torch.norm(mean_feature_estimate_anchor, dim=-1)**2 * torch.norm(features_from_csi_anchor, dim=-1)**2))

                feature_loss = feature_loss_per_batch.mean()

                # Compute Total Loss
                total_loss_batch = lambda_triplet * triplet_loss + lambda_feature * feature_loss
                total_loss_batch.backward()
                optimizer.step()

                with torch.no_grad():
                    # Compute mean distance error according to the paper 
                    true_pos_train = true_pos_train[valid_rows]
                    dist_temp = torch.linalg.vector_norm(mean_pos_anchor - true_pos_train, dim=-1)
                    dist_loss = dist_temp.mean()

                total_loss_triplet  += triplet_loss.item()
                total_loss_feature  += feature_loss.item()
                total_loss_mde      += dist_loss.item()
        
        if epoch in test_epoch_list:
            #Test performance in the testing set
            _, _, _, mean_pos_mse_temp, mean_triplet_loss_temp, mean_feature_mse_temp, _, _, _, _, _ = \
                testNN(val_loader, model, Tc, noise_var, criterion, 
                       n_APs, grid_positions, grid_features, columns, processing)
            #Test performance in the training set (to check bias)
            _, _, _, mde_train_temp, _, _, _, _, _, _, _ = \
                testNN(train_loader, model, Tc, noise_var, criterion,
                       n_APs, grid_positions, grid_features, columns, processing)
            mean_pos_mse_eval.append(mean_pos_mse_temp)
            mean_triplet_eval.append(mean_triplet_loss_temp)
            mean_feature_mse_eval.append(mean_feature_mse_temp)
            mde_train_eval.append(mde_train_temp)
            model.train()

        total_loss_triplet  /= (i+1)
        total_loss_feature  /= (i+1)
        total_loss_mde      /= (i+1)

        loss_triplet_it.append(total_loss_triplet)
        loss_feature_it.append(total_loss_feature)
        mde_it.append(total_loss_mde)

        if (epoch % 500) == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Triplet Loss: {total_loss_triplet:.2e}, \
                    Feature Loss: {total_loss_feature:.2e}", flush=True)

    return loss_triplet_it, loss_feature_it, mde_it, mean_pos_mse_eval, mean_triplet_eval, mean_feature_mse_eval, mde_train_eval


def trainNNSupervised(num_epochs, train_loader, Tc, noise_var, model, optimizer, pos_criterion,
                      triplet_criterion, n_APs, grid_positions, grid_features,
                      test_epoch_list, val_loader, input_nn='csi', processing='power', columns=13):
    '''
    Function to perform supervised training of the NN knowing the positions of the UE
    during training, i.e., the CSI fingerprinting baseline that minimizes (26).
    - input_nn: input of the positioning function
        + 'csi': CSI feature vector of (1)
        + 'power': received power per AP, used by the DT-based Power fingerprinting
    - processing, grid_features: only used to evaluate the DT-aided loss on val_loader
    Returns the per-epoch loss and the metrics evaluated at the epochs in test_epoch_list.
    '''
    #List to save training results
    loss_it = []
    #List to save testing results
    mean_dist_err_eval, mean_triplet_eval = [], []

    for epoch in range(num_epochs):
        model.train()               #This is done here instead of outside the loop because we can test within the training loop
        total_loss = 0.0

        for (i,batch) in enumerate(train_loader):
            #Forward pass
            csi_raw, true_pos, _ = batch  # Extract both preprocessed and raw CSI

            #Add noise to the CSI matrix
            csi_raw = addNoiseCSINonZero(csi_raw, noise_var)

            #Preprocess the CSI to obtain the inputs to the NN
            if input_nn=='csi':
                csi_preprocessed = preprocess_csi(csi_raw, columns)
            if input_nn=='power':   #We could also use the power from Wireless Insite, but this simplifies code
                csi_truncated = IFFTTruncated(csi_raw, columns)
                csi_preprocessed = torch.norm(csi_truncated.reshape(csi_raw.shape[0], n_APs, -1), dim=-1)**2  #(b_size, n_APs)

            probabilities = model(csi_preprocessed)     # (batch_size, N)
            # Compute mean estimated positions
            mean_pos = torch.sum(probabilities.unsqueeze(2) * grid_positions, dim=1)  # (batch_size, 2)
            
            # Compute Total Loss
            optimizer.zero_grad()
            total_loss_batch = pos_criterion(mean_pos, true_pos)
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
        
        if epoch in test_epoch_list:
            _, _, _, mean_dist_err_temp, mean_triplet_loss_temp, _, _, _, _, _, _ = \
                testNN(val_loader, model, Tc, noise_var, triplet_criterion, 
                       n_APs, grid_positions, grid_features, columns, processing, input_nn=input_nn)
            mean_dist_err_eval.append(mean_dist_err_temp)
            mean_triplet_eval.append(mean_triplet_loss_temp)
            model.train()

        total_loss /= (i+1)

        loss_it.append(total_loss)

        if (epoch % 1000) == 0:
            print(f"Epoch {epoch+1}/{num_epochs},loss: {total_loss:.2e}", flush=True)

    return loss_it, mean_dist_err_eval, mean_triplet_eval

def trainNNCCAndSupervised(num_epochs, train_loader, Tc, noise_var, model, optimizer, triplet_criterion, pos_criterion,
            n_APs, grid_positions, grid_features, lambda_triplet, lambda_sl,
            test_epoch_list, val_loader, columns=13, sl_ratio=1.0, processing='power',
            device='cuda'):
    '''
    Function to perform training of the NN combining the CC triplet loss of (3)
    with the supervised loss of (26), i.e., it minimizes the loss of (27) used to
    assess the effect of the amount of labeled data on fingerprinting (Fig. 4).
    - noise_var: variance of the noise to add to the CSI
    - grid_positions: positions of the grid points considered in the DT. Float tensor of shape (n_grid,2)
    - grid_features: features computed for the grid of points of the digital twin (power, angle_profile, etc.).
    Tensor of shape (n_grid, n_features). n_features depends on the considered features.
    - lambda_triplet, lambda_sl: weights lambda_CC and lambda_SL of (27)
    - processing: large-scale feature h(.) of Sec. III-C, only used for evaluation
        + 'power': pathloss from the DT and estimated power from the CSI
        + 'angle': angle-power profile from the CSI for the DT and the measurements
        + 'delay': delay-power profile from the CSI for the DT and the measurements
        + 'tdp': truncated delay profile (TDP), i.e., the preprocessed CSI used as input of the NN
        + 'cov_abs_fft': abs of H@H.transpose().conj(), where H is the beamspace CSI matrix
    - sl_ratio: ratio of sl samples to take w.r.t the total number of samples.
    Between 0.0 and 1.0 (better to avoid 0.0)
    Returns the per-epoch losses and evaluation metrics, together with the true
    positions of the labeled samples used in the supervised loss.
    '''
    #List to save training results
    loss_triplet_it, loss_sl_it, mde_it = [], [], []
    #List to save testing results
    mean_pos_mse_eval, mean_triplet_eval = [], []
    #List to save testing results with the training set (to check bias)
    mde_train_eval = []

    saved_train_sl_pos = torch.tensor([], device=device)
    
    for epoch in range(num_epochs):
        model.train()               #This is done here instead of outside the loop because we can test within the training loop
        total_loss_triplet = 0.0
        total_loss_sl      = 0.0
        total_loss_mde     = 0.0

        for (i,batch) in enumerate(train_loader):
            csi_raw, true_pos_train, timestamps = batch  # Extract both preprocessed and raw CSI

            #Add noise to the CSI matrix (those elements that are no zero)
            csi_raw = addNoiseCSINonZero(csi_raw, noise_var, device=device)
            #Preprocess the CSI to obtain the inputs to the NN
            csi_preprocessed = preprocess_csi(csi_raw, columns)

            #Select triplets for triplet loss
            triplets = sample_triplets((csi_preprocessed, timestamps), Tc)  # Get valid triplets

            if (triplets is not None) or (lambda_triplet == 0):  # Skip if no valid triplets found
                anchors, positives, negatives, valid_rows = triplets

                # Pass through the model
                probabilities_anchor = model(anchors)  # (batch_size, N)
                probabilities_positive = model(positives)
                probabilities_negative = model(negatives)

                # Compute mean estimated positions
                mean_pos_anchor = torch.sum(probabilities_anchor.unsqueeze(2) * grid_positions, dim=1)  # (batch_size, 2)
                mean_pos_positive = torch.sum(probabilities_positive.unsqueeze(2) * grid_positions, dim=1)
                mean_pos_negative = torch.sum(probabilities_negative.unsqueeze(2) * grid_positions, dim=1)

                # Compute Triplet Loss
                triplet_loss = triplet_criterion(mean_pos_anchor, mean_pos_positive, mean_pos_negative)

                # == Compute supervised learning loss == #
                if sl_ratio > 0.0:
                    #Compute the number of samples to consider in SL
                    batch_size = csi_raw.shape[0]
                    num_sl_samples = int(batch_size*sl_ratio)
                    csi_preprocessed_sl = csi_preprocessed[:num_sl_samples]
                    probabilities = model(csi_preprocessed_sl)     # (num_sl_samples, N)
                    mean_pos = torch.sum(probabilities.unsqueeze(2) * grid_positions, dim=1)  # (num_sl_samples, 2)
                    sl_loss = pos_criterion(mean_pos, true_pos_train[:num_sl_samples])
                else:
                    sl_loss = 0.0

                # Compute Total Loss
                optimizer.zero_grad()
                total_loss_batch = lambda_triplet * triplet_loss + lambda_sl * sl_loss
                total_loss_batch.backward()
                optimizer.step()

                with torch.no_grad():
                    # Compute mean distance error according to the paper 
                    true_pos_train = true_pos_train[valid_rows]
                    dist_temp = torch.linalg.vector_norm(mean_pos_anchor - true_pos_train, dim=-1)
                    dist_loss = dist_temp.mean()

                total_loss_triplet  += triplet_loss.item()
                total_loss_sl       += sl_loss.item()
                total_loss_mde      += dist_loss.item()

                #Save training position for SL loss in the first iteration
                if epoch == 0:
                    saved_train_sl_pos = torch.cat((saved_train_sl_pos, true_pos_train[:num_sl_samples]))
            
        if epoch in test_epoch_list:
            #Test performance in the testing set
            _, _, _, mean_pos_mse_temp, mean_triplet_loss_temp, _, _, _, _, _, _ = \
                testNN(val_loader, model, Tc, noise_var, triplet_criterion, 
                       n_APs, grid_positions, grid_features, columns, processing)
            #Test performance in the training set (to check bias)
            _, _, _, mde_train_temp, _, _, _, _, _, _, _ = \
                testNN(train_loader, model, Tc, noise_var, triplet_criterion, 
                       n_APs, grid_positions, grid_features, columns, processing)
            mean_pos_mse_eval.append(mean_pos_mse_temp)
            mean_triplet_eval.append(mean_triplet_loss_temp)
            mde_train_eval.append(mde_train_temp)
            model.train()

        total_loss_triplet  /= (i+1)
        total_loss_sl       /= (i+1)
        total_loss_mde      /= (i+1)

        loss_triplet_it.append(total_loss_triplet)
        loss_sl_it.append(total_loss_sl)
        mde_it.append(total_loss_mde)

        if (epoch % 500) == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Triplet Loss: {total_loss_triplet:.2e}, \
                    SL Loss: {total_loss_sl:.2e}", flush=True)

    return loss_triplet_it, loss_sl_it, mde_it, mean_pos_mse_eval, mean_triplet_eval, mde_train_eval, saved_train_sl_pos

#####################
# Testing functions #
#####################
def compute_KS(true_pos, est_pos):
    '''
    Function that computes Kruskal stress (KS) following (30). It measures the
    dissimilarity between the pairwise distances of the true and the estimated
    positions, and takes values in [0,1] with optimal value 0.
    Ack to S. Taner
    Inputs
    ----------
    true_pos: np.array of size (U, dimension=2 or 3)
        True positions of the UEs in the original space.
    est_pos: np.array of size (U, dimension=2 or 3)
        Positions of the UEs in the channel chart.
    '''
    d_true = euclidean_distances(true_pos, true_pos)  # distance between the true positions
    d_est = euclidean_distances(est_pos, est_pos)  # distance between the points in the latent space

    beta = np.sum(d_true * d_est) / np.linalg.norm(d_est, 'fro') ** 2
    ks = np.linalg.norm(d_true - beta * d_est, 'fro') \
            / np.linalg.norm(d_true, 'fro')
    return ks

def compute_RD(true_pos, est_pos, metric_param=None):
    '''
    Function that computes Rajski distance (RD) following (31)-(33), i.e., the
    difference between the mutual information and the joint entropy of the
    pairwise distances of the true and the estimated positions. RD takes values
    in [0,1] with optimal value 0.
    - metric_param: number of uniform bins used to quantize the distances (20 by default)
    Ack to S. Taner
    Inputs
    ----------
    true_pos: np.array of size (U, dimension=2 or 3)
        True positions of the UEs in the original space.
    est_pos: np.array of size (U, dimension=2 or 3)
        Positions of the UEs in the channel chart.
    '''
    if metric_param == None:
        num_bins = 20
    else:
        num_bins = metric_param

    d_true = euclidean_distances(true_pos, true_pos)  # distance between the true positions
    d_est = euclidean_distances(est_pos, est_pos)  # distance between the points in the latent space

    x = d_true.flatten()
    y = d_est.flatten()
    Px, _ = np.histogram(x, bins=num_bins)
    Px = Px / len(x)

    Py, _ = np.histogram(y, bins=num_bins)
    Py = Py / len(y)

    Pxy, _, _ = np.histogram2d(x, y, bins=num_bins)
    Pxy = Pxy / len(x)

    idcs = np.nonzero(Pxy)
    H = - np.sum(Pxy[idcs] * np.log2(Pxy[idcs]))
    I = np.sum(Pxy[idcs] * np.log2(Pxy[idcs] / (Px[idcs[0]] * Py[idcs[1]])))

    assert H != 0
    rd = 1 - I / H
    return rd    

def computeTWCT(true_pos, est_pos):
    '''
    Function that evaluates the trustworthiness (TW) and the continuity (CT) of a
    channel chart, following (28) and (29), respectively. Both metrics take values
    in [0,1] with optimal value 1 and are computed with J = 0.05*N neighbors.
    Ack to S. Taner
    Inputs
    ----------
    true_pos: np.array of size (U, dimension=2 or 3)
        True positions of the UEs in the original space.
    est_pos: np.array of size (U, dimension=2 or 3)
        Positions of the UEs in the channel chart.
    '''
    tw = trustworthiness(true_pos, est_pos, n_neighbors = int(0.05 * est_pos.shape[0]))    
    ct = trustworthiness(est_pos, true_pos, n_neighbors = int(0.05 * est_pos.shape[0]))

    return tw, ct


def testNN(test_loader, model, Tc, noise_var, criterion_triplet, n_APs, grid_positions, grid_features, columns=13,
           processing='power', save_prob=False, compute_feature_loss=True, input_nn='csi'):
    '''
    Function to test the performance of a NN and compute the evaluation metrics
    of Appendix A: TW (28), CT (29), KS (30), RD (31), MDE (34) and 95th PDE.
    save_prob: True to save the probability map, False otherwise
    compute_feature_loss: True to compute the loss corresponding to the feature, False only computes features related to position
    input_nn: 'csi' for the CSI feature of (1), 'power' for the received power per AP
    Returns the estimated and true positions (sorted by timestamp), the probability
    maps, the position, triplet and feature losses, and the evaluation metrics.
    '''

    # Testing Loop
    model.eval()
    predicted_positions, actual_positions = [], []
    test_timestamps = []            #Although I don't save this, this will be useful to order positions
    pos_losses, triplet_losses, feature_losses = [], [], []
    probabilities_map = []
    dist_list = []                  #List to save distances between true and estimated positions

    with torch.no_grad():
        for batch in test_loader:
            csi_raw, true_pos, test_time = batch

            #Add noise to the CSI matrix
            csi_raw = addNoiseCSINonZero(csi_raw, noise_var)
            #Preprocess the CSI to obtain the inputs to the NN
            csi = preprocess_csi(csi_raw, columns)

            batch_size = csi_raw.shape[0]
            n_antennas = int(csi_raw.shape[1]/n_APs)   
            n_subcarriers = csi_raw.shape[2]

            #First compute the metrics that do not require to sample form the triplet
            if input_nn=='csi':
                probabilities = model(csi)  # Get probability vector (batch_size, N)
            if input_nn=='power':
                csi_truncated = IFFTTruncated(csi_raw, columns)
                csi_preprocessed = torch.norm(csi_truncated.reshape(csi_raw.shape[0], n_APs, -1), dim=-1)**2  #(b_size, n_APs)
                probabilities = model(csi_preprocessed)

            # Compute mean estimated position
            mean_pos = torch.sum(probabilities.unsqueeze(2) * grid_positions, dim=1)  # (batch_size, 2)
            # Compute position loss according to the paper 
            dist_temp = torch.linalg.vector_norm(mean_pos - true_pos, dim=-1)
            dist_loss = dist_temp.mean()

            if compute_feature_loss:
                # Compute a weighted average of the features from the grid points in the DT
                mean_feature_estimate = (probabilities.unsqueeze(1) @ grid_features).squeeze(1) # (batch_size, num_features)

                # Process raw CSI according to the selected processing method in the input
                if processing == 'power':
                    # Compute power from raw CSI
                    csi_truncated = IFFTTruncated(csi_raw, columns)
                    features_from_csi = torch.norm(csi_truncated.reshape(csi_raw.shape[0], n_APs, -1), dim=-1)**2  #(b_size, A)
                elif processing == 'angle':
                    # Compute angle_profile from raw CSI
                    angle_prof_from_csi = torch.sum(torch.abs(fft.fft(csi_raw.reshape(batch_size,n_APs,n_antennas,n_subcarriers),dim=-2, norm='ortho'))**2, dim=-1)  #(b_size, 8,4)
                    features_from_csi = angle_prof_from_csi.reshape(batch_size,-1)    #Shape: (b_size, n_AP*n_antennas)
                elif processing == 'delay':
                    # Compute delay_profile from raw CSI
                    delay_prof_from_csi_anchor = torch.sum(torch.abs(fft.ifft(csi_raw.reshape(batch_size,n_APs,n_antennas,n_subcarriers),dim=-1, norm='ortho'))**2, dim=-2)  #(b_size, n_APs, n_subcarriers)
                    features_from_csi = delay_prof_from_csi_anchor.reshape(batch_size,-1)    #Shape: (b_size, n_AP*n_subcarriers)
                elif processing == 'tdp':
                    # Compute truncated delay profile from raw CSI
                    tdp_from_csi_anchor = torch.clone(csi)  #(b_size, n_APs, n_subcarriers)
                    features_from_csi = tdp_from_csi_anchor.reshape(batch_size,-1)    #Shape: (b_size, n_AP*n_subcarriers)
                elif processing == 'cov_abs_fft':
                    csi_raw_rsh = csi_raw.reshape(batch_size, n_APs, n_antennas, n_subcarriers)
                    csi_raw_rsh_fft = fft.fft(csi_raw_rsh, dim=-2, norm='ortho')
                    cov_per_AP = csi_raw_rsh_fft @ csi_raw_rsh_fft.transpose(-1,-2).conj()    #Shape: (b_size, n_APs, 4, 4)
                    features_from_csi = torch.abs(cov_per_AP.reshape(batch_size,-1))           #Shape: (b_size, n_APs*n_antennas**2)

                # Apply an exponential to the cosine similarity between the estimated and the measured features
                feature_loss_per_batch = torch.exp(-(torch.abs(mean_feature_estimate.unsqueeze(2).transpose(1,2).conj() @ features_from_csi.unsqueeze(2))**2).flatten()/
                                                   (torch.norm(mean_feature_estimate, dim=-1)**2 * torch.norm(features_from_csi, dim=-1)**2))

                feature_loss = feature_loss_per_batch.mean()
                feature_losses.append(feature_loss.item())

                #Compute triplet loss
                if input_nn=='csi':
                    triplets = sample_triplets((csi, test_time), Tc) 
                if input_nn=='power':
                    triplets = sample_triplets((csi_preprocessed, test_time), Tc)  # Get valid triplets
                if triplets is not None:
                    anchors, positives, negatives, _ = triplets

                # Pass through the model
                probabilities_anchor = model(anchors)  # (batch_size, N)
                probabilities_positive = model(positives)
                probabilities_negative = model(negatives)

                # Compute mean estimated positions
                mean_pos_anchor = torch.sum(probabilities_anchor.unsqueeze(2) * grid_positions, dim=1)  # (batch_size, 2)
                mean_pos_positive = torch.sum(probabilities_positive.unsqueeze(2) * grid_positions, dim=1)
                mean_pos_negative = torch.sum(probabilities_negative.unsqueeze(2) * grid_positions, dim=1)

                # Compute and save Triplet Loss
                triplet_loss = criterion_triplet(mean_pos_anchor, mean_pos_positive, mean_pos_negative)
                triplet_losses.append(triplet_loss.item())
            else:
                feature_losses = []
                triplet_losses = []     

            # Store results
            if save_prob:
                probabilities_map.append(probabilities.cpu().numpy())
            predicted_positions.append(mean_pos.cpu().numpy())
            actual_positions.append(true_pos.cpu().numpy())
            test_timestamps.append(test_time.cpu().numpy())
            pos_losses.append(dist_loss.item())
            dist_list.append(dist_temp.cpu().numpy())

        #Take the mean of loss functions 
        pos_losses = np.mean(pos_losses)
        if any(feature_losses):
            feature_losses = np.mean(feature_losses)
        else:
            feature_losses = []
        if any(triplet_losses):
            triplet_losses = np.mean(triplet_losses)
        else:
            triplet_losses = []

    # Convert lists to arrays
    test_timestamps = np.vstack(test_timestamps)
    predicted_positions = np.vstack(predicted_positions)
    actual_positions = np.vstack(actual_positions)
    dist_list = np.concatenate(dist_list)

    #Compute TW, CT, KS, RD and 95 percentile
    tw, ct = computeTWCT(actual_positions, predicted_positions)
    ks = compute_KS(actual_positions, predicted_positions)
    rd = compute_RD(actual_positions, predicted_positions)
    perc_95 = np.percentile(dist_list, 95)

    # Get the sorting indices based on test_timestamps
    sorted_indices = np.argsort(test_timestamps, axis=0)

    # Sort all arrays using the same indices
    test_timestamps = test_timestamps[sorted_indices[:, 0]]
    predicted_positions = predicted_positions[sorted_indices[:, 0]]
    actual_positions = actual_positions[sorted_indices[:, 0]]
    
    #Save probability maps only if needed
    if save_prob:
        probabilities_map = np.vstack(probabilities_map)
        probabilities_map = probabilities_map[sorted_indices[:, 0]]

    return predicted_positions, actual_positions, probabilities_map, pos_losses, triplet_losses, feature_losses, tw, ct, ks, rd, perc_95


################################################
# Compute affine transform (only for baseline) #
################################################
def computeAffineTransform(train_loader, model, noise_var, grid_positions, columns=13, device='cuda'):
    '''
    Function that computes the optimal affine transformation using all the data in
    train_loader, i.e., it solves the least-squares problem of (25) for the affine
    transformation baseline. The estimated positions are stacked with a column of
    ones so that the returned matrix contains both the matrix A and the bias b.
    '''

    #Gather all CSIs and true positions from the train loader (careful with memory)
    csi_raw_list, true_pos_list = [], []
    for batch in train_loader:
        csi_raw, true_pos, _ = batch
        csi_raw_list.append(csi_raw)
        true_pos_list.append(true_pos)
    csi_raw = torch.vstack(csi_raw_list)
    true_pos = torch.vstack(true_pos_list)

    #1. Compute estimated positions for all training dataset
    #Add noise to the CSI matrix
    csi_raw = addNoiseCSINonZero(csi_raw, noise_var)
    #Preprocess the CSI to obtain the inputs to the NN
    csi = preprocess_csi(csi_raw, columns)

    #First compute the metrics that do not require to sample form the triplet
    probabilities = model(csi)  # Get probability vector (batch_size, N)

    # Compute mean estimated position
    mean_pos = torch.sum(probabilities.unsqueeze(2) * grid_positions, dim=1)  # (batch_size, 2)

    # Stack estimations to formulate the problem as a LS problem
    mean_pos_stack = torch.cat((mean_pos, torch.ones((mean_pos.shape[0],1),device=device)), dim=-1)

    #2. Compute the optimal LS solution (matrix+bias)
    opt_matrix = torch.linalg.lstsq(mean_pos_stack, true_pos).solution
                
    return opt_matrix

def testAffineTransform(opt_matrix, test_loader, model, noise_var, grid_positions, columns=13, device='cuda'):
    '''
    Function to test the affine transformation baseline, where the positions
    estimated by the CC function are mapped to true coordinates with the optimal
    affine transformation opt_matrix obtained from computeAffineTransform.
    Returns the estimated and true positions (sorted by timestamp), the position
    loss and the evaluation metrics of Appendix A.
    '''
     # Testing Loop
    model.eval()
    predicted_positions, actual_positions = [], []
    test_timestamps = []            #Although I don't save this, this will be useful to order positions
    pos_losses = []
    dist_list = []                  #List to save distances between true and estimated positions

    with torch.no_grad():
        for batch in test_loader:
            csi_raw, true_pos, test_time = batch

            #Add noise to the CSI matrix
            csi_raw = addNoiseCSINonZero(csi_raw, noise_var)
            #Preprocess the CSI to obtain the inputs to the NN
            csi = preprocess_csi(csi_raw, columns)

            #First compute the metrics that do not require to sample form the triplet
            probabilities = model(csi)  # Get probability vector (batch_size, N)

            # Compute mean estimated position
            mean_pos = torch.sum(probabilities.unsqueeze(2) * grid_positions, dim=1)  # (batch_size, 2)
            # Apply affine transformation
            mean_pos_stacked = torch.cat((mean_pos, torch.ones((mean_pos.shape[0],1), device=device)), dim=-1)
            est_pos = mean_pos_stacked @ opt_matrix

            # Compute position loss according to the paper 
            dist_temp = torch.linalg.vector_norm(est_pos - true_pos, dim=-1)
            dist_loss = dist_temp.mean()

            #Store results
            predicted_positions.append(est_pos.cpu().numpy())
            actual_positions.append(true_pos.cpu().numpy())
            test_timestamps.append(test_time.cpu().numpy())
            pos_losses.append(dist_loss.item())
            dist_list.append(dist_temp.cpu().numpy())

    # Convert lists to arrays
    test_timestamps = np.vstack(test_timestamps)
    predicted_positions = np.vstack(predicted_positions)
    actual_positions = np.vstack(actual_positions)
    dist_list = np.concatenate(dist_list)

    #Compute TW, CT, KS, RD and 95 percentile
    tw, ct = computeTWCT(actual_positions, predicted_positions)
    ks = compute_KS(actual_positions, predicted_positions)
    rd = compute_RD(actual_positions, predicted_positions)
    perc_95 = np.percentile(dist_list, 95)

    # Get the sorting indices based on test_timestamps
    sorted_indices = np.argsort(test_timestamps, axis=0)

    # Sort all arrays using the same indices
    test_timestamps = test_timestamps[sorted_indices[:, 0]]
    predicted_positions = predicted_positions[sorted_indices[:, 0]]
    actual_positions = actual_positions[sorted_indices[:, 0]]

    return predicted_positions, actual_positions, pos_losses, tw, ct, ks, rd, perc_95
