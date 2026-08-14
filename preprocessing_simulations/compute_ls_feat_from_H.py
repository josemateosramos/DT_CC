'''
File to extract all large-scale features from the DT from the H matrix, i.e.,
the feature vectors v(p) = h(H(p)) of the predefined DT positions that are used
in the DT-aided loss of (5). It computes the large-scale features h(.) described
in Sec. III-C: the angle-power profile (APP), the delay-power profile (DPP), the
covariance and the truncated delay profile (TDP). The Power feature is obtained
directly from the ray-tracing software in read_power.py.
'''
import numpy as np
import numpy.fft as fft
import torch

def IFFTTruncated(CSI, C):
    '''
    Function that computes the truncated delay-domain CSI [H F_S^H]_{:,1:C} of (1).
    '''
    CSI_ifft = torch.fft.ifft(CSI, dim=-1, norm='ortho')  # Take IFFT along subcarrier axis
    CSI_truncated = CSI_ifft[:, :, :C]  # Keep first C columns
    return CSI_truncated

def preprocess_csi(CSI, C):
    """Computes the normalized truncated delay profile |h|/||h|| of (1).
    Applies IFFT along rows, keeps first C columns, vectorizes and normalizes."""
    CSI_truncated = IFFTTruncated(CSI, C)
    CSI_vectorized = CSI_truncated.reshape(CSI.shape[0], -1)  # Flatten to vector
    norms = torch.norm(CSI_vectorized, dim=1, keepdim=True)
    return torch.abs(CSI_vectorized) / norms  # Normalize by its norm

#Simulation parameters from the digital twin
nAP = 8
nAntennas = 4
nSubcarriers = 52

#Load the CSI matrix for all antennas, APs and positions
#Path must end with a slash. It contains the H_all.npy simulated at the DT grid points
#(see read_csi_to_H.py), and the features are saved next to it, i.e., in the
#corresponding Wireless_InSite_data/Output_data/data_dt_* folder
root_folder = 'path_to_folder_with_H_all.npy/'
H_matrix = np.load(root_folder + 'H_all.npy')        #Shape: (num_grid_points, nAP*nAntennas, nSubcarriers)
H_matrix_rsh = H_matrix.reshape(H_matrix.shape[0], nAP, nAntennas, nSubcarriers)
n_points = H_matrix.shape[0]

'''
APP: angle-power profile of Sec. III-C2, i.e., the power received along each
angular direction (D = K per AP)
'''

#Take FFT along the antenna dimension and sum along the subcarrier dimension to obtain the angle profiles
H_matrix_fft = fft.fft(H_matrix_rsh,axis=-2, norm='ortho')            #Shape: (num_grid_points, nAP, nAntennas, nSubcarriers)
angle_prof = np.sum(abs(H_matrix_fft)**2, axis=-1)  #Shape: (num_grid_points, nAP, nAntennas)

#Concatenate the angle profiles of each AP
angle_prof_concat = angle_prof.reshape(-1,nAP*nAntennas)

#Save results
np.save(root_folder + 'angle_profiles.npy', angle_prof_concat)

'''
DPP: delay-power profile of Sec. III-C3, i.e., the power received at each of the
S delay taps (D = S per AP)
'''
H_matrix_ifft = fft.ifft(H_matrix_rsh,axis=-1, norm='ortho')            #Shape: (num_grid_points, nAP, nAntennas, nSubcarriers)
delay_prof = np.sum(abs(H_matrix_ifft)**2, axis=-2)  #Shape: (num_grid_points, nAP, nSubcarriers)

#Concatenate the delay profiles of each AP
delay_prof_concat = delay_prof.reshape(-1,nAP*nSubcarriers)

#Save results
np.save(root_folder + 'delay_profiles.npy', delay_prof_concat)

'''
Covariance: magnitude of the covariance matrix of the beamspace CSI of
Sec. III-C4 (D = K^2 per AP)
'''
H_matrix_rsh_fft = np.fft.fft(H_matrix_rsh,axis=-2, norm='ortho')    #FFT across antenna dimension

#Compute the covariance
H_cov_fft = H_matrix_rsh_fft @ H_matrix_rsh_fft.transpose(0,1,3,2).conjugate()   #Shape: (num_grid_points, nAP, nAntennas, nAntennas)
H_cov_fft_rsh = H_cov_fft.reshape(H_cov_fft.shape[0],-1)

#Compute the abs of the covariance
H_cov_fft_rsh_abs = np.abs(H_cov_fft_rsh)

np.save(root_folder + 'cov_profiles_abs_fft.npy', H_cov_fft_rsh_abs)

'''
TDP: truncated delay profile of Sec. III-C5, i.e., the CSI feature of (1) used
both as input of the positioning function and as large-scale feature (D = K*C per AP)
'''
columns = 13

H_matrix = torch.tensor(np.load(root_folder + 'H_all.npy'), device='cuda')        #Shape: (num_grid_points, nAP*nAntennas, nSubcarriers)

#Preprocess the CSI matrix for the input of the NN
H_tdp = preprocess_csi(H_matrix, columns)

#Save results
np.save(root_folder + 'tdp.npy', H_tdp.cpu().detach())
