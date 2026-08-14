'''
File that builds the matrix of predefined DT positions X of Sec. II, i.e., the
grid of points represented in Fig. 2. The floor plan is covered with several
rectangular UE sets defined in Wireless InSite, each one given by its bottom-left
corner and its sides along X and Y. The spacing between grid points corresponds
to the parameter Delta studied in Table III.
'''
import numpy as np

#Information from Wireless Insite
# num_UE_sets = 2
# spacing = 1.0
# initial_point = np.array([
#     [0.230161, 0.251768],
#     [7.19394, 16.0596]
# ])
# side_x = np.array([9.10874, 2.05906])
# side_y = np.array([15.6077, 10.1606])

# The grid of points with a spacing of 0.25m
num_UE_sets = 6
spacing = 0.25
initial_point = np.array([
    [0.054340, 0.074431],
    [0.100312, 6.42125],
    [3.13796, 6.62215],
    [3.18449, 9.16146],
    [2.95066, 13.1853],
    [7.10878, 15.9925]
])
side_x = np.array([
    9.34, 
    2.77,
    6.21,
    6.21,
    6.41,
    2.29
    ])
side_y = np.array([
    6.31, 
    9.42,
    2.28,
    3.71,
    2.68,
    10.42
    ])

# The original grid of points (spacing of 0.5m)
'''
num_UE_sets = 6
spacing = 0.5
initial_point = np.array([
    [0.054340, 0.074431],
    [0.100312, 6.421246],
    [3.137958, 6.622153],
    [3.184485, 9.161460],
    [2.950663, 13.18526],
    [7.10878, 15.99255]
])
side_x = np.array([
    9.34, 
    2.77,
    6.21,
    6.21,
    6.41,
    2.29
    ])
side_y = np.array([
    6.31, 
    9.42,
    2.28,
    3.71,
    2.68,
    10.42
    ])
'''
eps = 1e-6                              #small value so that the grid includes the last point

#Loop through all UE sets to obtain a matrix of (X,Y) positions
#We don't know how long each grid of point will be
pos_x = []
pos_y = []
for i in range(num_UE_sets):
    arange_x = np.arange(initial_point[i,0],initial_point[i,0]+side_x[i]+eps, spacing)
    arange_y = np.arange(initial_point[i,1],initial_point[i,1]+side_y[i]+eps, spacing)

    mesh_x, mesh_y = np.meshgrid(arange_x, arange_y)

    pos_x.extend(mesh_x.flatten())
    pos_y.extend(mesh_y.flatten())

pos_matrix = np.zeros((len(pos_x),2))
pos_matrix[:,0] = pos_x
pos_matrix[:,1] = pos_y

#Save the grid in the DT folder that corresponds to the selected spacing
save_folder = 'Wireless_InSite_data/Output_data/data_dt_0_25_spacing/'
np.save(save_folder + 'pos_matrix.npy', pos_matrix)
