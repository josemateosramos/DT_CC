'''
NN architectures used as positioning function of the modified CC of Sec. III-B1.
'''
import torch.nn as nn

class MLPSoftmaxOutputDropout(nn.Module):
    '''
    MLP with ReLU activations and dropout that implements the positioning
    function g(.) of Sec. III-B1. The softmax output returns the probability
    mass function p of size P, whose ith entry is the probability that the ith
    predefined position of the DT is the true position of the UE.
    Inputs
    ----------
    layer_sizes: list with the input dimension followed by the number of
        neurons of each hidden layer.
    num_grid_points: number of predefined positions P in the DT.
    dropout_rate: dropout rate applied after every hidden layer.
    '''
    def __init__(self, layer_sizes, num_grid_points, dropout_rate):
        super(MLPSoftmaxOutputDropout, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i == 0:  # First layer uses Glorot Initialization
                nn.init.xavier_uniform_(layers[-1].weight)
            else:  # Remaining layers use He Initialization
                nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(layer_sizes[-1], num_grid_points))  
        layers.append(nn.Softmax(dim=1))  # Probability distribution

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

