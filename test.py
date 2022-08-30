import os, sqlite3
import numpy as np
from rulkov_network import RulkovNetwork

# create a 4x4 adjacency matrix
adj_matrix = np.array([[0, 1, 1, 1],
                          [1, 0, 1, 1],
                            [1, 1, 0, 1],
                                [1, 1, 1, 0]])

# create the delta_t 4x4 matrix as a numpy array with random positive floats
delta_t = np.random.rand(4, 4)
w_max = 0.2
# create a rulkov network object
network = RulkovNetwork(adj_matrix, w_max, 0.1, simulation_id='test', save_weights_mode=False, save_nodes_mode=False, save_maxima_mode=False)
network.delta_t = delta_t