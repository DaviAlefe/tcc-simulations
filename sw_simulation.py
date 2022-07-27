import jax.numpy as jnp
from networkx import watts_strogatz_graph
import networkx as nx
from rulkov_network import RulkovNetwork
from datetime import datetime

# Network Parameters
N = 1000
k = 4
p = 0.2

graph = watts_strogatz_graph(N, k, p)
adjcency_matrix = nx.to_numpy_matrix(graph)

w_max = 0.2
w_0 = 0.2*w_max

network = RulkovNetwork(adjcency_matrix, w_max, w_0)

# Simulation Parameters
transient = 1e4
T = 1.5e6
# The network does not update weights for transient time steps
network.freeze_weights = True

simulation_start = datetime.now()
print(30*'-')
print(f'Simulation started.')
while network.t < T:
    # Set the weights to be updated after transient time steps
    if network.t == transient:
        network.freeze_weights = False
        print('End of transient period.')

    network.evolve()

    # Mark time every 500 iterations and predict the time remaining
    if network.t % 500 == 0:
        current_time = datetime.now()
        time_elapsed = current_time - simulation_start
        print(f'\t\t {network.t} iterations, {time_elapsed} elapsed.')
        time_remaining = time_elapsed * (T - network.t) / network.t
        print(f'\t\t Approximately {time_remaining} remaining.')
        print(network.t*'=',(T-network.t)*'-')
