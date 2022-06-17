from RulkovNetwork import RulkovNetwork
import jax.numpy as jnp

adjacency_matrix=jnp.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
network = RulkovNetwork(adjacency_matrix)
while network.t < 20:
    network.evolve()