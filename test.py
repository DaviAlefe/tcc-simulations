from RulkovNetwork import RulkovNetwork
import jax.numpy as jnp

# adjacency_matrix=jnp.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
adjacency_matrix=jnp.array([[0]])
network = RulkovNetwork(adjacency_matrix)
network.nodes_x = jnp.array([[-1]])
network.nodes_y = jnp.array([[-2.75]])
network.nodes = jnp.array([[-1, -2.75]])
network.alpha = jnp.array([[4.1]])
while network.t < 3000:
    network.evolve()
    print(network.local_maximizers)
    print(f'Iteration {network.t}')
    print(f'\t Nodes: {network.nodes}')