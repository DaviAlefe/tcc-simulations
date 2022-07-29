from rulkov_network import RulkovNetwork
import jax.numpy as jnp


adjacency_matrix=jnp.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
# adjacency_matrix=jnp.array([[0]])
network = RulkovNetwork(adjacency_matrix, w_max=0.1, w_0=0.1)
# network.nodes_x = jnp.array([[1]])
# network.nodes_y = jnp.array([[-1]])
# network.nodes = jnp.concatenate((network.nodes_x, network.nodes_y), axis=1)
# network.alpha = jnp.array([[4.1]])

while network.t < 10:
    network.update_weights([1])
    network.t +=1
    print(f'Iteration {network.t}')
    print(f'\t Nodes: {network.nodes}')
    print(f'\t increment_count: {network.increment_count}')
