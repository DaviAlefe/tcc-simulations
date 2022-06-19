from dataclasses import dataclass
import jax
import sqlite3
from datetime import datetime


# A Rulkov network is initialized by an nxn adjacency matrix given by a jax array.
# The adjacency matrix is a jax array of shape (n, n) where n is the number of nodes in the network.
@dataclass
class RulkovNetwork:
    adjacency_matrix: jax.numpy.ndarray
    n: int
    weights: jax.numpy.ndarray
    nodes: jax.numpy.ndarray
    alpha: jax.numpy.ndarray
    sigma: float
    beta: float
    t: int
    nodes_x: jax.numpy.ndarray
    nodes_y: jax.numpy.ndarray

    # The constructor for the RulkovNetwork class.
    # The default value of simulation_id is simulation_ current datetime
    def __init__(self, adjacency_matrix, simulation_id=f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
        self.adjacency_matrix = adjacency_matrix
        self.n = adjacency_matrix.shape[0]
        self.simulation_id = simulation_id
        # An attribute of this class is the matrix for weights of the network, initialized to be the adjacency matrix.
        self.weights = adjacency_matrix
        # The network nodes are represented by a jax array of shape (n, 2) initialized to random floats in the interval [-2,2] for the first dimension and in the interval [-4, 0] for the second dimension.
        self.nodes_x = jax.random.uniform(jax.random.PRNGKey(0), shape=(self.n, 1), minval=-2, maxval=2)
        self.nodes_y = jax.random.uniform(jax.random.PRNGKey(0), shape=(self.n, 1), minval=-4, maxval=0)
        # The network nodes are represented by a composition of nodes_x and nodes_y side by side.
        self.nodes = jax.numpy.concatenate((self.nodes_x, self.nodes_y), axis=1)
        # Each node is associated to a parameter alpha in the interval [4.1,4.3]
        self.alpha = jax.random.uniform(jax.random.PRNGKey(0), shape=(self.n, 1), minval=4.1, maxval=4.3)
        # There are parameters sigma and beta, both equal 0.001
        self.sigma = 0.001
        self.beta = 0.001
        # The network evolves in the variable t for integer timesteps
        self.t = 0
        # A dictionary called local maximizers will be used to store the local maximizers of the nodes' y variable, with one key for each node.
        self.local_maximizers = {}
        # An array called increment count will be used to count the number of times the y variable increases in a streak.
        self.increment_count = jax.numpy.zeros((self.n, 1))

    
    # The method save_nodes saves the nodes of the network in a sqlite file in the same directory, associated with the current time variable.
    def save_nodes(self):
        # The connection to a sqlite file in a directory named after the simulation_id is opened or created if doesnt exist.
        conn = sqlite3.connect(f'simulations_data/{self.simulation_id}.db')
        # The cursor is created.
        c = conn.cursor()
        # The table variables is created if it does not exist.
        c.execute('''CREATE TABLE IF NOT EXISTS Variables (
            neuron_idx INTEGER,
            t INTEGER,
            x REAL,
            y REAL
        )''')
        # The nodes are concatenated into an array called data, with the array indices as neuron_ids in first column, time as the second column and repeated for each node.
        data = jax.numpy.concatenate((jax.numpy.arange(self.n).reshape((self.n, 1)), self.t*jax.numpy.ones((self.n, 1)), self.nodes), axis=1)
        # transform data to list of tuples
        data = data.tolist()
        # The array data is then saved in the table variables of the sqlite file, in columns x, y, t.
        c.executemany('''INSERT INTO Variables (neuron_idx, t, x, y) VALUES (?, ?, ?, ?)''', data)
        # The changes are committed.
        conn.commit()
        # The connection is closed.
        conn.close()
    
    # The method save_maxima takes the local maximizers as input and saves the nodes' y variable in the table LocalMaxima in the sqlite db, associated with the node id and time variable.
    def save_maxima(self, maximum, maximum_time, neuron_id):
        # The connection to a sqlite file in a directory named after the simulation_id is opened or created if doesnt exist.
        conn = sqlite3.connect(f'simulations_data/{self.simulation_id}.db')
        # The cursor is created.
        c = conn.cursor()
        # The table LocalMaxima is created if it does not exist.
        c.execute('''CREATE TABLE IF NOT EXISTS LocalMaxima (
            neuron_idx INTEGER,
            t INTEGER,
            y REAL
        )''')
        # The maximum is saved in the table LocalMaxima in the columns neuron_idx, t and y.
        c.execute('''INSERT INTO LocalMaxima (neuron_idx, t, y) VALUES (?, ?, ?)''', (neuron_id, maximum_time, maximum))

    # The method watch_increments receives the previous and compares to the current value of the y variable and accounts for the increment count for each node if the y variable increases in a streak.
    def watch_increments(self, previous):
        # If the y variable increases for a node, the increment count for that node is incremented in the index for that node.
        self.increment_count = self.increment_count.at[jax.numpy.where(self.nodes_y > previous)].set(1)
        # If the y variable decreases for a node and increment count for that node is greater than 50, the current time is appended to the list that is a value in local maximizers in the node's index as key and the increment count for that node is reset to 0.
        for i in jax.numpy.where(self.nodes_y < previous)[0]:
            if self.increment_count[i] > 50:
                self.local_maximizers[i] = jax.numpy.append(self.local_maximizers[i], self.t)
                self.increment_count[i] = 0
                # The method save_maxima is called to save the local maximizer of the node in the sqlite db.
                self.save_maxima(self.nodes_y[i], self.t, i)


    # The method fire implements the Rulkov Map, updating the network nodes.
    def fire(self):
        # the coupling term is the dot product of the transpose of weights and the nodes_x.
        coupling = jax.numpy.matmul(self.weights.T, self.nodes_x)
        # nodes_x is updated to \frac{\alpha}{2+x_n^2}+y_n+I_i,t
        self.nodes_x = self.alpha/(2+jax.numpy.square(self.nodes_x))+self.nodes_y + coupling
        # nodes_y is updated to y_n- \sigma x_n - \beta
        self.nodes_y = self.nodes_y - self.sigma*self.nodes_x - self.beta
        # The nodes are updated to the composition of nodes_x and nodes_y side by side.
        self.nodes = jax.numpy.concatenate((self.nodes_x, self.nodes_y), axis=1)

    def evolve(self):
        # The current value of the y variable is saved for each node in the previous y variable for later use in the watch_increments method.
        previous_y = self.nodes_y
        # The fire method is called.
        self.fire()
        # The network evolves in the variable t for integer timesteps
        self.t += 1
        # The watch_increments method is called to account for the increment count for each node if the y variable increases in a streak.
        self.watch_increments(previous_y)
        # The new values of the nodes are saved in the sqlite file.
        self.save_nodes()
