import jax.numpy as jnp
from networkx import watts_strogatz_graph
import networkx as nx
from rulkov_network import RulkovNetwork
from datetime import datetime
import logging, sys


class SWSimulation:
    def __init__(self, w_0_mult, base_dir='.'):
        self.simulation_id=f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.base_dir = base_dir

        # Network Parameters
        self.N = 100
        self.k = 4
        self.p = 0.2

        self.graph = watts_strogatz_graph(self.N, self.k, self.p)
        self.adjcency_matrix = nx.to_numpy_matrix(self.graph)

        self.w_max = 0.2
        self.w_0 = w_0_mult * self.w_max

        self.network = RulkovNetwork(self.adjcency_matrix, self.w_max, self.w_0,
         simulation_id=self.simulation_id)

        # Simulation Parameters
        self.transient = 1e4
        self.T = 1.5e6

        # Logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
        force=True,
        handlers = [logging.FileHandler(f'{base_dir}/simulations_data/{self.simulation_id}.log'), logging.StreamHandler(sys.stdout)])
        # Log the simulation parameters
        logging.info(f'Simulation parameters:')
        logging.info(str(self.__dict__))
    
    def run(self):
        # The network does not update weights for transient time steps
        self.network.freeze_weights = True

        simulation_start = datetime.now()
        logging.info(30*'-')
        logging.info(f'Simulation started.')
        while self.network.t < self.T:
            # Set the weights to be updated after transient time steps
            if self.network.t == self.transient:
                self.network.freeze_weights = False
                logging.info('End of transient period.')

            self.network.evolve()

            # Mark time every 100 iterations and predict the time remaining
            if self.network.t % 100 == 0:
                current_time = datetime.now()
                time_elapsed = current_time - simulation_start
                logging.info(f'\t\t {self.network.t} iterations, {time_elapsed} elapsed.')
                time_remaining = time_elapsed * (self.T - self.network.t) / self.network.t
                logging.info(f'\t\t Approximately {time_remaining} remaining.')
        return "done"