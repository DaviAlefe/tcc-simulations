import jax.numpy as jnp
import numpy as np
from networkx import watts_strogatz_graph
import networkx as nx
from rulkov_network import RulkovNetwork
from datetime import datetime
import logging, sys, json, os


class SWSimulation:
    def __init__(self, w_0_mult, base_dir='.'):
        self.simulation_id=f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.base_dir = base_dir

        # Network Parameters
        self.N = 100
        self.k = 4
        self.p = 0.2


        try:
            self.load_adj_matrix()
        except:
            self.graph = watts_strogatz_graph(self.N, self.k, self.p)
            self.adjcency_matrix = jnp.array(nx.to_numpy_matrix(self.graph))

        self.w_max = 0.2
        self.w_0 = w_0_mult * self.w_max

        self.network = RulkovNetwork(self.adjcency_matrix, self.w_max, self.w_0,
        simulation_id=self.simulation_id, save_weights_mode=False, save_nodes_mode=False, save_maxima_mode=False)

        # Simulation Parameters
        self.transient = 1e4
        self.T = 1.5e6

        # Logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
        force=True,
        handlers = [logging.FileHandler(f'{base_dir}/simulations_data/{self.simulation_id}.log'), logging.StreamHandler(sys.stdout)])

        # Create dir inside simulations_data with the current simulation id
        if not os.path.exists(f'{base_dir}/simulations_data/{self.simulation_id}'):
            os.makedirs(f'{base_dir}/simulations_data/{self.simulation_id}')
        self.save_parameters()

    # the load_adj_matrix method
    def load_adj_matrix(self):
        path = '/home/davialefe/tcc/simulations/sw_adj_matrix.npy'
        self.adjcency_matrix = jnp.array(np.load(path))

    # Save parameters method
    def save_parameters(self):
        # Log the simulation parameters
        logging.info(f'Simulation parameters:')
        logging.info(str(self.__dict__))
        # save the simulation dict, except self.graph, in a file
        with open(f'{self.base_dir}/simulations_data/{self.simulation_id}.json', 'w') as f:
            dict_to_save = self.__dict__.copy()
            if 'graph' in dict_to_save:
                dict_to_save.pop('graph')
            # set the rulkov network key to be the serializable objets in rulkov network's __dict__
            rulkov_network_dict = self.network.__dict__.copy()
            # turn every np or jnp array in the rulkov network dict to lists
            for key in rulkov_network_dict:
                if isinstance(rulkov_network_dict[key], np.ndarray):
                    rulkov_network_dict[key] = rulkov_network_dict[key].tolist()
                elif isinstance(rulkov_network_dict[key], jnp.ndarray):
                    rulkov_network_dict[key] = rulkov_network_dict[key].tolist()
            dict_to_save['network'] = rulkov_network_dict

            # convert every jnp or numpy matrix array to lists
            for key, value in dict_to_save.items():
                if isinstance(value, jnp.ndarray):
                    dict_to_save[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    dict_to_save[key] = value.tolist()
            json.dump(dict_to_save, f)

            # save adjacency matrix in a npy file
            np.save(f'{self.base_dir}/simulations_data/{self.simulation_id}/adj_matrix.npy', self.adjcency_matrix)
    
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
                logging.info('Beginning of initial state.')

            self.network.evolve()

            # Mark time every 100 iterations and predict the time remaining
            if self.network.t % 100 == 0:
                current_time = datetime.now()
                time_elapsed = current_time - simulation_start
                logging.info(f'\t\t {self.network.t} iterations, {time_elapsed} elapsed.')
                time_remaining = time_elapsed * (self.T - self.network.t) / self.network.t
                logging.info(f'\t\t Approximately {time_remaining} remaining.')
            
            
            # Save the network weights at t=2e4
            if self.network.t == self.transient + 1e4:
                # Save the network weights in the range [0, N-1]
                self.network.save_weights(jnp.arange(self.N))
                # Save the network weights as a npy file
                self.network.save_weights_matrix(f'{self.base_dir}/simulations_data/{self.simulation_id}')
                logging.info(f'\t\t Saved weights at t={self.network.t}.')
                logging.info('\t\t End of initial state.')

            
            # Define window to save maxima
            initial_window = lambda t: (t > self.transient - 500) and (t < self.transient + 10500)
            final_window = lambda t: (t > self.T - 10500) and (t < self.T)
            if initial_window(self.network.t) or final_window(self.network.t):
                self.network.save_maxima_mode = True
            else:
                self.network.save_maxima_mode = False

            # Log beginning of final state
            if self.network.t == self.T - 1e4:
                logging.info('\t\t Beginning of final state.')
        
        # Save the network weights at the end of the simulation
        self.network.save_weights(jnp.arange(self.N))
        # Save the network weights as a npy file
        self.network.save_weights_matrix(f'{self.base_dir}/simulations_data/{self.simulation_id}')
        logging.info(f'\t\t Saved weights at t={self.network.t}.')
        logging.info(f'Simulation finished. {datetime.now() - simulation_start} elapsed.')

        # Move every file from the simulations_data folder to the simulations_data/{self.simulation_id} folder
        for file in os.listdir(f'{self.base_dir}/simulations_data'):
            # Move only files, not directories
            if file.startswith(self.simulation_id) and os.path.isfile(f'{self.base_dir}/simulations_data/{file}'):
                os.rename(f'{self.base_dir}/simulations_data/{file}', f'{self.base_dir}/simulations_data/{self.simulation_id}/{file}')

            
        return "done"