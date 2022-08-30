import os, sqlite3
import pandas, numpy as np

simulations_data_path = '/home/davialefe/tcc/simulations/simulations_data'
analysis_db_path = '/home/davialefe/tcc/simulations/analysis/data.db'

def get_weights(simulation_id):
    # get the paths to the weights matrices saved as npy files inside the folder
    weights_matrices_files = [filename for filename in os.listdir(os.path.join(simulations_data_path, simulation_id))  if \
        filename.endswith(f'_weights_simulation_{simulation_id}.npy')]
    print(weights_matrices_files)