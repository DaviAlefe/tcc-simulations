import os, sqlite3
import pandas, numpy as np

simulations_data_path = '/home/davialefe/tcc/simulations/simulations_data'
analysis_db_path = '/home/davialefe/tcc/simulations/analysis/data.db'

def get_weights(simulation_id):
    # get the paths to the weights matrices saved as npy files inside the folder
    weights_matrices_files = [filename for filename in os.listdir(os.path.join(simulations_data_path, simulation_id))  if \
        filename.endswith(f'_weights_{simulation_id}.npy')]
    # load the weights matrices
    initial_weights_file = [filename for filename in weights_matrices_files if filename.startswith('20000')][0]
    initial_weights = np.load(os.path.join(simulations_data_path, simulation_id, initial_weights_file))
    final_weights_file = [filename for filename in weights_matrices_files if filename.startswith('1500000')][0]
    final_weights = np.load(os.path.join(simulations_data_path, simulation_id, final_weights_file))
    # load the adj_matrix.npy file in the folder
    adj_matrix = np.load(os.path.join(simulations_data_path, simulation_id, 'adj_matrix.npy'))

    initial_weights = np.multiply(initial_weights, adj_matrix)
    final_weights = np.multiply(final_weights, adj_matrix)

    # calculate the mean of the weights matrices
    initial_weights_mean = np.sum(initial_weights)/np.sum(adj_matrix)
    final_weights_mean = np.sum(final_weights)/np.sum(adj_matrix)

    # divide by the networks' max weight
    initial_weights_mean /= 0.2
    final_weights_mean /= 0.2

    return (initial_weights_mean, final_weights_mean)

if __name__ == '__main__':
    # open the analysis database and get the associations table as a df
    conn = sqlite3.connect(analysis_db_path)
    associations = pandas.read_sql_query('SELECT * FROM associations', conn)
    # for each simulation_id in the associations table, get the initial and final window kuramoto
    associations['initial_mean_weight'], associations['final_mean_weight'] = zip(*associations['simulation_id'].apply(get_weights))
    # save the associations table to the analysis database
    associations.to_sql('associations', conn, if_exists='replace', index=False)
    # close the connection
    conn.close()