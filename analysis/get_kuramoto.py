import os, json, sqlite3
import pandas, pandasql, numpy as np

simulations_data_path = '/home/davialefe/tcc/simulations/simulations_data'
analysis_db_path = '/home/davialefe/tcc/simulations/analysis/data.db'

def get_kuramoto(simulation_id):
    # get the path to the .db file inside the folder
    db_name = [filename for filename in os.listdir(os.path.join(simulations_data_path, simulation_id))  if \
        filename.endswith('.db')][0]
    if db_name:
        db_path = os.path.join(simulations_data_path, simulation_id, db_name)
    # open the .db file
    conn = sqlite3.connect(db_path)
    # get the phases from the InitialWindow table
    initial_window = pandas.read_sql_query('SELECT neuron_id, t, phi FROM InitialWindow', conn)
    # get the phases from the FinalWindow table
    final_window = pandas.read_sql_query('SELECT neuron_id, t, phi FROM FinalWindow', conn)
    # close the connection
    conn.close()
    # calculate phasor for the initial window
    initial_window['phasor'] = np.exp(1j*initial_window['phi'])
    # calculate phasor for the final window
    final_window['phasor'] = np.exp(1j*final_window['phi'])
    # calculate the mean phasor's magnitudes for the initial window grouped by t
    initial_window_kuramotos = np.abs(initial_window.groupby('t')['phasor'].mean())
    initial_window_mean_kuramoto = np.mean(initial_window_kuramotos)
    # calculate the mean phasor's magnitudes for the final window grouped by t
    final_window_kuramotos = np.abs(final_window.groupby('t')['phasor'].mean())
    final_window_mean_kuramoto = np.mean(final_window_kuramotos)

    return (initial_window_mean_kuramoto, final_window_mean_kuramoto)
    

if __name__ == '__main__':
    # open the analysis database and get the associations table as a df
    conn = sqlite3.connect(analysis_db_path)
    associations = pandas.read_sql_query('SELECT * FROM Associations', conn)
    # for each simulation_id in the associations table, get the initial and final window kuramoto
    associations['initial_kuramoto'], associations['final_kuramoto'] = zip(*associations['simulation_id'].apply(get_kuramoto))
    # save the associations table to the analysis database
    associations.to_sql('associations', conn, if_exists='replace', index=False)
    # close the connection
    conn.close()