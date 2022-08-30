from get_kuramoto import get_kuramoto
import os
import pandas as pd
import sqlite3

def save_kuramoto(db_path, t_start, t_end, n_nodes, table_name):
    df = get_kuramoto(db_path, t_start, t_end, n_nodes)
    # Save the df to the database, creating the table table_name if it doesn't exist
    con = sqlite3.connect(db_path)
    df.to_sql(table_name, con, if_exists='replace', index=False)

# for each folder in simulations_data, get kuramoto and save to the .db file inside the folder
for folder in os.listdir('/home/davialefe/tcc/simulations/simulations_data'):
    try:
        # if it is a folder
        simulations_data_path = '/home/davialefe/tcc/simulations/simulations_data'
        if os.path.isdir(os.path.join(simulations_data_path, folder)):
            # get the path to the .db file inside the folder
            db_name = [filename for filename in os.listdir(os.path.join(simulations_data_path, folder))  if \
                filename.endswith('.db')][0]
            if db_name:
                db_path = os.path.join(simulations_data_path, folder, db_name)
            N = 100
            initial_window = {'start': 10000, 'end': 20000}
            final_window = {'start': 1490000, 'end': 1500000}
            
            save_kuramoto(db_path, initial_window['start'], initial_window['end'], N, 'InitialKuramoto')
            save_kuramoto(db_path, final_window['start'], final_window['end'], N, 'FinalKuramoto')
    except Exception as e:
        print(e)
        print(f'Error saving kuramoto for {folder}')
        continue