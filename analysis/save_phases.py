from get_phases import get_phases
import os
import pandas as pd
import sqlite3

def save_phases(db_path, t_start, t_end, n_nodes, table_name):
    df = get_phases(db_path, t_start, t_end, n_nodes)
    # Save the df to the database, creating the table table_name if it doesn't exist
    con = sqlite3.connect(db_path)
    df.to_sql(table_name, con, if_exists='replace', index=False)

if __name__ == '__main__':
    # for each folder in simulations_data, get phases and save to the .db file inside the folder
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
                
                save_phases(db_path, initial_window['start'], initial_window['end'], N, 'InitialWindow')
                save_phases(db_path, final_window['start'], final_window['end'], N, 'FinalWindow')
        except Exception as e:
            print(e)
            print(f'Error saving phases for {folder}')
            continue