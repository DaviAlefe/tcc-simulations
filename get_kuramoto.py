import sqlite3
import numpy as np
import pandas as pd

db_path = '/home/davialefe/tcc/simulations/simulations_data/simulation_20220821_183756/simulation_20220821_183756.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

N=100
T=1e4
# Build a df with 2 columns, one for the time from T to 2T and the other for the N nodes indexes, one for each time step
df = pd.DataFrame(columns=['time', 'node_index'])
for i in range(N):
    df.loc[i] = [T, i] 