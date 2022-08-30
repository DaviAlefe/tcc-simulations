import os, sqlite3
import pandas, numpy as np, matplotlib.pyplot as plt

simulation_path = '/home/davialefe/tcc/simulations/simulations_data/simulation_20220824_104001/simulation_20220824_104001.db'

def plot_weights():
    # open the simulation database and get the weights table as a df
    conn = sqlite3.connect(simulation_path)
    weights = pandas.read_sql_query('SELECT * FROM LocalMaxima WHERE neuron_idx = 0', conn)
    # close the connection
    conn.close()
    # plot the weights
    # plt.plot(weights['time'], weights['mean_weight'])
    # plt.xlabel('time')
    # plt.ylabel('mean weight')
    # plt.show()
    print(weights)

if __name__ == '__main__':
    plot_weights()