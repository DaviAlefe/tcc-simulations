import sqlite3
import numpy as np
import pandas as pd
from pandasql import sqldf
from itertools import product


def get_kuramoto(db_path, t_start, t_end, n_nodes):
    print(db_path)
    con = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM LocalMaxima',con)
    initial_df = pd.DataFrame(list(product(np.arange(n_nodes),np.arange(t_start-500,t_end + 500))),columns=['neuron_id','t'])
    query = '''
    WITH j AS 
    (SELECT *
    FROM initial_df i
    LEFT JOIN
    (SELECT neuron_idx, t AS t_burst
    FROM df) r
    ON i.t = r.t_burst AND i.neuron_id = r.neuron_idx)

    SELECT neuron_id, t, t_burst
    FROM j
    '''
    initial_df = sqldf(query)
    df_t_next = initial_df.interpolate(method='bfill')
    df_t_next.rename(columns={'t_burst':'t_next'}, inplace=True)
    df_t_last = initial_df.interpolate(method='pad')
    df_t_last.rename(columns={'t_burst':'t_last'}, inplace=True)
    query = '''
    WITH j AS
    (SELECT *
    FROM df_t_next n
    LEFT JOIN
    df_t_last l
    ON l.neuron_id = n.neuron_id AND l.t = n.t)

    SELECT neuron_id, t, t_last, t_next
    FROM j
    '''

    initial_df = sqldf(query)

    query = '''
    SELECT neuron_id, t, t_last, t_next,
    (DENSE_RANK() OVER (
    PARTITION BY neuron_id
    ORDER BY t_last ASC
    )
     ) AS burst_num
    FROM initial_df
    '''

    initial_df = sqldf(query)

    query = f'''
    SELECT neuron_id, t, t_last, t_next, burst_num
    FROM initial_df
    WHERE t >= {t_start} AND t <= {t_end}
    '''
    initial_df = sqldf(query)

    return initial_df