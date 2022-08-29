import os, json, sqlite3

data = []

# The path to simulations_data
simulations_data_path = '/home/davialefe/tcc/simulations/simulations_data'

for _ in os.listdir(simulations_data_path):
    # if _ is dir
    if os.path.isdir(os.path.join(simulations_data_path, _)):
        obj = {}
        obj['simulation_id'] = _
        # open the json file in _
        with open(os.path.join(simulations_data_path, _, f'{_}.json'), 'r') as f:
            # extract w_0_mult from the json file
            json_data = json.load(f)
            obj['w_0'] = json_data['w_0']
            # get w_max from the json file
            obj['w_max'] = json_data['w_max']
        
        obj['w_0_mult'] = round(obj['w_0'] / obj['w_max'],3)
        
        data.append(obj)

print(f'Data : {data}')
# create a .db file in the analysis folder
conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'data.db'))
# create a associations table in the .db file
# delete table if it exists
conn.execute('DROP TABLE IF EXISTS associations')
conn.execute('''CREATE TABLE associations
                (simulation_id text, w_0 real, w_0_mult real)''')
# order data by w_0
data = sorted(data, key=lambda x: x['w_0'])
# insert the data into the table
conn.executemany('INSERT INTO associations VALUES (:simulation_id, :w_0, :w_0_mult)', data)
# commit the changes
conn.commit()
# close the connection
conn.close()