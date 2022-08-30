import os, sqlite3

simulations_data_path = '/home/davialefe/tcc/simulations/simulations_data'
# for each folder in simulations_data, open the .db file and delete the InitialKuramoto and FinalKuramoto tables
for _ in os.listdir(simulations_data_path):
    if os.path.isdir(os.path.join(simulations_data_path, _)):
        db_name = [filename for filename in os.listdir(os.path.join(simulations_data_path, _))  if \
            filename.endswith('.db')][0]
        if db_name:
            db_path = os.path.join(simulations_data_path, _, db_name)
            conn = sqlite3.connect(db_path)
            conn.execute('DROP TABLE IF EXISTS InitialKuramoto')
            conn.execute('DROP TABLE IF EXISTS FinalKuramoto')
            conn.commit()
            conn.close()
