# %%
from pymongo import MongoClient
import numpy as np
import datetime
client = MongoClient('mongodb://localhost:27017')
print(client.list_database_names())

db = client['ProvaDB']
RecordCollection=db['ColProva']
print(db.list_collection_names())

sample_to_add={
    'timestamp': datetime.datetime.utcnow()
    'varName': 'acc_x'
    'sampFreq': 2000
    'timeSerie': np.random.d}
# %%
