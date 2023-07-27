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
    'timestamp': datetime.datetime.now().replace(tzinfo=None),
    'varName': 'acc_x',
    'sampFreq': 2000,
    'timeSerie': np.random.randn(2000).tolist()}

result=RecordCollection.insert_one(sample_to_add)
# %%

RecordCollection=db['ColProva']
result=RecordCollection.find().sort('timestamp',1).limit(1)
OldestRecord=[]
OldestRecord.append([x for x in result])
OldestRecord=OldestRecord[0]
OldestRecord=OldestRecord[0] # to convert to a dictionary
print(type(OldestRecord))

OldestRecord['timestamp']


