from datetime import datetime
import src
import yaml
from typing import Dict, Any
# MongoClient, Database, Collection = src.data.mongoConnect('IMS','RAW','mongodb://localhost:27017')
# mydate=datetime.fromisoformat('2003-10-22T12:09:13.000+00:00')
# res=Collection.find({'timestamp': mydate})[0]
# print(res)

# with open('config.yaml','r') as f:
#     config = yaml.safe_load(f)
# print(config)

# import pymongo
# client = pymongo.MongoClient('mongodb://localhost:27017')
# db = client['Shaft']
# col = db['UNCONSUMED']
# col.delete_many({}) # delete all documents in the collection



existing_dict: Dict[str, Any] = {'timestamp': src.data.IMS_filepathToTimestamp('c')}

# __update dictionary
__varname = 'sensor3'  # Example sensor name
__update = {
    __varname: {
        'sampFreq': 20000,
        'timeSerie': [7, 8, 9],
    }
}

# Merge __update into existing_dict while keeping 'timestamp' key
existing_dict.update(__update)

# Now existing_dict contains both 'timestamp' and the updated sensor information
print(existing_dict)
