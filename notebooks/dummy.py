import pymongo
from pyparsing import col

DB = pymongo.MongoClient("mongodb://localhost:27017/")["test"]
print(DB.list_collection_names())
col = DB["test"]
res=col.update_one({"_id": "id"}, {"$set": {"_id": "id", "test": [9,1,2,3]}})
print(res.matched_count)