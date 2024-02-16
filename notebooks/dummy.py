from pymongo import MongoClient
import datetime as dt

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Shaft']

# Define the threshold timestamp
threshold = dt.datetime.fromisoformat("2003-11-23T00:00:00.000+00:00")  # Replace with your desired threshold

# Get the source and destination collections
source_collection = db['QUARANTINED']
destination_collection = db['BACKUP']

# Find documents with timestamp greater than the threshold
documents_to_move = source_collection.find({"timestamp": {"$gt": threshold}})

# Move documents to the destination collection
for document in documents_to_move:
    destination_collection.insert_one(document)

# Delete the moved documents from the source collection
source_collection.delete_many({"timestamp": {"$gt": threshold}})

# Close the MongoDB connection
client.close()