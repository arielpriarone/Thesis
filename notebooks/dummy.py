from pymongo import MongoClient
import datetime as dt

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Shaft']

# Define the threshold timestamp
threshold = dt.datetime.fromisoformat("2004-02-18T18:00:00Z")  # Replace with your desired threshold

# Get the source and destination collections
source_collection = db['QUARANTINED']
destination_collection = db['FAULTY']

# Find documents with timestamp greater than the threshold
documents_to_move = source_collection.find({"timestamp": {"$gt": threshold}})

# Move documents to the destination collection
for document in documents_to_move:
    # Remove the _id field
    document.pop('_id', None)
    destination_collection.insert_one(document)

# Delete the moved documents from the source collection
source_collection.delete_many({"timestamp": {"$gt": threshold}})

# Close the MongoDB connection
client.close()