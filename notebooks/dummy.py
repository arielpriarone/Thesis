from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Shaft']

# Get the source and destination collections
source_collection = db['QUARANTINED']
destination_collection = db['HEALTHY']

# Find the oldest 300 documents based on the "timestamp" field
documents_to_move = source_collection.find().sort("timestamp", 1).limit(300)

# Move documents to the destination collection
for document in documents_to_move:
    # Remove the _id field
    document_id = document.pop('_id', None)
    destination_collection.insert_one(document)
    # Delete the moved document from the source collection
    source_collection.delete_one({"_id": document_id})

# Close the MongoDB connection
client.close()