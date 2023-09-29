from datetime import datetime
import src
MongoClient, Database, Collection = src.data.mongoConnect('IMS','RAW','mongodb://localhost:27017')
mydate=datetime.fromisoformat('2003-10-22T12:09:13.000+00:00')
res=Collection.find({'timestamp': mydate})[0]
print(res)