from pymongo import MongoClient
from configs import (CONNECTION_STRING, DB_NAME, COLLECTION_NAME)


from pymongo import MongoClient
client = MongoClient(CONNECTION_STRING)
collection = client[DB_NAME][COLLECTION_NAME]