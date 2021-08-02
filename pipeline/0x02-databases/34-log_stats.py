#!/usr/bin/env python3
"""Log stats"""
from pymongo import MongoClient


if __name__ == "__main__":
    """provides some stats about Nginx logs stored in MongoDB"""
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_collection = client.logs.nginx
    nb_doc = logs_collection.count_documents({})
    print('{} logs'.format(nb_doc))
    print('Methods:')
    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for item in method:
        methode = logs_collection.count_documents({"method": item})
        print('\tmethod {}: {}'.format(item, methode))
    filter = {"method": "GET", "path": "/status"}
    print("{} status check".format(logs_collection.count_documents(filter)))
