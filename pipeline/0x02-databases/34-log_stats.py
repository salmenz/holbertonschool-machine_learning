#!/usr/bin/env python3
"""provides some stats about Nginx logs stored in MongoDB"""
from pymongo import MongoClient


if __name__ == "__main__":
    """provides some stats about Nginx logs stored in MongoDB"""

    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx
    count_documents = logs.count_documents({})
    print('{} logs'.format(count_documents))
    print('Methods:')
    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for m in method:
        methode = logs.count_documents({"method": m})
        print('\tmethod {}: {}'.format(m, methode))
    filter_count = {"method": "GET", "path": "/status"}
    print("{} status check".format(logs.count_documents(filter_count)))