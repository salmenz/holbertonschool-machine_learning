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
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for m in methods:
        num_met = logs.count_documents({"method": m})
        print('\tmethod {}: {}'.format(m, num_met))

    filter_path = {"method": "GET", "path": "/status"}
    num = logs.count_documents(filter_path)
    print("{} status check".format(num))
