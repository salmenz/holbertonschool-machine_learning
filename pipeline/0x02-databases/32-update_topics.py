#!/usr/bin/env python3
"""changes all topics of a school document based on the name"""



def update_topics(mongo_collection, name, topics):
    """changes all topics of a school document based on the name"""
    values = {'$set': {'topics': topics}}
    mongo_collection.update_many({'name': name}, values)
