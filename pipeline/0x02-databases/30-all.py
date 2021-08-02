#!/usr/bin/env python3
"""lists all documents in a collection"""


def list_all(mongo_collection):
    """lists all documents in a collection"""

    docs = []
    collection = mongo_collection.find()
    for doc in collection:
        docs.append(doc)

    return docs
