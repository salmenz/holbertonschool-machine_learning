#!/usr/bin/env python3
"""returns the list of school having a specific topic"""


def schools_by_topic(mongo_collection, topic):
    """returns the list of school having a specific topic"""

    research = []
    result = mongo_collection.find({'topics': {'$all': [topic]}})
    for res in result:
        research.append(res)

    return research
    