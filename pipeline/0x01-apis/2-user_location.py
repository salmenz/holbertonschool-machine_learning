#!/usr/bin/env python3
"""script that prints the location of a specific user"""
import requests
import sys
import time


if __name__ == '__main__':
    r = requests.get(sys.argv[1])

    if r.status_code == 404:
        print("Not found")

    if r.status_code == 200:
        print(r.json()["location"])

    if r.status_code == 403:
        X = r.headers["X-Ratelimit-Reset"]
        X = (int(X) - int(time.time())) / 60
        print("Reset in {} min".format(int(X)))
