#!/usr/bin/env python3
"""return the list of ships that can hold a given number of passengers"""
import requests


def availableShips(passengerCount):
    """return the list of ships that can hold a given number of passengers"""
    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []

    while url is not None:
        r = requests.get(url).json()
        results = r['results']
        for ship in results:
            p = ship['passengers']
            p = p.replace(',', '')
            if p.isnumeric() and int(p) >= passengerCount:
                ships.append(ship['name'])
        url = r['next']

    return ships
