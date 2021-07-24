#!/usr/bin/env python3
"""return the list of names of the home planets of all sentient species"""
import requests


def sentientPlanets():
    """return the list of names of the home planets of all sentient species"""
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url is not None:
        r = requests.get(url).json()
        results = r['results']
        for specie in results:
            if specie['designation'] == "sentient" and specie["homeworld"]:
                homeworld = requests.get(specie["homeworld"]).json()
                planets.append(homeworld["name"])

        url = r["next"]

    return planets
