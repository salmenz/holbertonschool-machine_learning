#!/usr/bin/env python3
"""display the upcoming launch with these information"""
import requests

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/"
    launches = requests.get(url).json()
    dict_rocket = {}
    for launch in launches:
        rocket = launch["rocket"]
        url = "https://api.spacexdata.com/v4/rockets/{}/".format(rocket)
        rocketname = requests.get(url).json()["name"]
        if rocketname in dict_rocket.keys():
            dict_rocket[rocketname] += 1
        else:
            dict_rocket[rocketname] = 1

    for rocket in sorted(dict_rocket, key=dict_rocket.get, reverse=True):
        print('{}: {}'.format(rocket, dict_rocket[rocket]))
