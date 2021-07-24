#!/usr/bin/env python3
"""display the upcoming launch with these information"""
import requests


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    launches = requests.get(url).json()
    date = float('inf')
    print(date)
    for launch in launches:
        if date > launch["date_unix"]:
            date = launch["date_unix"]
            x = launches.index(launch)
    name = launches[x]["name"]
    date = launches[x]["date_local"]
    rocket = launches[x]["rocket"]
    url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket)
    rocketname = requests.get(url).json()["name"]
    lu = launches[x]["launchpad"]
    lp = requests.get("https://api.spacexdata.com/v4/launchpads/{}".format(lu))
    launchname = lp.json()["name"]
    launclocale = lp.json()["locality"]

    print('{} ({}) {} - {} ({})'.format(name, date, rocketname,
                                        launchname, launclocale))
