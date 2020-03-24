'''
    File: dsky_query.py

    Description:
        Collect data from Dark Sky Weather API
    store it in an SQL database.

    Author: M Boals

    Date Created: 3/5/2020
    Date Modified:

    Update Log:


    Functions Defined:

'''

import os
import sys
import pandas as pd
import numpy as np
import re

import datetime as dt
from dateutil import parser
import requests
import json

from pprint import pprint

# Define strings that will be used later
DARK_SKY_REQUEST='https://api.darksky.net/forecast/{0}/{1},{2},{3}'
#DARK_SKY_REQUEST='https://api.darksky.net/forecast/{0}/{1}, {2}'

def get_dsky_forecast(api_key, date, lat, lng):
    # Make a request
    response = requests.get( \
        DARK_SKY_REQUEST.format(api_key, lat, lng, date))
    # print(DARK_SKY_REQUEST.format(api_key, lat, lng, date))
    if (response.status_code == requests.codes.ok) :
        return response.text
    else :
        return None

def get_dsky_temps(api_key, lat, lng, date) :
    time_list = []
    temp_list = []

    # Get the full weather forecast
    forecast = get_dsky_forecast(api_key, lat, lng, date)

    # Extract the list of temperatures from the json
    if forecast != None :
        j_fcast = json.loads(forecast)
        for entry in j_fcast['hourly']['data'] :
            time_list.append(entry['time'])
            temp_list.append(entry['temperature'])

    # return the tempertures
    return time_list, temp_list

def main() :
    # Get the api key
    DRKSKY_API_KEY = os.environ.get('DarkSkyAPIKey')

    now = dt.datetime(2018, 12, 1)
    print(DRKSKY_API_KEY)
    times, temps = get_dsky_temps(DRKSKY_API_KEY, \
        str(now.timestamp()).split('.')[0], 37.8715, -122.27)

    print(times[:10], temps[:10])

if __name__ == '__main__':
  main()
