'''
    File: eia_query.py

    Description:
        Collect data from the Energy Information Administration on 
        production and store it in an SQL database.

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
from collections import defaultdict

import datetime as dt
from dateutil import parser
import requests
import json

from dsky_query import get_dsky_temps

# Define strings that will be used later
SERIES_REQUEST='http://api.eia.gov/series/?series_id={0}&api_key={1}&out=json'
GEOSET_REQUEST='http://api.eia.gov/geoset/?geoset_id={0}&regions={2}&api_key={1}&out=json'

# This is the datafile that contains the entire energy demand
DATA_FILE = '/Users/maboals/Downloads/EBA.txt'

# Dataset IDS
CALIFORNIA_ELECTRIC_DEMAND='EBA.CAL-ALL.D.H'
USA_LOWER_48_ELECTRIC_DEMAND='EBA.US48-ALL.D.H'
CALIFORNIA_INDEPENDENT_SYSTEM_OPERATOR='EBA.CISO-ALL.D.H'
NORTHERN_CALIFORNIA_BALANCING_AUTHORITY='EBA.BANC-ALL.D.H'
CALIFORNIA_POPULATION='SEDS.TPOPP.CA.A'

CALIFORNIA_DEMAND = [CALIFORNIA_ELECTRIC_DEMAND]

#
# Define locations for major cities in CA
#
location_dict = {'Los Angeles': (34.052 , -118.244),
    'San Diego': (32.716, -117.165),
    'San Jose': (37.339, -121.895),
    'San Francisco': (37.775, -122.419),
    'Sacramento': (38.5816, -121.4944),
    'Fresno': (36.7378, -119.7871)}
#
# For Net generation
#
# NORTHERN_CALIFORNIA_BALANCING_AUTHORITY='EBA.BANC-ALL.D.H'

def make_serieslist_from_file(file) :
    '''
        Function: make_serieslist_from_file
            Read a bulk datafile from EIA and extract the series ID's

        Argumenst:
            file - string containing the full path to the datafile

        Return:
            list of series_id's from the file
    '''
    series_list = []
    # Open the specified datafile and read the json data
    try:
        fp=open(file, 'r')

        for j_line in fp :
            # For each entry check for a series id
            js = json.loads(j_line)

            if 'series_id' in js.keys() :
                # Add the series id to the list
                series_list.append(js['series_id'])
    except OSError as err :
        print('OS error: {0}'.format(err))
    except :
        print('Unexpected error:', sys.exc_info()[0])
        return None

    return series_list

def get_eia_series(api_key, id, region):
    # Make a request
    response = requests.get(SERIES_REQUEST.format(id, api_key, region))
    if (response.status_code == requests.codes.ok) :
        return response.text
    else :
        return None

def split_date_string(date_str) :
    return parser.isoparse(re.split('Z-', date_str)[0])


def main() :
    temp_data = defaultdict()
    time_list = []

    # Get the api key
    EIA_API_KEY = os.environ.get('EIAAPIKey')
    DRKSKY_API_KEY = os.environ.get('DarkSkyAPIKey')
    '''
    now = dt.datetime(2018, 12, 1)
    print(DRKSKY_API_KEY)
    forecast = get_dsky_forecast(DRKSKY_API_KEY, \
        str(now.timestamp()).split('.')[0], 37.8715, -122.27)
    '''

    # Read the series_id's from the data file
    # j_line = get_eia_series(EIA_API_KEY, 'EBA.D.H', 'US-CA')
    for series in CALIFORNIA_DEMAND :
        j_line = get_eia_series(EIA_API_KEY, series, '')
        js = json.loads(j_line)
        print('Response Keys: ', js.keys())

        # If we received a valid response
        if 'series' in js.keys() :
            # Run through each dictionary in the json to find the data
            for new_series in js['series'] :
                for key in new_series.keys() :
                    if(key != 'data') :
                        print(key, new_series[key])
                    else :
                        print('Data Len: ', len(new_series[key]))

                        # Data found, put it in the spark dataframe
                        dt_time = [
                            split_date_string(y[0])
                            for y in new_series[key]
                            ]

                        demand = [x[1] for x in new_series[key]]
                        xform_data = [
                            (dt_time[idx], demand[idx])
                            for idx in range(len(dt_time)) 
                            ]

                        # Use pandas to create a dataframe
                        demand_df = pd.DataFrame(xform_data,
                            columns=['time','demand'])

                        # Run through all the dates and create a list
                        # of hourly temperatures for the cities of interest
                        last_str_date = ''
                        times = demand_df['time']

                        for time in times :
                            str_date = time.isoformat().split('T')[0]
                            if(str_date != last_str_date) :
                                time_list.append(time.timestamp())
                                last_str_date = str_date

                        for city in location_dict :
                            big_time_list = []
                            big_temp_list = []
                            for day in time_list :
                                hour_list, temp_list = \
                                    get_dsky_temps(DRKSKY_API_KEY,
                                    str(day).split('.')[0],
                                    location_dict[city][0],
                                    location_dict[city][1])
                                big_time_list.extend(hour_list)
                                big_temp_list.extend(temp_list)

                            temp_data[city] = big_temp_list

                        temp_data['time'] = \
                            [dt.datetime.fromtimestamp(x, tz=dt.timezone.utc) 
                                for x in big_time_list]
                        temp_df = pd.DataFrame(temp_data)

                        # Save the data to csv files
                        demand_df.to_csv('elec_demand.csv')
                        temp_df.to_csv('temp_data.csv')

                        # Join the two tables
                        big_df = demand_df.merge(temp_df, how='inner',
                            left_on='time', right_on='time')

        if 'data' in js.keys() :
           print('Data Len: ', len(js['data']))

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()
