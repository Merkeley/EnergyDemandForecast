'''
    File: add_census.py

    Description:
        Add census data to a electricity demand data file
'''

import os
import sys

import requests
from census import Census
from us import states

import numpy as np
import pandas as pd

import datetime as dt
from dateutil import parser


CENSUS_FIELD_NAME = 'B25001_001E'

def main() :
    pop_data = []
    pop_list = []
    if len(sys.argv) != 3:
        print('Usage: add_census <data file1> <state code>')
        sys.exit(1)
    else:
        in_file1 = sys.argv[1]
        state_abrev = sys.argv[2]

    try :
        df1 = pd.read_csv(in_file1)
    except :
        raise

    CENSUS_API_KEY = os.environ.get('CensusDotGovAPIKey')
    
    if len(CENSUS_API_KEY) == 0 :
        print('No API Key defined at CensusDotGovAPIKey')
        sys.exit(1)

    first_date = parser.isoparse(df1.time.min())
    last_date = parser.isoparse(df1.time.max())
    first_year = first_date.year
    last_year = last_date.year

    if last_year > 2018 :
        search_date = 2018
    else :
        search_date = last_year

    # Run through all the years in the range
    for year in range(first_year, search_date+1, 1) :
        c = Census(CENSUS_API_KEY, year=year)
        pop_data.append((year, \
            c.acs5.state(('NAME', CENSUS_FIELD_NAME), states.CA.fips)[0][CENSUS_FIELD_NAME]))

    # Make the resulting list a dictionary for ease of access
    pop_dict = dict(pop_data)

    if last_year > search_date :
        for year in range(search_date+1, last_year+1, 1) :
           pop_dict[year] = 2 * pop_dict[year-1] - pop_dict[year-2]

    # Now run through all the rows in the dataframe and create a list of
    #    populations for the corresponding time period
    for date in df1.time :
        dt_time = parser.isoparse(date)
        pop_list.append(pop_dict[dt_time.year])

    df1[state_abrev+'_pop'] = pop_list

    df1.to_csv(in_file1)


if __name__ == '__main__':
  main()
