'''
    File: data_merge.py

    Description:
        This script simply reads in two data files that contain components
    of the whole electricity demand data set that will be used for forecasting
    and joins them together with a common datetime index.









'''

import os
import sys

import numpy as np
import pandas as pd

def main() :
    if len(sys.argv) != 4:
        print('Usage: data_merge <input file1> <input file2> <output file>')
        sys.exit(1)
    else:
        in_file1 = sys.argv[1]
        in_file2 = sys.argv[2]
        out_file = sys.argv[3]

    try :
        df1 = pd.read_csv(in_file1)
        df2 = pd.read_csv(in_file2)
        df1.drop('Unnamed: 0', axis=1, inplace=True)
        df2.drop('Unnamed: 0', axis=1, inplace=True)
        big_df = df1.merge(df2, how='inner', left_on='time', right_on='time')
        big_df.to_csv(out_file)
    except :
        raise


if __name__ == '__main__':
  main()
