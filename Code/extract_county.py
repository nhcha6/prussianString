import os
import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
from tabulate import tabulate
import re
import math
import operator

pd.set_option("display.max_rows", None, "display.max_columns", None)

WORKING_DIRECTORY = '/Users/nicolaschapman/Documents/PrussianStringMatching/Data/'

## import quality data and add the counties with <60% match rate.
# bad_match
quality_data = pd.read_excel(os.path.join(WORKING_DIRECTORY, 'Output', 'MergeDetails.xlsx'))
bad_match_df = quality_data[quality_data['match_perc']<70]
bad_match_big_df = bad_match_df[bad_match_df['county_size']>=20]
bad_matches = set()
for bad_match in bad_match_big_df['county']:
    bad_matches.add(bad_match)
print(bad_matches)
print(len(bad_matches))

for bad_match in bad_matches:
    print(f'''\n\n{bad_match}''')
    merged_df = pd.read_excel(os.path.join(WORKING_DIRECTORY, 'BadMatches/FirstBadOutput', bad_match, 'Merged_Data_' + bad_match + '.xlsx'))
    AG_count = {}
    for county in merged_df['AG']:
        if county in AG_count.keys():
            AG_count[county]+=1
        else:
            AG_count[county] = 1
    for county in merged_df['Kr']:
        if county in AG_count.keys():
            AG_count[county]+=1
        else:
            AG_count[county] = 1
    #AG_count = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    sorted_AG = sorted(AG_count.items(), key=lambda x: x[1], reverse=True)
    print(sorted_AG)
