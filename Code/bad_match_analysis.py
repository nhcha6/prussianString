import os
import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
from tabulate import tabulate
import re
import math
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None, "display.max_columns", None)

WORKING_DIRECTORY = '/Users/nicolaschapman/Documents/PrussianStringMatching/Data/'

# load saved data frame
df_census= pd.read_pickle(WORKING_DIRECTORY+"census_df_pickle")

# load merge data
df_merge_data = pd.read_excel(os.path.join(WORKING_DIRECTORY,'Output', 'MergeDetails.xlsx'))

for county in df_merge_data['county']:
    df_county = df_census[df_census['county'] == county]
    df_merge_data.loc[df_merge_data['county']==county,'county_size'] = df_county.shape[0]

df_bad_match = df_merge_data[df_merge_data['match_perc']<80]

df_bad_match_small = df_bad_match[df_bad_match['county_size']<20].reset_index()

df_bad_match_big = df_bad_match[df_bad_match['county_size']>19].reset_index()


df_bad_match_small.to_excel(os.path.join(WORKING_DIRECTORY, 'BadMatches', 'smallBadMatches.xlsx'),index=False)
df_bad_match_big.to_excel(os.path.join(WORKING_DIRECTORY, 'BadMatches', 'bigBadMatches.xlsx'),index=False)

df_merge_data.to_excel(os.path.join(WORKING_DIRECTORY, 'Output', 'MergeDetails.xlsx'),index=False)

for county in df_bad_match_big['county']:
    df_merged = pd.read_excel(os.path.join(WORKING_DIRECTORY,'Output', county, 'Merged_Data_'+county+'.xlsx'))
    df_merged_nomatch = df_merged[df_merged['id'].isnull()]
    df_merged_nomatch.iloc[:,:20].to_excel(os.path.join(WORKING_DIRECTORY, 'BadMatches/BigNoMatch', 'Unmerged_Data_'+county+'.xlsx'),index=False)


