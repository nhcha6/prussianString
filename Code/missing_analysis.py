# import libraries
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


def missing_at_random(county):
    county_merged_df = pd.read_excel("Merged_Data/" + county + "/Merged_Data_" + county + '.xlsx')
    within_data = county_merged_df[county_merged_df['geometry'].notnull()|(county_merged_df['geo_names'])]
    within_data = within_data.drop_duplicates(subset=['loc_id'], keep='first')

    missing = ~county_merged_df.loc_id.isin(within_data.loc_id)
    missing_data = county_merged_df[missing]
    missing_data = missing_data.drop_duplicates(subset=['loc_id'], keep='first')
    print(within_data.shape[0])
    print(missing_data.shape[0])
    county_merged_df = county_merged_df.drop_duplicates(subset=['loc_id'], keep='first')
    print(county_merged_df.shape[0])

missing_at_random('koenigsberg')
