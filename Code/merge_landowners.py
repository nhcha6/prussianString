import numpy as np
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from map_details import *
import mapclassify as mc
import os

# set working directory path as location of data
WORKING_DIRECTORY = '/Users/nicolaschapman/Documents/NicMergeData/'

pd.set_option("display.max_rows", None, "display.max_columns", None)

# extract landowners data
landowners_1882_df = pd.read_stata(os.path.join(WORKING_DIRECTORY, 'Eddie', 'land_owners1882.dta'))
landowners_1882_df = pd.read_stata(os.path.join(WORKING_DIRECTORY, 'Eddie', 'land_owners1882.dta'))
landowners_1882_df = pd.read_stata(os.path.join(WORKING_DIRECTORY, 'Eddie', 'land_owners1882.dta'))
kreis_set = set()
for kreis in landowners_1882_df['kreis']:
    kreis_set.add(kreis)
kreis_list = list(kreis_set)

# upload updated census details
    df_census_updated = pd.read_excel(os.path.join(WORKING_DIRECTORY, 'OutputSummary/', 'PrussianCensusUpdated.xlsx'))

for kreis in kreis_set:

