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
census_df = pd.read_pickle(wdir+"census_df_pickle")

# don't have a better way to deal with the exception atm
df_county = df_census[df_census['county'] == 'berlin']



