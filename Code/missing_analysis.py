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
import scipy

pd.set_option("display.max_rows", None, "display.max_columns", None)

DATA_HEADERS = ['locname', 'type', 'pop_male', 'pop_female', 'pop_tot', 'protestant', 'catholic', 'other_christ', 'jew',
                'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate']

# set working directory path as location of data
WORKING_DIRECTORY = '/Users/nicolaschapman/Documents/PrussianStringMatching/Data/'


def missing_at_random(county):
    county_merged_df = pd.read_excel(os.path.join(WORKING_DIRECTORY, 'Output/', county, 'Merged_Data_' + county + '.xlsx'))

    # convert all data to proportion of population
    for data in DATA_HEADERS:
        if data in ['pop_tot', 'type', 'locname']:
            continue
        county_merged_df[data] = county_merged_df[data] / county_merged_df['pop_tot']
        county_merged_df.loc[county_merged_df[data] > 1, data] = 1
    # add child to mother ratio:
    county_merged_df['child_per_woman'] = county_merged_df['age_under_ten'] / county_merged_df['pop_female']

    # amalagamation code for certain cities:
    if county in ['trier stadtkreis', 'frankfurt am main', 'liegnitz stadtkreis', 'Communion-Bergamts-Bezirk Goslar']:
        county_merged_df.loc[county_merged_df['geometry'].isnull() & (county_merged_df['geo_names'] == False), 'geo_names'] = True

    within_data = county_merged_df[county_merged_df['geometry'].notnull()|(county_merged_df['geo_names'])]

    within_data = within_data.drop_duplicates(subset=['loc_id'], keep='first')

    missing = ~county_merged_df.loc_id.isin(within_data.loc_id)
    missing_data = county_merged_df[missing]
    missing_data = missing_data.drop_duplicates(subset=['loc_id'], keep='first')

    county_merged_df = county_merged_df.drop_duplicates(subset=['loc_id'], keep='first')

    df_missing, df_county_summary = county_summary(county, county_merged_df, missing_data, within_data)

    return df_missing, df_county_summary

def county_summary(county, county_merged_df, missing_data, within_data):
    num_missing = missing_data.shape[0]
    num_total = county_merged_df.shape[0]
    num_within = within_data.shape[0]

    # declare basic summary data
    temp = {'name': [county], 'num_total': [num_total], 'num_missing': [num_missing], 'num_within': [num_within]}
    df_county_summary = pd.DataFrame(temp)
    df_county_summary['plot_%'] = 100*df_county_summary['num_within']/df_county_summary['num_total']
    try:
        df_county_summary['stadt_plot_%'] = 100*within_data[within_data['class']=='stadt'].shape[0]/county_merged_df[county_merged_df['class']=='stadt'].shape[0]
    except ZeroDivisionError:
        df_county_summary['stadt_plot_%'] = 'NA'
    try:
        df_county_summary['z_manor_plot_%'] = 100*within_data[within_data['class']=='gutsbezirk'].shape[0]/county_merged_df[county_merged_df['class']=='gutsbezirk'].shape[0]
    except ZeroDivisionError:
        df_county_summary['z_manor_plot_%'] = 'NA'
    try:
        df_county_summary['village_plot_%'] = 100*within_data[within_data['class']=='landgemeinde'].shape[0]/county_merged_df[county_merged_df['class']=='landgemeinde'].shape[0]
    except ZeroDivisionError:
        df_county_summary['village_plot_%'] = 'NA'

    df_missing_total = mean_comp(within_data, missing_data, 'total', county)
    df_missing_stadt = mean_comp(within_data[within_data['class']=='stadt'], missing_data[missing_data['class']=='stadt'], 'stadt', county)
    df_missing_manor = mean_comp(within_data[within_data['class']=='gutsbezirk'], missing_data[missing_data['class']=='gutsbezirk'], 'manor', county)
    df_missing_village = mean_comp(within_data[within_data['class']=='landgemeinde'], missing_data[missing_data['class']=='landgemeinde'], 'village', county)

    df_missing = pd.concat([df_missing_total, df_missing_stadt, df_missing_manor, df_missing_village], ignore_index=True)

    return df_missing, df_county_summary

def mean_comp(within_data, missing_data, subset, county):
    df_county_missing = pd.DataFrame({'name': [county], 'subset': subset})

    for data in DATA_HEADERS:
        if data in ['type', 'locname', 'pop_male', 'pop_female']:
            continue
        within_list = list(within_data[data])
        missing_list = list(missing_data[data])
        # within_mean = np.mean(within_list)
        # missing_mean = np.mean(missing_list)
        # t_value = scipy.stats.ttest_ind(within_list, missing_list)[0]
        df_county_missing[data+ '_plotted_mean'] = np.mean(within_list)
        df_county_missing[data+ '_missing_mean'] = np.mean(missing_list)
        df_county_missing[data+ '_t_value'] = scipy.stats.ttest_ind(within_list, missing_list)[0]

    return df_county_missing

def run():
    # read in merged data
    merge_details = pd.read_excel(os.path.join(WORKING_DIRECTORY, 'Output/', 'MergeDetails.xlsx'))
    flag = True
    for county in merge_details['county']:
    #for county in ['trier stadtkreis', 'frankfurt am main', 'liegnitz stadtkreis', 'Communion-Bergamts-Bezirk Goslar', 'altona', 'magdeburg stadtkreis']:
        df_county_missing, df_county_summary = missing_at_random(county)
        if flag:
            df_missing = df_county_missing
            df_summary = df_county_summary
            flag = False
        else:
            df_missing = pd.concat([df_missing, df_county_missing], ignore_index=True)
            df_summary = pd.concat([df_summary, df_county_summary], ignore_index=True)

    df_missing.to_excel(os.path.join(WORKING_DIRECTORY, 'Output','MissingAnalysis.xlsx'),index=False)
    df_summary.to_excel(os.path.join(WORKING_DIRECTORY, 'Output','MappingSummary.xlsx'),index=False)

run()

