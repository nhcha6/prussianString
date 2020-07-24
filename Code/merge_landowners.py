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

def merge_landowners(landowners_df, df_census_update):
    # clean kries data
    landowners_df = clean_kreis_data(landowners_df)

    kreis_set = set()
    for kreis in landowners_df['kreis']:
        kreis_set.add(kreis)
    kreis_list = list(kreis_set)

    for kreis in kreis_list:
        landowners_kreis_df = landowners_df[landowners_df['kreis']==kreis]
        df_census_kreis = filter_census(landowners_kreis_df, df_census_update)

        # crude merge
        df_join = merge_STATA(landowners_kreis_df, df_census_kreis,  how='left', left_on='property', right_on='orig_name')
        # set aside merged locations
        df_lev_merge1 = df_join[df_join['_merge'] == 'both']
        print(df_lev_merge1['property'])
        # select locations without a match
        df_nomatch = df_join[df_join['_merge'] == 'left_only']
        print(df_nomatch['property'])
        break

def filter_census(landowners_kreis_df, df_census_update):
    kreis = landowners_kreis_df['kreis'].iloc[0]
    cleaned_kreis = landowners_kreis_df['cleaned_kreis'].iloc[0]
    alt_kreis = landowners_kreis_df['alt_kreis'].iloc[0]
    print(kreis)
    print(cleaned_kreis)
    print(alt_kreis)
    df_census_kreis = df_census_update[(df_census_update['Kr'].str.contains(cleaned_kreis, na=False)) | (df_census_update['district'].str.contains(cleaned_kreis.lower(), na=False))|(df_census_update['Kr'].str.contains(alt_kreis, na=False))| (df_census_update['district'].str.contains(alt_kreis.lower(), na=False))]
    print(df_census_kreis.shape[0])
    return df_census_kreis

def clean_kreis_data(landowners_df):
    landowners_df['cleaned_kreis'] = landowners_df['kreis'].str.split().str[0]
    landowners_df.loc[landowners_df['kreis'].str.contains('Pr. '),'cleaned_kreis'] = landowners_df['kreis'].str.replace('Pr. ', '')
    landowners_df['cleaned_kreis'] = landowners_df['cleaned_kreis'].str.replace('Ã¶', 'oe')
    landowners_df['cleaned_kreis'] = landowners_df['cleaned_kreis'].str.replace('Ã¼', 'ue')
    landowners_df['cleaned_kreis'] = landowners_df['cleaned_kreis'].str.replace('Ã¤', 'ä')

    # adjust for - and c/k switch
    landowners_df['alt_kreis'] = landowners_df['cleaned_kreis']
    landowners_df.loc[landowners_df['kreis'].str.contains('-'),'cleaned_kreis'] = landowners_df['cleaned_kreis'].str.split('-').str[0]
    landowners_df.loc[landowners_df['alt_kreis'].str.contains('-'),'alt_kreis'] = landowners_df['alt_kreis'].str.split('-').str[-1]
    landowners_df.loc[landowners_df['cleaned_kreis'].str.contains('C'),'alt_kreis'] = landowners_df['cleaned_kreis'].str.replace('C','K')

    # singular errors:
    landowners_df.loc[landowners_df['cleaned_kreis']=='Uckermuende','cleaned_kreis'] = 'Ukermuende'
    landowners_df.loc[landowners_df['cleaned_kreis']=='Lauenberg','cleaned_kreis'] = 'Lauenburg'

    return landowners_df

def merge_STATA(master, using, how='outer', on=None, left_on=None, right_on=None, indicator=True,
                suffixes=('_master', '_using'), drop=None, keep=None, drop_merge=False):
    """
    function that imitates STATA's merge command and initializes most options of pandas DataFrame merge.
    ----
    Requirements: library "tabulate"
    ----
    Parameters:
    master:        Master DataFrame
    using:         Using DataFrame
    how:           Type of merge to be performed: default is set to ‘outer‘ (as in STATA).
                   Other options are equivalent to pd.merge, i.e. {‘left’, ‘right’, ‘inner’}.
    on:            Column or index level names to join on. These must be found in both DataFrames.
    left_on:       Column or index level names to join on in the left (master) DataFrame
    right_on:      Column or index level names to join on in the right (using) DataFrame.
    indicator:     If True (default) adds column “_merge” to output DataFrame with information
                   on the source of each row.
    suffixes:      Suffix to apply to overlapping column names in the left (master) and right
                   (using) side. Default set to {‘_master’, ‘_master’}
    drop:          If specified, rows labeled as either “left_only” (STATA: _merge==1),
                   “right_only” (_merge==2), or “both” (_merge==3) are dropped after the merge.
                   By default, no rows are dropped.
    keep:          If specified, only rows labeled as either “left_only” (STATA: _merge==1),
                   “right_only” (_merge==2), or “both” (_merge==3) are kept after the merge.
                   By default, all rows are kept.
    drop_merge:    Drop column _merge if True
    ----
    Return:        (Merged) Dataframe

    """

    if left_on == None and right_on == None:  # if variables the same in both df
        merge = master.merge(using, how=how, on=on, indicator=indicator, suffixes=suffixes)

    if on == None:
        merge = master.merge(using, how=how, left_on=left_on, right_on=right_on, indicator=indicator, suffixes=suffixes)

    if left_on == None and right_on == None and on == None:
        print("Either ‘on‘ or {‘left_on’, ‘right_on’} have to be defined")

    if indicator == True:
        # define variables needed for
        result = merge["_merge"].value_counts()
        not_matched_master = result["left_only"]  # STATA: _merge==1
        not_matched_using = result["right_only"]  # STATA: _merge==2
        matched = result["both"]  # STATA: _merge==3
        # define "STATA" merge table
        table = [['not matched', '', not_matched_master + not_matched_using],
                 ['', 'from master', not_matched_master],
                 ['', 'from using', not_matched_using],
                 ['matched', '', matched]]
        try:
            print(tabulate(table, headers=['Result', '', '# of obs.'], tablefmt="fancy_grid"))
        except:
            print("Error: Merge table could not be compiled! Please import library tabulate")
    if drop != None:
        merge = merge[~merge['_merge'] == drop]  # drop rows that equal drop expression
        print("Drop if {}".format(drop))

    if keep != None:
        merge = merge[merge['_merge'] == keep]  # keep rows that equal keep expression
        print("Keep if {}".format(keep))

    if drop_merge:
        merge.drop(['_merge'], axis=1, inplace=True)

    return merge

def run_landowners_merge():
    # extract landowners data
    landowners_1882_df = pd.read_stata(os.path.join(WORKING_DIRECTORY, 'Eddie', 'land_owners1882.dta'))
    #landowners_1895_df = pd.read_stata(os.path.join(WORKING_DIRECTORY, 'Eddie', 'land_owners1882.dta'))
    #landowners_1907_df = pd.read_stata(os.path.join(WORKING_DIRECTORY, 'Eddie', 'land_owners1882.dta'))

    # upload updated census details
    df_census_updated = pd.read_excel(os.path.join(WORKING_DIRECTORY, 'OutputSummary/', 'PrussianCensusUpdated.xlsx'))

    merge_landowners(landowners_1882_df, df_census_updated)

run_landowners_merge()




