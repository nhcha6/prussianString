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
WORKING_DIRECTORY = 'NicMergeData/'

pd.set_option("display.max_rows", None, "display.max_columns", None)

def merge_landowners(landowners_df, df_census_update):
    # clean kries data
    landowners_df = clean_kreis_data(landowners_df)

    kreis_set = set()
    for kreis in landowners_df['kreis']:
        kreis_set.add(kreis)
    kreis_list = list(kreis_set)
    print(kreis_list)

    flag=True

    count = 0
    for kreis in kreis_list:

        count+=1
        print(count)

        landowners_kreis_df = landowners_df[landowners_df['kreis']==kreis]
        df_census_kreis = filter_census(landowners_kreis_df, df_census_update)

        # clean census data:
        df_census_kreis = clean_census(df_census_kreis)
        landowners_kreis_df = clean_landowner_names(landowners_kreis_df)

        # crude merge
        columns = list(landowners_kreis_df.columns)
        df_join = merge_STATA(landowners_kreis_df, df_census_kreis[df_census_kreis['alt_name'].notnull()],  how='left', left_on='comp_name', right_on='alt_name')
        # set aside merged locations
        df_merge1 = df_join[df_join['_merge'] == 'both']
        # select locations without a match
        df_nomatch = df_join[df_join['_merge'] == 'left_only']
        df_nomatch = df_nomatch[columns]

        # crude merge
        df_join = merge_STATA(df_nomatch, df_census_kreis[df_census_kreis['alt_name'].notnull()], how='left', left_on='simp_name', right_on='alt_name')
        # set aside merged locations
        df_merge2 = df_join[df_join['_merge'] == 'both']
        # select locations without a match
        df_nomatch = df_join[df_join['_merge'] == 'left_only']
        df_nomatch = df_nomatch[columns]

        # crude merge
        df_join = merge_STATA(df_nomatch, df_census_kreis[df_census_kreis['name'].notnull()], how='left', left_on='comp_name', right_on='name')
        # set aside merged locations
        df_merge3 = df_join[df_join['_merge'] == 'both']
        # select locations without a match
        df_nomatch = df_join[df_join['_merge'] == 'left_only']
        df_nomatch = df_nomatch[columns]

        # crude merge
        df_join = merge_STATA(df_nomatch, df_census_kreis[df_census_kreis['name'].notnull()], how='left', left_on='simp_name', right_on='name')
        # set aside merged locations
        df_merge4 = df_join[df_join['_merge'] == 'both']
        # select locations without a match
        df_nomatch = df_join[df_join['_merge'] == 'left_only']
        df_nomatch = df_nomatch[columns]

        # concat all dataFrames Dataframes
        df_combined = pd.concat([df_merge1, df_merge2, df_merge3, df_merge4, df_join], ignore_index=True)

        match_rate = (landowners_kreis_df.shape[0]-df_nomatch.shape[0])/landowners_kreis_df.shape[0]

        merge_dict = {'county': [kreis], 'match_rate': [match_rate]}
        if flag:
            merge_details = pd.DataFrame.from_dict(merge_dict)
            combined_landowners = df_combined
            flag = False
        else:
            merge_temp = pd.DataFrame.from_dict(merge_dict)
            merge_details = pd.concat([merge_details, merge_temp], ignore_index=True)
            combined_landowners = pd.concat([combined_landowners, df_combined], ignore_index=True)

    return combined_landowners, merge_details

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

def clean_census(df_county):

   ########## GENERAL CLEAN CENSUS DATA FOR MATCHING ##########

    # now we need to clean location names
    df_county['name'] = df_county['orig_name']
    df_county['name'] = df_county['name'].replace('ü', 'ue')
    df_county['name'] = df_county['name'].replace('ö', 'oe')

    # sanct needs to be replaced by sankt
    df_county.loc[df_county['orig_name'].str.contains('Sanct'), 'name'] = df_county.loc[df_county['orig_name'].str.contains('Sanct'), 'orig_name'].str.replace('Sanct', 'Sankt')

    # extract alternative writing of location name: in parantheses after =
    df_county['alt_name'] = df_county['name'].str.extract(r'.+\(=(.*)\).*', expand=True)
    # extract alternative name without appendixes such as bei in
    df_county['suffix'] = df_county['name'].str.extract(r'.+\(([a-zA-Z\.-]+)\).*', expand=True)
    # replace '-' with "\s" in suffix, 'niederr' with 'nieder'  (and special case nied. with nieder)
    df_county['suffix'] = df_county['suffix'].str.replace(r'-', ' ')
    df_county['suffix'] = df_county['suffix'].str.replace('Nied.', 'Nieder')
    df_county['suffix'] = df_county['suffix'].str.replace('Niederr', 'Nieder')
    # drop substring after parantheses, ',' or a space from name
    df_county['name'] = df_county['name'].str.replace(r'\(.+', '')
    df_county['name'] = df_county['name'].str.replace(r',.+', '')
    df_county['name'] = df_county['name'].str.replace(r'\s.+', '')

    # account for cases with appendixes such as Neuguth bei Reisen and Neuguth bei Fraustadt
    pattern = '\sa\/|\sunt\s|\sa\s|\sunterm\s|\si\/|\si\s|\sb\s|\sin\s|\sbei\s|\sam\s|\san\s|'
    split = df_county['name'].str.split(pattern, expand=True)
    if len(split.columns) == 2:
        df_county[['temp_name', 'appendix']] = df_county['name'].str.split(pattern, expand=True)
        # attribute more restrictive name (i.e. with appendix) to `alt_name` if there exists an appendix
        df_county.loc[df_county['appendix'].notnull(), 'alt_name'] = df_county.loc[df_county['appendix'].notnull(), 'name']
        # attribute more restrictive name (i.e. with appendix) to `alt_name` if there exists an appendix
        df_county.loc[df_county['appendix'].notnull(), 'name'] = df_county.loc[df_county['appendix'].notnull(), 'temp_name']
        df_county.drop(columns=["temp_name"], inplace=True)
    else:
        df_county['appendix'] = np.nan
        df_county['appendix'] = df_county['appendix'].astype(str)

    # concate 'suffix' and 'name'
    df_county.loc[df_county['suffix'].notnull(), 'alt_name'] = df_county.loc[
        df_county['suffix'].notnull(), ['suffix', 'name']].apply(lambda x: ' '.join(x), axis=1)

    ############ MORE SPECIFIC CLEANING MADE BY ANALYSING MISSED MATCHES ############

    # may need to do this after everything else to ensure other matches are attempted first.

    # for entries with a) or b) etc, extract the last word as the 'name':
    df_county.loc[df_county['orig_name'].str.contains(r'^.\)'), 'name'] = \
    df_county.loc[df_county['orig_name'].str.contains(r'^.\)'), 'orig_name'].str.split().str[-1]
    # (Stadt) similar case:
    df_county.loc[df_county['orig_name'].str.contains(r'\(Stadt\)'), 'name'] = \
    df_county.loc[df_county['orig_name'].str.contains(r'\(Stadt\)'), 'orig_name'].str.split().str[-1]
    df_county.loc[df_county['orig_name'].str.contains('Schöppingen'), 'name'] = \
    df_county.loc[df_county['orig_name'].str.contains('Schöppingen'), 'orig_name'].str.split().str[-1]

    # update alt_name for those with a dash in it (further alt-name cleaning done later)
    df_county.loc[(df_county["alt_name"].isnull()) & (df_county["name"].str.contains('-')), "alt_name"] = df_county.loc[(df_county["alt_name"].isnull()) & (df_county["name"].str.contains('-')), "name"].str.split('-').str[0]

    ############ FINAL CLEAN BEFORE OUTPUT ############

    # remove spaces in alt-name to account for case of 'kleinvargula' and 'klein vargula' simultaneously.
    df_county['alt_name'] = df_county['alt_name'].str.replace(r'\s', '')
    # also remove commasl
    df_county['alt_name'] = df_county['alt_name'].str.replace(r',', '')

    # strip all [name, alt_name, suffix] of white spaces
    # df_master.replace(np.nan, '', regex=True)
    for c in ['name', 'alt_name', 'suffix', 'appendix']:
        df_county[c] = df_county[c].str.strip()
        df_county[c] = df_county[c].str.lower()

    #print(df_county[['orig_name', 'name', 'alt_name']])
    print(f'Number of locations in census file equals {df_county.shape[0]}')
    return df_county

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


def clean_landowner_names(df_county):
    ########## GENERAL CLEAN CENSUS DATA FOR MATCHING ##########

    # now we need to clean location names
    df_county['simp_name'] = df_county['property']
    df_county.loc[df_county['simp_name']=='', 'simp_name'] = df_county.loc[df_county['simp_name']=='', 'subunit']

    df_county['simp_name'] = df_county['simp_name'].str.replace('Gr.', 'Gross')
    df_county['simp_name'] = df_county['simp_name'].str.replace('Kl.', 'Klein')
    df_county['simp_name'] = df_county['simp_name'].str.replace('Adl.', 'Adlig')
    df_county['simp_name'] = df_county['simp_name'].str.replace(':', '')
    df_county['simp_name'] = df_county['simp_name'].str.replace('+', '')
    df_county['simp_name'] = df_county['simp_name'].replace('ü', 'ue')
    df_county['simp_name'] = df_county['simp_name'].replace('ö', 'oe')

    # sanct needs to be replaced by sankt
    df_county.loc[df_county['property'].str.contains('Sanct'), 'simp_name'] = df_county.loc[
        df_county['property'].str.contains('Sanct'), 'property'].str.replace('Sanct', 'Sankt')

    # extract alternative writing of location name: in parantheses after =
    df_county['comp_name'] = df_county['simp_name'].str.extract(r'.+\((.*)\).*', expand=True)
    # extract alternative name without appendixes such as bei in
    df_county['suffix'] = df_county['simp_name'].str.extract(r'.+,\s(.*)', expand=True)
    # replace '-' with "\s" in suffix, 'niederr' with 'nieder'  (and special case nied. with nieder)
    df_county['suffix'] = df_county['suffix'].str.replace(r'-', ' ')
    df_county['suffix'] = df_county['suffix'].str.replace('Nied.', 'Nieder')
    df_county['suffix'] = df_county['suffix'].str.replace('Niederr', 'Nieder')
    # df_county['suffix'] = df_county['suffix'].str.replace('Gr.', 'Gross')
    # df_county['suffix'] = df_county['suffix'].str.replace('Kl.', 'Klein')
    # df_county['suffix'] = df_county['suffix'].str.replace('Adl.', 'Adlig')
    # df_county['suffix'] = df_county['suffix'].str.replace(':', '')
    # df_county['suffix'] = df_county['suffix'].str.replace('+', '')
    # drop substring after parantheses, ',' or a space from name
    df_county['simp_name'] = df_county['simp_name'].str.replace(r'\(.+', '')
    df_county['simp_name'] = df_county['simp_name'].str.replace(r',.+', '')
    df_county['simp_name'] = df_county['simp_name'].str.replace(r'\s.+', '')

    # account for cases with appendixes such as Neuguth bei Reisen and Neuguth bei Fraustadt
    pattern = '\sa\/|\sunt\s|\sa\s|\sunterm\s|\si\/|\si\s|\sb\s|\sin\s|\sbei\s|\sam\s|\san\s|'
    split = df_county['simp_name'].str.split(pattern, expand=True)
    if len(split.columns) == 2:
        df_county[['temp_name', 'appendix']] = df_county['simp_name'].str.split(pattern, expand=True)
        # attribute more restrictive name (i.e. with appendix) to `alt_name` if there exists an appendix
        df_county.loc[df_county['appendix'].notnull(), 'comp_name'] = df_county.loc[
            df_county['appendix'].notnull(), 'simp_name']
        # attribute more restrictive name (i.e. with appendix) to `alt_name` if there exists an appendix
        df_county.loc[df_county['appendix'].notnull(), 'simp_name'] = df_county.loc[
            df_county['appendix'].notnull(), 'temp_name']
        df_county.drop(columns=["temp_name"], inplace=True)
    else:
        df_county['appendix'] = np.nan
        df_county['appendix'] = df_county['appendix'].astype(str)

    # concate 'suffix' and 'name'
    df_county.loc[df_county['suffix'].notnull(), 'comp_name'] = df_county.loc[
        df_county['suffix'].notnull(), ['suffix', 'simp_name']].apply(lambda x: ' '.join(x), axis=1)

    ############ MORE SPECIFIC CLEANING MADE BY ANALYSING MISSED MATCHES ############

    # may need to do this after everything else to ensure other matches are attempted first.

    # for entries with a) or b) etc, extract the last word as the 'name':
    df_county.loc[df_county['property'].str.contains(r'^.\)'), 'simp_name'] = \
        df_county.loc[df_county['property'].str.contains(r'^.\)'), 'property'].str.split().str[-1]
    # (Stadt) similar case:
    df_county.loc[df_county['property'].str.contains(r'\(Stadt\)'), 'simp_name'] = \
        df_county.loc[df_county['property'].str.contains(r'\(Stadt\)'), 'property'].str.split().str[-1]
    df_county.loc[df_county['property'].str.contains('Schöppingen'), 'simp_name'] = \
        df_county.loc[df_county['property'].str.contains('Schöppingen'), 'property'].str.split().str[-1]

    # update alt_name for those with a dash in it (further alt-name cleaning done later)
    df_county.loc[(df_county["comp_name"].isnull()) & (df_county["simp_name"].str.contains('-')), "comp_name"] = \
    df_county.loc[(df_county["comp_name"].isnull()) & (df_county["simp_name"].str.contains('-')), "simp_name"].str.split('-').str[0]

    ############ FINAL CLEAN BEFORE OUTPUT ############

    # remove spaces in alt-name to account for case of 'kleinvargula' and 'klein vargula' simultaneously.
    df_county['comp_name'] = df_county['comp_name'].str.replace(r'\s', '')
    # also remove commasl
    df_county['comp_name'] = df_county['comp_name'].str.replace(r',', '')

    # strip all [name, alt_name, suffix] of white spaces
    # df_master.replace(np.nan, '', regex=True)
    for c in ['simp_name', 'comp_name', 'suffix', 'appendix']:
        df_county[c] = df_county[c].str.strip()
        df_county[c] = df_county[c].str.lower()

    #print(df_county[['property', 'comp_name', 'simp_name']])

    df_county.rename(columns={"appendix": "app",
                              "suffix": "suff",
                              "province": "prov",
                              }, inplace=True)

    return df_county

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
    landowners_1895_df = pd.read_stata(os.path.join(WORKING_DIRECTORY, 'Eddie', 'land_owners1895.dta'))
    landowners_1907_df = pd.read_stata(os.path.join(WORKING_DIRECTORY, 'Eddie', 'land_owners1907.dta'))

    # upload updated census details
    df_census_updated = pd.read_excel(os.path.join(WORKING_DIRECTORY, 'OutputSummary/', 'PrussianCensusUpdated.xlsx'))

    # 1882
    combined_landowners, merge_details = merge_landowners(landowners_1882_df, df_census_updated)
    combined_landowners.to_excel(os.path.join(WORKING_DIRECTORY, 'LandownersOutput', 'Combined_Landowners_1882.xlsx'),index=False)
    merge_details.to_excel(os.path.join(WORKING_DIRECTORY, 'LandownersOutput', 'Merge_Details_1882.xlsx'),index=False)

    # 1895
    combined_landowners, merge_details = merge_landowners(landowners_1895_df, df_census_updated)
    combined_landowners.to_excel(os.path.join(WORKING_DIRECTORY, 'LandownersOutput', 'Combined_Landowners_1895.xlsx'),index=False)
    merge_details.to_excel(os.path.join(WORKING_DIRECTORY, 'LandownersOutput', 'Merge_Details_1895.xlsx'),index=False)

    # 1907
    combined_landowners, merge_details = merge_landowners(landowners_1907_df, df_census_updated)
    combined_landowners.to_excel(os.path.join(WORKING_DIRECTORY, 'LandownersOutput', 'Combined_Landowners_1907.xlsx'),index=False)
    merge_details.to_excel(os.path.join(WORKING_DIRECTORY, 'LandownersOutput', 'Merge_Details_1907.xlsx'),index=False)




