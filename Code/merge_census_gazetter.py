# coding: utf-8

# Script is essentially the same as fraustadt_MeyersGazetter.py except it has been generalised and functionised to work
# for all counties.
# Fraustadt has the `placeID = "10505026"` in the Gazetter.

# Note: This test script builds on the scraped and parsed Meyers Gazetter entry data in the `ordiniertenbuch_continued.ipynb`

"""------------------------------------- INITIALISATION CODE -----------------------------------"""

# import libraries
import os
import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
from tabulate import tabulate
import re
import math

# set working directory path as location of data
wdir = '/Users/nicolaschapman/Documents/PrussianStringMatching/Data/'

# set county name to be run
county = "Fraustadt"
alt_county = "Lissa"

"""---------------------------------- DECLARE FUNCTIONS ----------------------------------"""

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


def levenshtein(seq1, seq2):
    """returns the minimum number of changes (replacement, insertion, deletion) required to convert between two stings.
    code taken from: https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/#:~:text=The%20Levenshtein%20Distance,-This%20method%20was&text=The%20distance%20value%20describes%20the,strings%20with%20an%20unequal%20length."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1, size_y - 1])


def lev_array(unmatched_gazetter_name, unmatched_census_name):
    """
    algorithm which calculates the levenshtein distance for all unmatches census names and gazetter names beginning with
    the same letter. If the ratio of the levenshtein distance to the length of the census_name is less than the a
    certain value (0.3 at the moment) the pair is added to an output array.
    output is list of lists: [[census_name, closest gazzeter_name, levenshtein ratio]]
    """
    levenshtein_array = []
    for census_name in unmatched_census_name:
        if census_name == 'nan':
            continue
        for gazetter_name in unmatched_gazetter_name:
            if gazetter_name[0] != census_name[0]:
                continue
            ldist = levenshtein(gazetter_name, census_name)
            if ldist / len(census_name) < 0.3:
                entry = [census_name, gazetter_name, ldist / len(census_name)]
                levenshtein_array.append(entry)
    return levenshtein_array


""" ----------------------------------- LOAD GAZETTE DATA AND CLEAN FOR MERGE -----------------------------------"""

def gazetter_data(county, alt_county=None):
    # load in json file of (combinded) Gazetter entries
    # commented out as saving of df means it need only run once

    # file_path = os.path.join(wdir, 'Matching', 'json_merge.json')
    # with open(file_path, 'r', encoding="utf8") as json_file:
    #     data = json.load(json_file)
    #     df = json_normalize(data)
    #     print(f'The number of entries in Meyer Gazetter is: {df.shape[0]}')
    #
    # # save df to file so that we do not need to load json file again.
    # df.to_pickle(wdir+"df_pickle")

    # load saved data frame
    df = pd.read_pickle(wdir + "df_pickle")
    print(f'The number of entries in Meyer Gazetter is: {df.shape[0]}')

    # First, let's see how many cities have Fraustadt in any of the columns (drop duplicates created by this technique).
    # next check which rows have Fraustadt in any of the "abbreviation columns"
    # search for lissa too, because fraustadt was split to Lissa and Frastadt after the census but before the Meyers
    # Gazetter data was compiled.
    # two methods for extraction are seen, one that uses only the Kr column for the desired country and one that searches
    # for any exact reference to the county in any column
    # df_fraustadt = df[(df.values=="Fraustadt")|(df.values=="Lissa")]
    if alt_county != None:
        df_county = df[df['Kr'].str.startswith(county, na=False) | df['Kr'].str.startswith(alt_county, na=False) | df[
        'AG'].str.startswith("Fraustadt", na=False) | df['AG'].str.startswith("Lissa", na=False)]
    else:
        df_county = df[df['Kr'].str.startswith(county, na=False)  | df['AG'].str.startswith("Fraustadt", na=False)]

    # duplicated columns: keep only first
    df_county = df_county.groupby(['id']).first().reset_index()
    # create column for merge
    df_county['merge_name'] = df_county['name'].str.strip()
    df_county['merge_name'] = df_county['merge_name'].str.lower()
    print(f'The number of locations in the Gazetter with "{county}" in any of their columns is {df_county.shape[0]}')
    # rename name column to indicate gazetter
    df_county.rename(columns={'name': 'name_gazetter'}, inplace=True)

    # extract base name, eg: Zedlitz from Nieder Zedlitz
    df_county['base_merge_name'] = df_county['merge_name'].str.extract(r'.*\s(.*)', expand=True)

    # Let's define a dictionary to match the [Meyers gazetter types](https://www.familysearch.org/wiki/en/Abbreviation_Table_for_Meyers_Orts_und_Verkehrs_Lexikon_Des_Deutschen_Reichs) to the three classes `stadt, landgemeinde, gutsbezirk`.
    dictionary_types = {"HptSt.": "stadt",  # Hauptstadt
                        "KrSt.": "stadt",  # Kreisstadt
                        "St.": "stadt",  # Stadt
                        "D.": "landgemeinde",  # Dorf
                        "Dr.": "landgemeinde",  # Dörfer
                        "Rg.": "landgemeinde",  # Rittergut
                        "G.": "gutsbezirk",  # Gutsbezirk (but also Gericht)
                        "FG": "gutsbezirk",  # Forstgutsbezirk
                        "Gutsb.": "gutsbezirk"  # Gutsbezirk
                        }


    # Next we need to create a column that entails the "translated" type.
    # Note: I rely on the follwing order `stadt > landgemeinde > gutsbezirk` for classification. For instance, if we have
    # a location that has the types `G.` and `D.`, I will attribute the type `landgemeinde` to the location. Also note that
    # `stadt > landgemeinde > gutsbezirk` is the reverse alpahbetical ordering!
    def check_type(string, dictionary=dictionary_types):
        """
        This is a helper that takes the type dictionary and returns
        the correct class of the string.
        """
        matches = []
        # get all matches of string
        for key in list(dictionary.keys()):
            regex = key.replace(".", "\.").lower()
            # only tag "G." if it is not preceeded by an "r" (i.e. if "Rg.")
            if key == "G.":
                regex = '(?<!r)' + regex
            match = re.search(regex, string.lower())
            if match:
                matches.append(key)
                # if there exists a match return the correct class label (in accordance with ordering)
        if matches:
            classes = []
            for match in matches:
                classes.append(dictionary[match])
            return sorted(classes, reverse=True)[0]


    # test the check_type function
    """
    testset = ["D. u. Rg.", 
               "GutsB.", 
               "D. u. Dom. (aus: Mittel, Nieder u. Ober D.)", 
               "St.", 
               "St. u. D.", # to check if ordering works
               "D. u. GutsB."   # to check if ordering works
              ]
    
    for string in testset:
        print("\n" + string)
        print(check_type(string))
    """

    # Now that we are all set let's create the column `class`. And inspect if it worked
    # make sure column Type has only string values
    df_county['Type'] = df_county['Type'].astype(str)
    # make apply check_type() function
    df_county['class_gazetter'] = df_county['Type'].apply(check_type)
    # # check results
    # print("checking the class_gazette column has been successfully and accurately added")
    # print(df_fraustadt[['id', 'name_gazetter', 'lat', 'lng','Type', 'merge_name', 'class_gazetter']].head())
    return df_county

# df_county = gazetter_data(county,alt_county)
#
# # split df_fraustadt into lat/long == null and lat/long!=null
# df_county_latlong = df_county[df_county['lat'] != 0]
# df_county_null = df_county[df_county['lat'] == 0]

df_fraustadt = gazetter_data(county,alt_county)

# split df_fraustadt into lat/long == null and lat/long!=null
df_fraustadt_latlong = df_fraustadt[df_fraustadt['lat'] != 0]
df_fraustadt_null = df_fraustadt[df_fraustadt['lat'] == 0]

""" ----------------------------LOAD CLEANED CENSUS DATA AND APPLY STRING CLEANING -----------------------------------"""

# Load Posen-Fraustadt-kreiskey-134.xlsx` file we want to match with Gazetter entries. Clean file before merge!
# !! To-Do: Improve string split Pattern
# Improve on split pattern for locations with appendix to accomodate "all" cases:
# `pattern = \sa\/|\sunt\s|\sa\s|\sunterm\s|\si\/|\si\s|\sb\s|\sin\s|\sbei\s|\sam\s|\san\s`

# load saved data frame
df_census = pd.read_pickle(wdir+"census_df_pickle")
df_master = df_census[df_census['county']==county.lower()]

# upload cleaned data
#df_master = pd.read_excel(os.path.join(wdir, 'PrussianCensus1871/Fraustadt', 'Posen-Fraustadt-kreiskey-134.xlsx'))
# rename columns
df_master.rename(columns={"regbez": "province",
                          "kreiskey1871": "province_id",
                          "county": "district",
                          'type': "class",
                          "page": "type_id",
                          "running_number": "loc_id",
                          "locname": "orig_name"
                          }, inplace=True)
# drop rows with "a) Stadtgemeinden" and "b) Landgemeinden" as these are headings and not data
#df_master = df_master[~df_master['orig_name'].isin(['a) Stadtgemeinden', 'b) Landgemeinden', "c) Gutsbezirke"])]
# now we need to clean location names
df_master['name'] = df_master['orig_name']
# extract alternative writing of location name: in parantheses after =
df_master['alt_name'] = df_master['name'].str.extract(r'.+\(=(.*)\).*', expand=True)
# extract alternative name without appendixes such as bei in
df_master['suffix'] = df_master['name'].str.extract(r'.+\(([a-zA-Z\.-]+)\).*', expand=True)
# replace '-' with "\s" in suffix, 'niederr' with 'nieder'  (and special case nied. with nieder)
df_master['suffix'] = df_master['suffix'].str.replace(r'-', ' ')
df_master['suffix'] = df_master['suffix'].str.replace('Nied.', 'Nieder')
df_master['suffix'] = df_master['suffix'].str.replace('Niederr', 'Nieder')
# drop substring after parantheses or ',' from name
df_master['name'] = df_master['name'].str.replace(r'\(.+', '')
df_master['name'] = df_master['name'].str.replace(r',.+', '')
# account for cases with appendixes such as Neuguth bei Reisen and Neuguth bei Fraustadt
pattern = '\sa\/|\sunt\s|\sa\s|\sunterm\s|\si\/|\si\s|\sb\s|\sin\s|\sbei\s|\sam\s|\san\s'
df_master[['temp_name', 'appendix']] = df_master['name'].str.split(pattern, expand=True)
# attribute more restrictive name (i.e. with appendix) to `alt_name` if there exists an appendix
df_master.loc[df_master['appendix'].notnull(), 'alt_name'] = df_master.loc[df_master['appendix'].notnull(), 'name']
# attribute more restrictive name (i.e. with appendix) to `alt_name` if there exists an appendix
df_master.loc[df_master['appendix'].notnull(), 'name'] = df_master.loc[df_master['appendix'].notnull(), 'temp_name']
df_master.drop(columns=["temp_name"], inplace=True)
# strip all [name, alt_name, suffix] of white spaces
# df_master.replace(np.nan, '', regex=True)
for c in ['name', 'alt_name', 'suffix', 'appendix']:
    df_master[c] = df_master[c].str.strip()
    df_master[c] = df_master[c].str.lower()
# concate 'suffix' and 'name'
df_master.loc[df_master['suffix'].notnull(), 'alt_name'] = df_master.loc[
    df_master['suffix'].notnull(), ['suffix', 'name']].apply(lambda x: ' '.join(x), axis=1)
print(f'Number of locations in master file equals {df_master.shape[0]}')
# Check if all went to plan for the distribution of appendices and suffices
# print(df_master[df_master['appendix'].notnull()].head())
# print(df_master[df_master['suffix'].notnull()].head())


"""----------------------------------- MERGE THE TWO DATA FRAMES -----------------------------------"""

# Now let's try out the merege in a 4-step procedure. This procedure is iterated through twice, once for gazetter entries
# that have a valid lat-long, and one for entries that do not.
# 1. merge on the "more restrivtive" `alt_name` that takes into considerations suffixes such as "Nieder" and the `class` label
# 2. "non-matched" locations will be considered in a second merge based on the location `name` which is the location name without any suffixes and the `class` label
# 3. "non-matched" locations will be considered in a third merge based on "more restrivtive" `alt_name` **but not** on `class` label
# 4. "non-matched" locations will be considered in a fourth merge based on `name` **but not** on `class` label
# 5. "non-matched" locations will be considered in a fourth merge based only on the most basic form of both names.

#  1.)
columns = list(df_master.columns)
df_fraustadt_latlong = df_fraustadt_latlong.assign(merge_round=1)
print("Merging if detailed census name and class match a gazetter entry with location data")
df_join = merge_STATA(df_master, df_fraustadt_latlong, how='left', left_on=['alt_name', 'class'],
                      right_on=['merge_name', 'class_gazetter'])
# set aside merged locations
df_merged1 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 2.)
df_fraustadt_latlong = df_fraustadt_latlong.assign(merge_round=2)
print("Merging if simplified census name and class match a gazetter entry with location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_latlong, how='left', left_on=['name', 'class'],
                      right_on=['merge_name', 'class_gazetter'])
# set aside merged locations
df_merged2 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 3.)
df_fraustadt_latlong = df_fraustadt_latlong.assign(merge_round=3)
print("Merging if detailed census name matches a gazetter entry with location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_latlong, how='left', left_on='alt_name', right_on='merge_name')
# set aside merged locations
df_merged3 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 4.)
df_fraustadt_latlong = df_fraustadt_latlong.assign(merge_round=4)
print("Merging if simplified census name matches a gazetter entry with location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_latlong, how='left', left_on='name', right_on='merge_name')
# set aside merged locations
df_merged4 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 5.)
df_fraustadt_latlong = df_fraustadt_latlong.assign(merge_round=5)
print("Merging if simplified census name matches simplified gazetter entry with location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_latlong, how='left', left_on='name', right_on='base_merge_name')
# set aside merged locations
df_merged5 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# Repeat for gazetter entries with null lat-long just for clarity of a match
#  1.)
df_fraustadt_null = df_fraustadt_null.assign(merge_round=1)
print("Merging if detailed census name and class match a gazetter entry WITHOUT location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_null, how='left', left_on=['alt_name', 'class'],
                      right_on=['merge_name', 'class_gazetter'])
# set aside merged locations
df_merged6 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 2.)
df_fraustadt_null = df_fraustadt_null.assign(merge_round=2)
print("Merging if simplified census name and class match a gazetter entry WITHOUT location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_null, how='left', left_on=['name', 'class'],
                      right_on=['merge_name', 'class_gazetter'])
# set aside merged locations
df_merged7 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 3.)
df_fraustadt_null = df_fraustadt_null.assign(merge_round=3)
print("Merging if detailed census name matches a gazetter entry WITHOUT location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_null, how='left', left_on='alt_name', right_on='merge_name')
# set aside merged locations
df_merged8 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 4.)
df_fraustadt_null = df_fraustadt_null.assign(merge_round=4)
print("Merging if simplified census name matches a gazetter entry WITHOUT location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_null, how='left', left_on='name', right_on='merge_name')
# set aside merged locations
df_merged9 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 5.)
df_fraustadt_null = df_fraustadt_null.assign(merge_round=5)
print("Merging if simplified census name matches simlified gazetter entry WITHOUT location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_null, how='left', left_on='name', right_on='base_merge_name')
# set aside merged locations
df_merged10 = df_join[df_join['_merge'] == 'both']

# concat all dataFrames Dataframes
df_output = pd.concat(
    [df_merged1, df_merged2, df_merged3, df_merged4, df_merged5, df_merged6, df_merged7, df_merged8, df_merged9,
     df_merged10], ignore_index=True)

# How well did we do?
# Note: We now do not consider duplicates but compare to the original excel-file entries
exact_match_perc = (df_master.shape[0] - df_join[df_join["_merge"] == "left_only"].shape[0]) / df_master.shape[0] * 100

# !! To-Do: Eliminate Duplicates: duplicates currently only occur when two matches are found on the same round of
# a merge. Gazetter entries without location data are only considered for the municipalities in the census that do not
# find a match that has location data.

"""---------------------------- ANALYSIS OF UNMATCHED ENTRIES - LEVENSHTEIN DISTANCE ---------------------------"""

# Meyers Gazetter: Which locations were not matched?
# Finally, we would like to know which Gazetter entries are not matched.
id_gazetter = set(df_fraustadt['id'].values)
id_merge = set(df_output['id'].values)
diff = id_gazetter - id_merge
df_remainder = df_fraustadt[df_fraustadt['id'].isin(diff)]
df_remainder.to_excel(os.path.join(wdir, 'PrussianCensus1871/Fraustadt', 'Gazetter_Fraustadt_Lissa_Remainder.xlsx'),
                      index=False)

# extract unmatched names from census data:
unmatched_census_df = df_join[df_join['_merge'] == 'left_only']
unmatched_census_df = unmatched_census_df[columns]
unmatched_name_census = unmatched_census_df["name"]
unmatched_altname_census = unmatched_census_df["alt_name"].astype(str)

# extract entries in gazetter data:
unmatched_name_gazetter = df_fraustadt['merge_name']
unmatched_base_name_gazetter = df_fraustadt['base_merge_name'].astype(str)

# call levenshtein comparison function
levenshtein_matches = lev_array(unmatched_name_gazetter, unmatched_name_census)
levenshtein_matches += lev_array(unmatched_name_gazetter, unmatched_altname_census)
levenshtein_matches += lev_array(unmatched_base_name_gazetter, unmatched_name_census)

# convert list of lists to data frame
unmatched_census_df = unmatched_census_df.assign(lev_match=unmatched_census_df['name'])
for match in levenshtein_matches:
    unmatched_census_df.loc[unmatched_census_df['lev_match'] == match[0], 'lev_match'] = match[1]

""" ------------------------------- CONDUCT MERGE ROUNDS BASED ON LEVENSTEIN DISTANCE ---------------------------- """

# lev_merge_df = merge_STATA(unmatched_census_df, df_remainder, how='left', left_on='lev_match', right_on='merge_name')

# column data has been updated to inculde lev_match so redeclare it
columns = list(unmatched_census_df.columns)

# merge round 5 for levenshtein merge
df_fraustadt_latlong = df_fraustadt_latlong.assign(merge_round=6)
df_fraustadt_null = df_fraustadt_null.assign(merge_round=6)

# follow the same iterative procedure as before, except using 'lev_match' instead of 'name' or 'alt_name'
# first check gazetter entries with location data.
#  1.) Merge if class and levenshtein distance match
print(
    "Merging if name matches via levenshtein distance algortithm and class matches a gazetter entry with location data")
df_join = merge_STATA(unmatched_census_df, df_fraustadt_latlong, how='left', left_on=['lev_match', 'class'],
                      right_on=['merge_name', 'class_gazetter'])
# set aside merged locations
df_lev_merge1 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 2.) Merge if levenshtein distance only matches
print("Merging if name matches via levenshtein distance algortithm a gazetter entry with location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_latlong, how='left', left_on='lev_match', right_on='merge_name')
# set aside merged locations
df_lev_merge2 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# check unmatched entries against non-location gazetter data
#  1.) Merge if class and levenshtein distance match
print(
    "Merging if name matches via levenshtein distance algortithm and class matches a gazetter entry WITHOUT location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_null, how='left', left_on=['lev_match', 'class'],
                      right_on=['merge_name', 'class_gazetter'])
# set aside merged locations
df_lev_merge3 = df_join[df_join['_merge'] == 'both']
# select locations without a match
df_nomatch = df_join[df_join['_merge'] == 'left_only']
df_nomatch = df_nomatch[columns]

# 2.) Merge if levenshtein distance only matches
print("Merging if name matches via levenshtein distance algortithm a gazetter entry WITHOUT location data")
df_join = merge_STATA(df_nomatch, df_fraustadt_null, how='left', left_on='lev_match', right_on='merge_name')

# add to output, but also write to output dedicated to matched made by levenshtein merge.
# write file to disk
lev_merge_df = pd.concat([df_lev_merge1, df_lev_merge2, df_lev_merge3, df_join], ignore_index=True)
df_output = pd.concat([df_output, lev_merge_df], ignore_index=True, sort=False)

# drop duplicates from output arbitratilly. !! Need a method.
df_output_nodups = df_output.drop_duplicates(subset=['loc_id'], keep='last')

# prepare for total output write to file
df_output.drop(columns=["_merge"], inplace=True)
df_output.sort_values(by="loc_id", inplace=True)
df_output.drop(columns=["lev_match"], inplace=True)
df_output.to_excel(os.path.join(wdir, 'PrussianCensus1871/Fraustadt', 'Posen-Fraustadt-kreiskey-134-merged.xlsx'),
                   index=False)

# prepare for levenshtein matches write to file
lev_merge_df.drop(columns=["_merge"], inplace=True)
lev_merge_df.drop(columns=["lev_match"], inplace=True)
lev_merge_df.sort_values(by="loc_id", inplace=True)
lev_merge_df.to_excel(
    os.path.join(wdir, 'PrussianCensus1871/Fraustadt', 'Posen-Fraustadt-kreiskey-134-lev_match_merge.xlsx'),
    index=False)

""" ------------------------------------ CALCULATE QUALITY STATISTICS ----------------------------------------"""
print(f'''\n{exact_match_perc:.2f}% of locations have match with the same name''')
match_perc = 100 * (df_output_nodups.shape[0] - df_output_nodups[df_output_nodups['id'].isnull()].shape[0]) / \
             df_output_nodups.shape[0]
print(f'''\n{match_perc:.2f}% of locations were matched when levenshtein distance was considered''')
loc_perc = 100 * (df_output_nodups.shape[0] - df_output_nodups[df_output_nodups['lat'] == 0].shape[0]) / \
           df_output_nodups.shape[0]
print(f'''\n{loc_perc:.2f}% of locations were matched to geocode data''')
round1_perc = 100 * df_output_nodups[df_output_nodups['merge_round'] == 1].shape[0] / df_output_nodups.shape[0]
print(f'''\n{round1_perc:.2f}% of locations were matched by classification and the exact name''')
round2_perc = 100 * df_output_nodups[df_output_nodups['merge_round'] == 2].shape[0] / df_output_nodups.shape[0]
print(f'''\n{round2_perc:.2f}% of locations were matched by classification and the simplified name''')
round3_perc = 100 * df_output_nodups[df_output_nodups['merge_round'] == 3].shape[0] / df_output_nodups.shape[0]
print(f'''\n{round3_perc:.2f}% of locations were matched by exact name only''')
round4_perc = 100 * df_output_nodups[df_output_nodups['merge_round'] == 4].shape[0] / df_output_nodups.shape[0]
print(f'''\n{round4_perc:.2f}% of locations were matched by simplified name only''')
round5_perc = 100 * df_output_nodups[df_output_nodups['merge_round'] == 5].shape[0] / df_output_nodups.shape[0]
print(
    f'''\n{round5_perc:.2f}% of locations were matched by simplifying the name from both the census and meyers gazetter''')

# import csv to data frame, edit it and output new data frame to csv
df_merge_details = pd.read_excel(os.path.join(wdir, 'PrussianCensus1871/', 'MergeDetails.xlsx'))

# if county has never been run, add new entry
if df_merge_details[df_merge_details['county'] == county].empty:
    new_county = df_merge_details.loc[0]
    new_county = new_county.replace(new_county['county'], county)
    df_merge_details = df_merge_details.append(new_county)

df_merge_details.loc[df_merge_details['county'] == county, 'exact_match_perc'] = exact_match_perc
df_merge_details.loc[df_merge_details['county'] == county, 'match_perc'] = match_perc
df_merge_details.loc[df_merge_details['county'] == county, 'loc_perc'] = loc_perc
df_merge_details.loc[df_merge_details['county'] == county, 'round1_perc'] = round1_perc
df_merge_details.loc[df_merge_details['county'] == county, 'round2_perc'] = round2_perc
df_merge_details.loc[df_merge_details['county'] == county, 'round3_perc'] = round3_perc
df_merge_details.loc[df_merge_details['county'] == county, 'round4_perc'] = round4_perc
df_merge_details.loc[df_merge_details['county'] == county, 'round5_perc'] = round5_perc

lev_typo = ""
for match in levenshtein_matches:
    lev_typo += match[0] + " - " + match[1] + " | "
df_merge_details.loc[df_merge_details['county'] == county, 'lev_typo'] = lev_typo

df_merge_details.to_excel(os.path.join(wdir, 'PrussianCensus1871/', 'MergeDetails.xlsx'), index=False)

