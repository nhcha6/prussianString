
# coding: utf-8

# ## Match Locations in "Landkreis" Fraustadt to Meyers Gazetter entries
# Fraustadt has the `placeID = "10505026"` in the Gazetter.
# 
# Note: This test script builds on the scraped and parsed Meyers Gazetter entry data in the `ordiniertenbuch_continued.ipynb`

# In[3]:


import os
import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
from tabulate import tabulate
import re


# In[1]:


def merge_STATA(master, using, how='outer', on=None, left_on=None, right_on=None, indicator=True,
                suffixes=('_master','_using'), drop=None, keep=None, drop_merge=False):
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

    if left_on == None and right_on==None: # if variables the same in both df
        merge = master.merge(using, how=how, on=on, indicator=indicator, suffixes=suffixes)

    if on == None:
        merge = master.merge(using, how=how, left_on=left_on, right_on=right_on, indicator=indicator, suffixes=suffixes)

    if left_on == None and right_on==None and on == None:
        print("Either ‘on‘ or {‘left_on’, ‘right_on’} have to be defined")

    if indicator == True:
        # define variables needed for
        result = merge["_merge"].value_counts()
        not_matched_master = result["left_only"] # STATA: _merge==1
        not_matched_using = result["right_only"] # STATA: _merge==2
        matched = result["both"] # STATA: _merge==3
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
        merge = merge[~merge['_merge'] == drop] # drop rows that equal drop expression
        print("Drop if {}".format(drop))

    if keep != None:
        merge = merge[merge['_merge'] == keep] # keep rows that equal keep expression
        print("Keep if {}".format(keep))

    if drop_merge:
        merge.drop(['_merge'], axis=1, inplace=True)

    return merge


# In[2]:


# set working directory path
wdir = '/Users/david/Dropbox/WittenbergerOrdiniertenbuch/'
# load in json file of (combinded) Gazetter entries
file_path = os.path.join(wdir, 'Matching', 'json_merge.json')
with open(file_path, 'r', encoding="utf8") as json_file:  
    data = json.load(json_file)
    df = json_normalize(data)
    print(f'The number of entries in Meyer Gazetter is: {df.shape[0]}')


# First check if `Fraustadt` was indeed successfully scraped

# In[4]:


df[df['id']=='10505026']


# First, let's see how many cities have Fraustadt in any of the columns (drop duplicates created by this technique). 

# In[25]:


# next check which rows have Fraustadt in any of the "abbreviation columns"
df_fraustadt = df[(df.values=="Fraustadt")|(df.values=="Lissa")] 


# # To-Do: Improve "Landkreis" Selection
# Find a better way to select the correct "Landkreis". For instance, we want to account for cases as [Storchnest](https://www.meyersgaz.org/place/20888081) for which the `Kr` and `AG` is `Lissa B. Posen` and not `Lissa`. Need to work with substrings for value selection!

# In[26]:


# duplicated columns: keep only first
df_fraustadt = df_fraustadt.groupby(['id']).first().reset_index()
# create column for merge 
df_fraustadt['merge_name'] = df_fraustadt['name'].str.strip()
df_fraustadt['merge_name'] = df_fraustadt['merge_name'].str.lower()
print(f'The number of locations in the Gazetter with "Fraustadt" in any of their columns is {df_fraustadt.shape[0]}')
# rename name column to indicate gazetter
df_fraustadt.rename(columns = {'name': 'name_gazetter'}, inplace=True)
# sanity check if Fraustadt is still present
df_fraustadt[df_fraustadt['id']=='10505026']


# Let's define a dictionary to match the [Meyers gazetter types](https://www.familysearch.org/wiki/en/Abbreviation_Table_for_Meyers_Orts_und_Verkehrs_Lexikon_Des_Deutschen_Reichs) to the three classes `stadt, landgemeinde, gutsbezirk`. 

# In[27]:


dictionary_types  = {"HptSt.": "stadt",     # Hauptstadt
                     "KrSt.": "stadt",      # Kreisstadt
                     "St.": "stadt",        # Stadt
                     "D.": "landgemeinde",  # Dorf
                     "Dr.": "landgemeinde", # Dörfer
                     "Rg.": "landgemeinde", # Rittergut
                     "G.": "gutsbezirk",    # Gutsbezirk (but also Gericht)
                     "FG": "gutsbezirk",    # Forstgutsbezirk
                     "Gutsb.": "gutsbezirk" # Gutsbezirk
                    }


# Next we need to create a column that entails the "translated" type.  
# 
# Note: I rely on the follwing order `stadt > landgemeinde > gutsbezirk` for classification. For instance, if we have a location that has the types `G.` and `D.`, I will attribute the type `landgemeinde` to the location. Also note that `stadt > landgemeinde > gutsbezirk` is the reverse alpahbetical ordering!

# In[28]:


def check_type(string, dictionary = dictionary_types):
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
             


# In[29]:


testset = ["D. u. Rg.", 
           "GutsB.", 
           "D. u. Dom. (aus: Mittel, Nieder u. Ober D.)", 
           "St.", 
           "St. u. D.", # to check if ordering works
           "D. u. GutsB."   # to check if ordering works
          ]


# In[30]:


for string in testset:
    print("\n" + string)
    print(check_type(string))


# Now that we are all set let's creat the column `class`. And inspect if it worked

# In[31]:


# make sure column Type has only string values
df_fraustadt['Type'] = df_fraustadt['Type'].astype(str)
# make apply check_type() function
df_fraustadt['class_gazetter'] = df_fraustadt['Type'].apply(check_type)
# check results
df_fraustadt[['id', 'name_gazetter', 'lat', 'lng','Type', 'merge_name', 'class_gazetter']].head()


# Load in `Posen-Fraustadt-kreiskey-134.xlsx` file we want to match with Gazetter entries. Clean file before merge!
# 
# # To-Do: Improve string split Pattern
# Improve on split pattern for locations with appendix to accomodate "all" cases:  
# `pattern = \sa\/|\sunt\s|\sa\s|\sunterm\s|\si\/|\si\s|\sb\s|\sin\s|\sbei\s|\sam\s|\san\s`

# In[11]:


df_master = pd.read_excel(os.path.join(wdir, 'Fraustadt', 'Posen-Fraustadt-kreiskey-134.xlsx'))
# rename columns
df_master.rename(columns= {"posen": "province",
                           "134": "province_id",
                           "fraustadt": "district",
                           'Unnamed: 3': "class",
                           "102": "type_id",
                           "Unnamed: 5": "loc_id",
                           "XIV. Kreis Fraustadt": "orig_name"
                          }, inplace=True)
# drop rows with "a) Stadtgemeinden" and "b) Landgemeinden"
df_master = df_master[~df_master['orig_name'].isin(['a) Stadtgemeinden', 'b) Landgemeinden', "c) Gutsbezirke"])]
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
df_master[['temp_name','appendix']] = df_master['name'].str.split(pattern, expand=True)
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
df_master.loc[df_master['suffix'].notnull(), 'alt_name'] = df_master.loc[df_master['suffix'].notnull(), ['suffix', 'name']].apply(lambda x: ' '.join(x), axis=1)
print(f'Number of locations in master file equals {df_master.shape[0]}')


# Check if all went to plan for the distribution of appendices and suffices

# In[32]:


df_master[df_master['appendix'].notnull()].head()


# In[33]:


df_master[df_master['suffix'].notnull()].head()


# ### Now let's try out the merege in a 4-step procedure: 
# 1. merge on the "more restrivtive" `alt_name` that takes into considerations suffixes such as "Nieder" and the `class` label 
# 2. "non-matched" locations will be considered in a second merge based on the location `name` which is the location name without any suffixes and the `class` label
# 3. "non-matched" locations will be considered in a third merge based on "more restrivtive" `alt_name` **but not** on `class` label 
# 4. "non-matched" locations will be considered in a fourth merge based on `name` **but not** on `class` label 

# # Question: Should we do this in a loop? 
# Kept it like this to make it "more" illustrative whats going on

# In[45]:


#  1.) 
columns = list(df_master.columns)
df_join = merge_STATA(df_master, df_fraustadt, how='left', left_on=['alt_name', 'class'], right_on=['merge_name', 'class_gazetter'])
# set aside merged locations
df_merged1 = df_join[df_join['_merge']=='both']
# select locations without a match
df_nomatch = df_join[df_join['_merge']=='left_only']
df_nomatch = df_nomatch[columns]

# 2.)
df_join = merge_STATA(df_nomatch, df_fraustadt, how='left', left_on=['name', 'class'], right_on=['merge_name', 'class_gazetter'])
# set aside merged locations
df_merged2 = df_join[df_join['_merge']=='both']
# select locations without a match
df_nomatch = df_join[df_join['_merge']=='left_only']
df_nomatch = df_nomatch[columns]


# 3.) 
df_join = merge_STATA(df_nomatch, df_fraustadt, how='left', left_on='alt_name', right_on='merge_name')
# set aside merged locations
df_merged3 = df_join[df_join['_merge']=='both']
# select locations without a match
df_nomatch = df_join[df_join['_merge']=='left_only']
df_nomatch = df_nomatch[columns]

# 4.) 
df_join = merge_STATA(df_nomatch, df_fraustadt, how='left', left_on='name', right_on='merge_name')
# concat all dataFrames Dataframes 
df_output = pd.concat([df_merged1, df_merged2, df_merged3, df_join], ignore_index=True)
print(f'{df_output[df_output["_merge"]=="both"].shape[0]} out of {df_output.shape[0]}')


# How well did we do?  
# Note: We now do not consider duplicates but compare to the original excel-file entries

# In[46]:


print(f'''\n{df_output[df_output["_merge"]=="left_only"].shape[0]} out of {df_master.shape[0]} locations were not matched\n{(df_master.shape[0]-df_output[df_output["_merge"]=="left_only"].shape[0])/df_master.shape[0]*100:.2f}% of locations have a match''')


# # To-Do: Eliminate Duplicates
# We can try to **eliminate duplicates** by keeping the Meyer's Gazetter entry with `[lat,lng] != {0,0}`.  

# In[18]:


# write file to disk
df_output.drop(columns=["_merge"], inplace=True)
df_output.sort_values(by="loc_id", inplace=True)
df_output.to_excel(os.path.join(wdir, 'Fraustadt', 'Posen-Fraustadt-kreiskey-134-merged.xlsx'), index=False)


# ### Meyers Gazetter: Which locations were not matched?
# Finally, we would like to know which Gazetter entries are not matched.

# In[20]:


id_gazetter = set(df_fraustadt['id'].values)
id_merge = set(df_output['id'].values)
diff = id_gazetter - id_merge
df_remainder = df_fraustadt[df_fraustadt['id'].isin(diff)]
df_remainder.to_excel(os.path.join(wdir, 'Fraustadt', 'Gazetter_Fraustadt_Lissa_Remainder.xlsx'), index=False)


# In[214]:




