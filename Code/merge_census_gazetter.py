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
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt

# set working directory path as location of data
# WORKING_DIRECTORY = '/Users/nicolaschapman/Documents/NicMergeData/'
#WORKING_DIRECTORY = '/Users/sbec0005/Dropbox/WittenbergerOrdiniertenbuch/PrussianCensusCode/NicMergeData/'
WORKING_DIRECTORY = 'NicMergeData/'

pd.set_option("display.max_rows", None, "display.max_columns", None)


""" ------------------------------------ EXTRACT COUNTY NAMES FROM CENSUS -------------------------------------"""
def extract_county_names(df_census):
    """
    Apply string cleaning and manual updates to the county names contained in the census data

    """
    # extract all counties:
    counties = []
    for county in df_census['county']:
        if county in counties:
            continue
        else:
            counties.append(county)

    # clean county names and extract names to be searched against meyers gazetter
    df_counties = pd.DataFrame(counties, columns = ['orig_name'])
    df_counties['simp_name'] = df_counties['orig_name']

    # fix up found typos and changes first:
    df_counties.loc[df_counties['orig_name'] == 'ostpriegnitz', 'simp_name'] = 'ostprignitz'
    df_counties.loc[df_counties['orig_name'] == 'westpriegnitz', 'simp_name'] = 'westprignitz'
    df_counties.loc[df_counties['orig_name'] == 'krossen', 'simp_name'] = 'crossen'
    df_counties.loc[df_counties['orig_name'] == 'kalau', 'simp_name'] = 'calau'
    df_counties.loc[df_counties['orig_name'] == 'kottbus', 'simp_name'] = 'cottbus'
    df_counties.loc[df_counties['orig_name'] == 'ukermuende', 'simp_name'] = 'ueckermünde'
    df_counties.loc[df_counties['orig_name'] == 'buk', 'simp_name'] = 'neutomischel-grätz'
    df_counties.loc[df_counties['orig_name'] == 'kroeben', 'simp_name'] = 'rawitsch-gostyn'
    df_counties.loc[df_counties['orig_name'] == 'chodziesen', 'simp_name'] = 'kolmar'
    df_counties.loc[df_counties['orig_name'] == 'inowraclaw', 'simp_name'] = 'hohensalza'
    df_counties.loc[df_counties['orig_name'] == 'freistadt', 'simp_name'] = 'freystadt'
    df_counties.loc[df_counties['orig_name'] == 'jerichow I', 'simp_name'] = 'jericho 1'
    df_counties.loc[df_counties['orig_name'] == 'jerichow II', 'simp_name'] = 'jericho 2'
    df_counties.loc[df_counties['orig_name'] == 'stader marschkreis', 'simp_name'] = 'kehdingen'
    df_counties.loc[df_counties['orig_name'] == 'stader geestkreis', 'simp_name'] = 'stade'
    df_counties.loc[df_counties['orig_name'] == 'koblenz', 'simp_name'] = 'coblenz'
    df_counties.loc[df_counties['orig_name'] == 'sanct goar', 'simp_name'] = 'sankt goarshausen'
    df_counties.loc[df_counties['orig_name'] == 'kochem', 'simp_name'] = 'cochem'
    df_counties.loc[df_counties['orig_name'] == 'lyk', 'simp_name'] = 'lyck'
    df_counties.loc[df_counties['orig_name'] == 'kammin', 'simp_name'] = 'cammin'
    df_counties.loc[df_counties['orig_name'] == 'unterwesterwald', 'simp_name'] = 'unterwesterwaldkreis'
    df_counties.loc[df_counties['orig_name'] == 'kleve', 'simp_name'] = 'cleve'
    df_counties.loc[df_counties['orig_name'] == 'chodziesen', 'simp_name'] = 'kolmar'
    df_counties.loc[df_counties['orig_name'] == 'kulm', 'simp_name'] = 'culm'
    df_counties.loc[df_counties['orig_name'] == 'jueterbock-luckenwalde', 'simp_name'] = 'jüterbog-luckenwalde'
    df_counties.loc[df_counties['orig_name'] == 'kalbe', 'simp_name'] = 'calbe'
    df_counties.loc[df_counties['orig_name'] == 'otterndorf', 'simp_name'] = 'ottendorf'

    # extract different forms of the name
    df_counties['simp_name'] = df_counties['simp_name'].str.replace(r'^pr\.\s', '')
    df_counties['simp_name'] = df_counties['simp_name'].str.replace(r'^st\.', 'sankt')
    df_counties['simp_name'] = df_counties['simp_name'].str.replace(r'^.*-.*-.*\s', '')
    df_counties['simp_name'] = df_counties['simp_name'].str.replace(r'oe','ö')
    df_counties['simp_name'] = df_counties['simp_name'].str.replace(r'ue','ü')
    pattern = '\sin\sder\s|\san\sder\s|\sin\s|\sam\s|\sa\.d\.\s|\s|-'
    df_counties[['simp_name','alt_name']] = df_counties['simp_name'].str.split(pattern, expand=True, n=1)

    # manually add alternate names found by inspection:
    df_counties.loc[df_counties['orig_name']=='fraustadt','man_name'] = 'lissa'
    df_counties.loc[df_counties['orig_name']=='konitz','man_name'] = 'tuchel'
    df_counties.loc[df_counties['orig_name']=='hildesheim','man_name'] = 'peine'
    df_counties.loc[df_counties['orig_name']=='osterode a.H.','man_name'] = 'duderstadt'
    df_counties.loc[df_counties['orig_name']=='wennigsen','man_name'] = 'springe'
    df_counties.loc[df_counties['orig_name']=='zellerfeld','man_name'] = 'ilfeld'
    df_counties.loc[df_counties['orig_name']=='pleschen','man_name'] = 'jarotschin'
    df_counties.loc[df_counties['orig_name']=='halberstadt','man_name'] = 'osterwieck'
    df_counties.loc[df_counties['orig_name']=='kosten','man_name'] = 'schmiegel'
    df_counties.loc[df_counties['orig_name']=='eiderstedt','man_name'] = 'garding'
    df_counties.loc[df_counties['orig_name']=='weissenfels','man_name'] = 'hohenmölsen'
    df_counties.loc[df_counties['orig_name']=='fallingbostel','man_name'] = 'soltau'
    df_counties.loc[df_counties['orig_name']=='goettingen','man_name'] = 'münden'
    df_counties.loc[df_counties['orig_name']=='wongrowitz','man_name'] = 'znin'
    df_counties.loc[df_counties['orig_name']=='herford','man_name'] = 'bünde'
    df_counties.loc[df_counties['orig_name']=='trier landkreis','man_name'] = 'hermeskeil'
    df_counties.loc[df_counties['orig_name']=='krotoschin','man_name'] = 'koschmin'
    df_counties.loc[df_counties['orig_name']=='wartenberg','man_name'] = 'gross wartenberg'
    df_counties.loc[df_counties['orig_name']=='nienburg','man_name'] = 'stolzenau'
    df_counties.loc[df_counties['orig_name']=='osnabrueck','man_name'] = 'wittlage'
    df_counties.loc[df_counties['orig_name']=='flensburg','man_name'] = 'kappeln'
    df_counties.loc[df_counties['orig_name']=='lennep','man_name'] = 'wermelskirchen'
    df_counties.loc[df_counties['orig_name']=='verden','man_name'] = 'achim'
    df_counties.loc[df_counties['orig_name']=='dannenberg','man_name'] = 'lüchow'
    df_counties.loc[df_counties['orig_name']=='siegkreis','man_name'] = 'siegburg'
    df_counties.loc[df_counties['orig_name']=='kiel','man_name'] = 'bordesholm'
    df_counties.loc[df_counties['orig_name']=='lueneburg','man_name'] = 'bleckede'
    df_counties.loc[df_counties['orig_name']=='schildberg','man_name'] = 'kempen'
    df_counties.loc[df_counties['orig_name']=='czarnikau','man_name'] = 'filehne'
    df_counties.loc[df_counties['orig_name']=='melle','man_name'] = 'iburg'
    df_counties.loc[df_counties['orig_name']=='gifhorn','man_name'] = 'isenhagen'
    df_counties.loc[df_counties['orig_name']=='lehe','man_name'] = 'geestemünde'
    df_counties.loc[df_counties['orig_name']=='lingen','man_name'] = 'bentheim'
    df_counties.loc[df_counties['orig_name']=='rheingau','man_name'] = 'sankt goarshausen'
    df_counties.loc[df_counties['orig_name']=='wiesbaden','man_name'] = 'höchst'
    df_counties.loc[df_counties['orig_name']=='wernigerode','man_name'] = 'wernigeroda'
    df_counties.loc[df_counties['orig_name']=='gnesen','man_name'] = 'witkowo'
    df_counties.loc[df_counties['orig_name']=='muelheim a.d. Ruhr','man_name'] = 'dinslaken'
    df_counties.loc[df_counties['orig_name']=='bochum','man_name'] = 'hattingen'
    df_counties.loc[df_counties['orig_name']=='obertaunus','man_name'] = 'usingen'
    df_counties.loc[df_counties['orig_name']=='neustadt i. pr.','man_name'] = 'putzig'
    df_counties.loc[df_counties['orig_name']=='nordhausen','man_name'] = 'grafschaft hohenstein'
    df_counties.loc[df_counties['orig_name']=='harburg','man_name'] = 'winsen'
    df_counties.loc[df_counties['orig_name']=='stader geestkreis','man_name'] = 'bremervörde'
    df_counties.loc[df_counties['orig_name']=='kolmar','man_name'] = 'colmar'
    df_counties.loc[df_counties['orig_name']=='stargard','man_name'] = 'dirschau'
    df_counties.loc[df_counties['orig_name']=='leer','man_name'] = 'weener'
    df_counties.loc[df_counties['orig_name']=='solingen','man_name'] = 'ppladen'
    df_counties.loc[df_counties['orig_name']=='solingen','man_name'] = 'ppladen'
    df_counties.loc[df_counties['orig_name']=='schubin','man_name'] = 'znin'
    df_counties.loc[df_counties['orig_name']=='halberstadt','man_name'] = 'osterwieck'
    df_counties.loc[df_counties['orig_name']=='liebenburg','man_name'] = 'goslar'
    df_counties.loc[df_counties['orig_name']=='recklinghausen','man_name'] = 'dorsten'
    df_counties.loc[df_counties['orig_name']=='recklinghausen','man_name'] = 'dorsten'
    df_counties.loc[df_counties['orig_name'] == 'krefeld stadtkreis', 'man_name'] = 'crefeld'
    df_counties.loc[df_counties['orig_name'] == 'krefeld landkreis', 'man_name'] = 'crefeld'
    df_counties.loc[df_counties['orig_name'] == 'koeln landkreis', 'man_name'] = 'cöln'
    df_counties.loc[df_counties['orig_name'] == 'koeln stadtkreis', 'man_name'] = 'cöln'
    df_counties.loc[df_counties['orig_name'] == 'kassel stadtkreis', 'man_name'] = 'cassel'
    df_counties.loc[df_counties['orig_name'] == 'kassel landkreis', 'man_name'] = 'cassel'
    df_counties.loc[df_counties['orig_name'] == 'adelnau', 'man_name'] = 'ostrowo'
    df_counties.loc[df_counties['orig_name'] == 'rotenburg stade', 'man_name'] = 'zeven'
    df_counties.loc[df_counties['orig_name'] == 'emden', 'man_name'] = 'norden'
    df_counties.loc[df_counties['orig_name'] == 'aurich', 'man_name'] = 'wittmund'
    df_counties.loc[df_counties['orig_name'] == 'marienburg b.H.', 'man_name'] = 'alfeld'
    df_counties.loc[df_counties['orig_name'] == 'hagen', 'man_name'] = 'schwelm'
    df_counties.loc[df_counties['orig_name'] == 'celle', 'man_name'] = 'burgdorf'
    df_counties.loc[df_counties['orig_name'] == 'inowraclaw', 'man_name'] = 'inowrazlaw'
    df_counties.loc[df_counties['orig_name'] == 'langensalza', 'man_name'] = 'weimar'
    df_counties.loc[df_counties['orig_name'] == 'rinteln', 'man_name'] = 'grafschaft schaumburg'
    df_counties.loc[df_counties['orig_name'] == 'essen landkreis', 'man_name'] = 'geldern'
    df_counties.loc[df_counties['orig_name'] == 'thorn', 'man_name'] = 'marienwerder'
    df_counties.loc[df_counties['orig_name'] == 'Communion-Bergamts-Bezirk Goslar', 'man_name'] = 'gandersheim'


    # strip all [name, alt_name, suffix] of white spaces
    # df_master.replace(np.nan, '', regex=True)
    for c in ['simp_name']:
        df_counties[c] = df_counties[c].str.strip()

    return df_counties

""" ----------------------------------- LOAD GAZETTE DATA AND CLEAN FOR MERGE -----------------------------------"""

def gazetter_data(county_names, df, map_names, prussia_map):
    """
    Filters the gazetter data based on the presence of the county name anywhere in a gazetter entry, and based on
    location data falling within the buffered county map region.

    """

    ########## FILTER GAZETTER DATA FOR RELEVANT ENTRIES ##########
    # initial entry to start dataframe
    df_county = df[df['Kr'].str.contains(county_names[0].title(), na=False)]
    # loop through each potential name and if it is a substring in any column of df, add it to df_county
    for name in county_names:
        # ignore names that are generic and filter poorly
        if name in ['stadt', 'landkreis', 'main', 'rhein'] or len(name)<3 or '.' in name:
            continue
        for header in list(df.columns.values):
            try:
                df_temp = df[df[header].str.contains(name.title(), na=False)]
                df_county = pd.concat([df_county, df_temp], ignore_index=True)
            except AttributeError:
                continue

    # drop the duplicate entries added
    df_county = df_county.drop_duplicates(subset=['id'], keep='first')

    # find gazetter entries from map:
    df_map_county, county_poly_buffered = gazetter_data_map(df, map_names, prussia_map, county_names[0])
    df.drop(columns=["geometry"], inplace=True)

    # concatenate that too:
    df_county = pd.concat([df_county, df_map_county], ignore_index=True, sort=False)
    # duplicated columns: keep only first
    df_county = df_county.groupby(['id']).first().reset_index()
    #print(df_county[df_county['geometry'].str.contains('Point', na=False)])
    #df[df['Kr'].str.contains(county_names[0].title(), na=False)]

    # create column for merge
    df_county['merge_name'] = df_county['name'].str.strip()
    df_county['merge_name'] = df_county['merge_name'].str.lower()
    print(f'The number of locations in the Gazetter with "{county_names[0]}" in any of their columns is {df_county.shape[0]}')
    print(f'The number of locations in the Gazetter with inside the county boundary is {df_county[df_county["geometry"].notnull()].shape[0]}')
    # rename name column to indicate gazetter
    df_county.rename(columns={'name': 'name_gazetter'}, inplace=True)


    ########### STRING CLEANING AND ALTERNATIVE NAME EXTRACTION FROM GAZETTER ENTRY ##########
    # extract base name, eg: Zedlitz from Nieder Zedlitz
    df_county['base_merge_name'] = df_county['merge_name'].str.extract(r'(.*)\s.*', expand=True)
    # alternative scenario where the first name is actually the one we want.
    df_county['base_merge_name_alt'] = df_county['merge_name'].str.extract(r'.*\s(.*)', expand=True)
    # similar process, but for a dash
    df_county.loc[df_county['merge_name'].str.contains('-'), 'base_merge_name'] = df_county.loc[df_county['merge_name'].str.contains('-'), 'merge_name'].str.split('-').str[0]
    # alternative scenario where the first name is actually the one we want.
    df_county.loc[df_county['merge_name'].str.contains('-'), 'base_merge_name_alt'] = df_county.loc[df_county['merge_name'].str.contains('-'), 'merge_name'].str.split('-').str[-1]

    # spaces in merge_name removed to match with spaces in alt_name removed.
    df_county['merge_name'] = df_county['merge_name'].str.replace(r'\s', '')

    # now account for cases where there is a comma
    df_county.loc[df_county['merge_name'].str.contains(','), 'base_merge_name'] = df_county.loc[df_county['merge_name'].str.contains(','), 'merge_name'].str.split(',').str[0]
    # alternative scenario where the first name is actually the one we want.
    df_county.loc[df_county['merge_name'].str.contains(','), 'base_merge_name_alt'] = df_county.loc[df_county['merge_name'].str.contains(','), 'merge_name'].str.split(',').str[-1]


    # EXTRACTING CORRECT CLASSIFICATION
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

    # Now that we are all set let's create the column `class`. And inspect if it worked
    # make sure column Type has only string values
    df_county['Type'] = df_county['Type'].astype(str)
    # make apply check_type() function
    df_county['class_gazetter'] = df_county['Type'].apply(check_type)


    #print(df_county[['merge_name', 'base_merge_name', 'base_merge_name_alt']])

    return df_county, county_poly_buffered

""" ---------------------------EXTRACT GAZETTER ENTRIES WHICH FALL INSIDE MAPPED REGION-------------------------"""

def extract_map_names(df_counties_census, prussia_map):
    df_counties_census['simp_name'] = df_counties_census['simp_name'].str.replace(r'ö', 'o')
    df_counties_census['simp_name'] = df_counties_census['simp_name'].str.replace(r'ü', 'u')

    county_map_name = {}

    for header in list(df_counties_census.columns.values):
        count = -1
        for county in df_counties_census[header]:
            count += 1
            if county in ['stadt', 'gross', 'ost', 'west', 'stader', 'sankt']:
                continue

            try:
                for map_county in prussia_map["NAME"]:
                    if county.upper() in map_county:
                        orig_name = df_counties_census.loc[df_counties_census[header] == county, 'orig_name']
                        if orig_name[count] not in county_map_name.keys():
                            county_map_name[orig_name[count]] = set([map_county])
                        else:
                            county_map_name[orig_name[count]].add(map_county)
            except AttributeError:
                continue

    # manually enter remaining:
    county_map_name['rinteln'] = set(["SCHAUMBURG"])
    county_map_name['kroeben'] = set(["KROBEN"])
    county_map_name['osterode'] = set(["OSTERODE"])
    county_map_name['jerichow I'] = set(["JERICHOW I"])
    county_map_name['jerichow II'] = set(["JERICHOW II"])
    county_map_name['stader geestkreis'] = set(["STADER GEESTKREIS"])
    county_map_name['lingen'] = set(["LINGEN"])
    county_map_name['koenigsberg in der neumark'] = set(["KONIGSBERG"])
    county_map_name['friedeberg in der neumark'] = set(["FRIEDEBERG"])
    county_map_name['mansfeld gebirgskreis'] = set(["MANSFELDER GEBIRGSKR"])
    county_map_name['mansfeld seekreis'] = set(["MANSFELDER SEEKREIS"])
    county_map_name['erfurt stadtkreis'] = set(['WIPPERFURTH', 'ERFURT'])
    county_map_name['erfurt landkreis'] = set(['WIPPERFURTH', 'ERFURT'])
    county_map_name['osterode a.H.'] = set(['OSTERODE AM HARZ'])
    county_map_name['muelheim am rhein'] = set(['MULHEIM'])
    county_map_name['ost-sternberg'] = set(['OSTSTERNBERG'])
    county_map_name['west-sternberg'] = set(['WESTSTERNBERG'])
    county_map_name['sanct goar'] = set(['SANKT GOAR'])
    county_map_name['otterndorf'] = set(['HADELN'])
    county_map_name['kolberg-koerlin'] = set(['FURSTENTUM'])
    county_map_name['zell'] = set(['ZELL'])
    county_map_name['stettin'] = set(['STETTIN'])
    county_map_name['stettin stadt'] = set(['STETTIN'])
    county_map_name['erfurt stadtkreis'] = set(['ERFURT'])
    county_map_name['erfurt landkreis'] = set(['ERFURT'])
    county_map_name['rotenburg kassel'] = set(['ROTENBURG'])
    county_map_name['rotenburg stade'] = set(['ROTENBURG'])
    county_map_name['muenster stadtkreis'] = set(['MUNSTER'])
    county_map_name['muenster landkreis'] = set(['MUNSTER'])
    county_map_name['essen landkreis'] = set(['ESSEN'])
    county_map_name['thorn'] = set(['THORN'])
    county_map_name['stettin stadt'] = set(['STETTIN'])
    county_map_name['obertaunus'] = set(['OBERTAUNUSKREIS'])
    county_map_name['hagen'] = set(['HAGEN'])
    county_map_name['ohlau'] = set(['OHLAU'])
    county_map_name['schildberg'] = set(['SCHILDBERG'])

    return county_map_name

def multiple_maps(map_name, county_gdf, county):
    if map_name == 'KONIGSBERG':
        if county == 'koenigsberg':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'koenigsberg in der neumark':
            county_gdf = county_gdf[county_gdf.index == 0]
        if county == 'koenigsberg stadt':
            county_gdf = county_gdf[county_gdf.index == 2]
    if map_name == 'POSEN':
        if county == 'posen landkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'posen stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'BRESLAU':
        if county == 'breslau landkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'breslau stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'LIEGNITZ':
        if county == 'liegnitz landkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'liegnitz stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'ERFURT':
        if county == 'erfurt landkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'erfurt stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'HANNOVER':
        if county == 'hannover landkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
        if county == 'hannover stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
    if map_name == 'ROTENBURG':
        if county == 'rotenburg stade':
            county_gdf = county_gdf[county_gdf.index == 0]
        if county == 'rotenburg kassel':
            county_gdf = county_gdf[county_gdf.index == 1]
    if map_name == 'MUNSTER':
        if county == 'muenster stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
        if county == 'muenster landkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
    if map_name == 'KASSEL':
        if county == 'kassel stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
        if county == 'kassel landkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
    if map_name == 'FRANKFURT':
        if county == 'frankfurt am main':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'frankfurt an der oder':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'ESSEN':
        if county == 'essen stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
        if county == 'essen landkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
    if map_name == 'MULHEIM':
        if county == 'muelheim am rhein':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'muelheim a.d. Ruhr':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'KOLN':
        if county == 'koeln stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'koeln landkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'TRIER':
        if county == 'trier stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
        if county == 'trier landkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
    if map_name == 'AACHEN':
        if county == 'aachen stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'aachen landkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'MARIENBURG':
        if county == 'marienburg b.H.':
            county_gdf = county_gdf[county_gdf.index == 0]
        if county == 'marienburg i. pr. ':
            county_gdf = county_gdf[county_gdf.index == 1]
    if map_name == 'NEUSTADT':
        if county == 'neustadt i. pr. ':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'neustadt in oberschlesien':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'ROSENBERG':
        if county == 'rosenberg i. pr.':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'rosenberg':
            county_gdf = county_gdf[county_gdf.index == 0]
    if map_name == 'DUSSELDORF':
        if county == 'duesseldorf landkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
        if county == 'duesseldorf stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
    if map_name == 'KREFELD':
        if county == 'krefeld stadtkreis':
            county_gdf = county_gdf[county_gdf.index == 1]
        if county == 'krefeld landkreis':
            county_gdf = county_gdf[county_gdf.index == 0]
    return county_gdf

def gazetter_data_map(df_gazetter, map_names, prussia_map, county):
    gdf_gazetter = gpd.GeoDataFrame(df_gazetter, geometry=gpd.points_from_xy(df_gazetter.lng, df_gazetter.lat))
    df_gazetter_map_county = None
    for map_name in map_names:
        if map_name == None:
            continue

        # extract county poly
        county_gdf = prussia_map[prussia_map['NAME'] == map_name]
        county_gdf.index = range(0, county_gdf.shape[0])

        # if there are multiple regions in the map data with the same name, extract the correct one.
        county_gdf = multiple_maps(map_name, county_gdf, county)

        # special case for rotenberg which has two distinct regions
        county_poly_buffered = county_gdf.buffer(0.05).iloc[0]
        within = gdf_gazetter[gdf_gazetter.within(county_poly_buffered)]
        index_in_county = set()

        for j in within.index:
            index_in_county.add(j)
        # update to only keep the locations deemed to be within the county
        if df_gazetter_map_county is None:
            df_gazetter_map_county = df_gazetter.reindex(index_in_county)
        else:
            df_gazetter_map_county = pd.concat([df_gazetter_map_county, df_gazetter.reindex(index_in_county)], ignore_index=True, sort=False)

    return df_gazetter_map_county, county_poly_buffered

""" ----------------------------LOAD CLEANED CENSUS DATA AND APPLY STRING CLEANING -----------------------------------"""
def census_data(county, df_census):

    # extract county data
    df_county = df_census[df_census['county']==county]

    # upload cleaned data
    #df_master = pd.read_excel(os.path.join(wdir, 'PrussianCensus1871/Fraustadt', 'Posen-Fraustadt-kreiskey-134.xlsx'))
    # rename columns
    df_county.rename(columns={"regbez": "province",
                              "kreiskey1871": "province_id",
                              "county": "district",
                              'type': "class",
                              "page": "type_id",
                              "running_number": "loc_id",
                              "locname": "orig_name"
                              }, inplace=True)

    # add flag denoting if an matching municipality exists with different class:
    df_county = df_county.assign(alt_class=False)
    for name in df_county['orig_name']:
        if df_county[df_county['orig_name'] == name].shape[0]>1:
            df_county.loc[df_county['orig_name']==name, 'alt_class'] = True

    ########## GENERAL CLEAN CENSUS DATA FOR MATCHING ##########

    # now we need to clean location names
    df_county['name'] = df_county['orig_name']

    # adjustment for big cities with strange naming conventions: use the county name as it generally better
    if df_county.shape[0]==1:
        df_county = df_county.assign(name=county)
        if county == 'krefeld stadtkreis':
            df_county = df_county.assign(name='crefeld')
        if county == 'kassel stadtkreis':
            df_county = df_county.assign(name='cassel')
        if county == 'koeln stadtkreis':
            df_county = df_county.assign(name='cöln')

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

    # adjustement for beuthen name changes:
    if county == 'beuthen':
        df_county.loc[df_county['name'].str.contains('Haiduck'), 'name'] = 'Bismarckhütte OSchles.'
        df_county.loc[df_county['name'].str.contains('Lagiewni'), 'name'] = 'Hohenlinde'


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
    print(f'Number of locations in master file equals {df_county.shape[0]}')

    return df_county

"""----------------------------------- MERGE THE TWO DATA FRAMES -----------------------------------"""
def merge_data(df_county_gaz, df_county_cens):

    # split df_fraustadt into lat/long == null and lat/long!=null
    df_county_gaz_latlong = df_county_gaz[df_county_gaz['geometry'].notnull()]
    df_county_gaz_null = df_county_gaz[df_county_gaz['geometry'].isnull()]

    # Now let's try out the merege in a 4-step procedure. This procedure is iterated through twice, once for gazetter entries
    # that have a valid lat-long, and one for entries that do not.
    # 1. merge on the "more restrivtive" `alt_name` that takes into considerations suffixes such as "Nieder" and the `class` label
    # 2. "non-matched" locations will be considered in a second merge based on the location `name` which is the location name without any suffixes and the `class` label
    # 3. "non-matched" locations will be considered in a third merge based on "more restrivtive" `alt_name` **but not** on `class` label
    # 4. "non-matched" locations will be considered in a fourth merge based on `name` **but not** on `class` label
    # 5. a fifth merge is then completed, attempting to match the simplified "name" from the census with simpler forms of the gazetter name.

    #  1.)
    columns = list(df_county_cens.columns)
    df_county_gaz_latlong = df_county_gaz_latlong.assign(merge_round=1)
    print("Merging if detailed census name and class match a gazetter entry with location data")
    df_join = merge_STATA(df_county_cens, df_county_gaz_latlong, how='left', left_on=['alt_name', 'class'],
                          right_on=['merge_name', 'class_gazetter'])
    # set aside merged locations
    df_merged1 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 2.)
    df_county_gaz_latlong = df_county_gaz_latlong.assign(merge_round=2)
    print("Merging if simplified census name and class match a gazetter entry with location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_latlong, how='left', left_on=['name', 'class'],
                          right_on=['merge_name', 'class_gazetter'])
    # set aside merged locations
    df_merged2 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 3.)
    df_county_gaz_latlong = df_county_gaz_latlong.assign(merge_round=3)
    print("Merging if detailed census name matches a gazetter entry with location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_latlong, how='left', left_on='alt_name', right_on='merge_name')
    # set aside merged locations
    df_merged3 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 4.)
    df_county_gaz_latlong = df_county_gaz_latlong.assign(merge_round=4)
    print("Merging if simplified census name matches a gazetter entry with location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_latlong, how='left', left_on='name', right_on='merge_name')
    # set aside merged locations
    df_merged4 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 5.)
    df_county_gaz_latlong = df_county_gaz_latlong.assign(merge_round=5)
    print("Merging if simplified census name matches simplified gazetter entry with location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_latlong, how='left', left_on='name', right_on='base_merge_name')
    # set aside merged locations
    df_merged5 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 5.1)
    print("Merging if simplified census name matches simplified gazetter entry with location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_latlong, how='left', left_on='name', right_on='base_merge_name_alt')
    # set aside merged locations
    df_merged5_alt = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # Repeat for gazetter entries with either no location data, or location data outside the county just for clarity of a match
    #  1.)
    df_county_gaz_null = df_county_gaz_null.assign(merge_round=1)
    print("Merging if detailed census name and class match a gazetter entry WITHOUT location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_null, how='left', left_on=['alt_name', 'class'],
                          right_on=['merge_name', 'class_gazetter'])
    # set aside merged locations
    df_merged6 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 2.)
    df_county_gaz_null = df_county_gaz_null.assign(merge_round=2)
    print("Merging if simplified census name and class match a gazetter entry WITHOUT location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_null, how='left', left_on=['name', 'class'],
                          right_on=['merge_name', 'class_gazetter'])
    # set aside merged locations
    df_merged7 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 3.)
    df_county_gaz_null = df_county_gaz_null.assign(merge_round=3)
    print("Merging if detailed census name matches a gazetter entry WITHOUT location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_null, how='left', left_on='alt_name', right_on='merge_name')
    # set aside merged locations
    df_merged8 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 4.)
    df_county_gaz_null = df_county_gaz_null.assign(merge_round=4)
    print("Merging if simplified census name matches a gazetter entry WITHOUT location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_null, how='left', left_on='name', right_on='merge_name')
    # set aside merged locations
    df_merged9 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    ########### LAST EFFORT: FINAL STRING CLEANING AND MATCH AGAINST ALTERNATIVE GAZETTER NAMES  ###########

    # before final merge, split at the dash and compare to all!
    df_nomatch.loc[df_nomatch["orig_name"].str.contains('-'), "alt_name"] = df_nomatch.loc[df_nomatch["orig_name"].str.contains('-'), "name"].str.split('-').str[0]
    df_nomatch.loc[df_nomatch["orig_name"].str.contains('-'), "name"] = df_nomatch.loc[df_nomatch["orig_name"].str.contains('-'), "name"].str.split('-').str[-1]
    df_nomatch.loc[df_nomatch["alt_name"].isnull(), "alt_name"] = 'xxxxxxxxxx'

    # 5.)
    df_county_gaz = df_county_gaz.assign(merge_round=5)
    print("Merging if simplified census name matches simplified gazetter entry with location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz, how='left', left_on='name', right_on='base_merge_name')
    # set aside merged locations
    df_merged10 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    print("Merging if simplified census name matches simplified gazetter entry with location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz, how='left', left_on='alt_name', right_on='base_merge_name')
    # set aside merged locations
    df_merged10_2 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    print("Merging if simplified census name matches simplified gazetter entry with location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz, how='left', left_on='name', right_on='base_merge_name_alt')
    # set aside merged locations
    df_merged10_alt = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 5.1)
    print("Merging if simplified census name matches simlified gazetter entry WITHOUT location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz, how='left', left_on='alt_name', right_on='base_merge_name_alt')
    # set aside merged locations
    df_merged10_alt_2 = df_join[df_join['_merge'] == 'both']

    # concat all dataFrames Dataframes
    df_combined = pd.concat(
        [df_merged1, df_merged2, df_merged3, df_merged4, df_merged5, df_merged5_alt, df_merged6, df_merged7, df_merged8, df_merged9,
         df_merged10, df_merged10_2, df_merged10_alt, df_merged10_alt_2], ignore_index=True)

    # How well did we do?
    # Note: We now do not consider duplicates but compare to the original excel-file entries
    exact_match_perc = (df_county_cens.shape[0] - df_join[df_join["_merge"] == "left_only"].shape[0]) / df_county_cens.shape[0] * 100

    return df_combined, exact_match_perc, df_join

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


"""---------------------------- ANALYSIS OF UNMATCHED ENTRIES - LEVENSHTEIN DISTANCE ---------------------------"""
def lev_dist_calc(df_county_cens, df_county_gaz, df_merged, county, df_join):
    # Meyers Gazetter: Which locations were not matched?
    # Finally, we would like to know which Gazetter entries are not matched.
    columns = list(df_county_cens.columns)
    id_gazetter = set(df_county_gaz['id'].values)
    id_merge = set(df_merged['id'].values)
    diff = id_gazetter - id_merge
    df_remainder = df_county_gaz[df_county_gaz['id'].isin(diff)]

    if not os.path.exists(os.path.join(WORKING_DIRECTORY, 'OutputCounty/', re.sub(r'\s$', '', county))):
        os.makedirs(os.path.join(WORKING_DIRECTORY, 'OutputCounty/', re.sub(r'\s$', '', county)))

    df_remainder.to_excel(os.path.join(WORKING_DIRECTORY, 'OutputCounty/', re.sub(r'\s$', '', county), 'Gazetter_Remainder_' + county + '.xlsx'),index=False)

    # extract unmatched names from census data:
    unmatched_census_df = df_join[df_join['_merge'] == 'left_only']
    unmatched_census_df = unmatched_census_df[columns]
    unmatched_name_census = unmatched_census_df["name"]
    unmatched_altname_census = unmatched_census_df["alt_name"].astype(str)

    # extract entries in gazetter data:
    unmatched_name_gazetter = df_county_gaz['merge_name']
    unmatched_base_name_gazetter = df_county_gaz['base_merge_name'].astype(str)
    unmatched_base_name_alt_gazetter = df_county_gaz['base_merge_name_alt'].astype(str)

    # call levenshtein comparison function
    levenshtein_matches = lev_array(unmatched_name_gazetter, unmatched_name_census)
    levenshtein_matches += lev_array(unmatched_name_gazetter, unmatched_altname_census)
    levenshtein_matches += lev_array(unmatched_base_name_gazetter, unmatched_name_census)
    levenshtein_matches += lev_array(unmatched_base_name_alt_gazetter, unmatched_name_census)

    # convert list of lists to data frame
    unmatched_census_df = unmatched_census_df.assign(lev_match=unmatched_census_df['name'])
    unmatched_census_df = unmatched_census_df.assign(lev_dist=100)

    for match in levenshtein_matches:
        unmatched_census_df.loc[(unmatched_census_df['name'] == match[0])&(unmatched_census_df['lev_dist'] > match[2]), 'lev_match'] = match[1]
        unmatched_census_df.loc[(unmatched_census_df['name'] == match[0])&(unmatched_census_df['lev_dist'] > match[2]), 'lev_dist'] = match[2]
        unmatched_census_df.loc[(unmatched_census_df['alt_name'] == match[0])&(unmatched_census_df['lev_dist'] > match[2]), 'lev_match'] = match[1]
        unmatched_census_df.loc[(unmatched_census_df['alt_name'] == match[0])&(unmatched_census_df['lev_dist'] > match[2]), 'lev_dist'] = match[2]

    return unmatched_census_df, levenshtein_matches

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
        if census_name == 'nan' or len(census_name) == 0:
            continue
        for gazetter_name in unmatched_gazetter_name:
            if gazetter_name == 'nan' or len(gazetter_name) == 0:
                continue

            # default is that must start with the same letter, except for umlaut c/k exception.
            if gazetter_name[0] != census_name[0]:
                if not (census_name[0:2] == 'ue' and gazetter_name[0] == 'ü'):
                    if not (census_name[0:2] == 'oe' and gazetter_name[0] == 'ö'):
                        if not (census_name[0]=='c' and gazetter_name[0]=='k'):
                            if not (census_name[0] == 'k' and gazetter_name[0] == 'c'):
                                continue
                else:
                    print(census_name)
                    print(gazetter_name)

            ldist = levenshtein(gazetter_name, census_name)
            if ldist / len(census_name) < 0.3:
                entry = [census_name, gazetter_name, ldist / len(census_name)]
                levenshtein_array.append(entry)
    return levenshtein_array


""" ------------------------------- CONDUCT MERGE ROUNDS BASED ON LEVENSTEIN DISTANCE ---------------------------- """
def lev_merge(df_county_gaz, df_merge, unmatched_census_df):
    # lev_merge_df = merge_STATA(unmatched_census_df, df_remainder, how='left', left_on='lev_match', right_on='merge_name')

    # column data has been updated to inculde lev_match so redeclare it
    columns = list(unmatched_census_df.columns)

    # split df_fraustadt into lat/long == null and lat/long!=null
    df_county_gaz_latlong = df_county_gaz[df_county_gaz['geometry'].notnull()]
    df_county_gaz_null = df_county_gaz[df_county_gaz['geometry'].isnull()]

    # merge round 6 for levenshtein merge
    df_county_gaz_latlong = df_county_gaz_latlong.assign(merge_round=6)
    df_county_gaz_null = df_county_gaz_null.assign(merge_round=6)

    # follow the same iterative procedure as before, except using 'lev_match' instead of 'name' or 'alt_name'
    # first check gazetter entries with location data.
    #  1.) Merge if class and levenshtein distance match
    print(
        "Merging if name matches via levenshtein distance algortithm and class matches a gazetter entry with location data")
    df_join = merge_STATA(unmatched_census_df, df_county_gaz_latlong, how='left', left_on=['lev_match', 'class'],
                          right_on=['merge_name', 'class_gazetter'])
    # set aside merged locations
    df_lev_merge1 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 2.) Merge if levenshtein distance only matches
    print("Merging if name matches via levenshtein distance algortithm a gazetter entry with location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_latlong, how='left', left_on='lev_match', right_on='merge_name')
    # set aside merged locations
    df_lev_merge2 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # check unmatched entries against non-location gazetter data
    #  1.) Merge if class and levenshtein distance match
    print(
        "Merging if name matches via levenshtein distance algortithm and class matches a gazetter entry WITHOUT location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_null, how='left', left_on=['lev_match', 'class'],
                          right_on=['merge_name', 'class_gazetter'])
    # set aside merged locations
    df_lev_merge3 = df_join[df_join['_merge'] == 'both']
    # select locations without a match
    df_nomatch = df_join[df_join['_merge'] == 'left_only']
    df_nomatch = df_nomatch[columns]

    # 2.) Merge if levenshtein distance only matches
    print("Merging if name matches via levenshtein distance algortithm a gazetter entry WITHOUT location data")
    df_join = merge_STATA(df_nomatch, df_county_gaz_null, how='left', left_on='lev_match', right_on='merge_name')

    # add to output, but also write to output dedicated to matched made by levenshtein merge.
    # write file to disk
    df_lev_merge = pd.concat([df_lev_merge1, df_lev_merge2, df_lev_merge3, df_join], ignore_index=True)
    df_merge = pd.concat([df_merge, df_lev_merge], ignore_index=True, sort=False)

    return df_merge, df_lev_merge

"""----------------------------------- FINAL SEARCH AGAINST GEONAMES DATABASE --------------------------------------"""

def search_geonames(df_geonames, county_merged, county_poly_buffered):
    merged_no_loc = county_merged.loc[county_merged['name_gazetter'].notnull() & (county_merged['lat'] == 0)]

    for name_gazetter in merged_no_loc['name_gazetter']:

        # extract geonames which match
        geonames_slice = df_geonames[(df_geonames['name']==name_gazetter)|(df_geonames['asciiname']==name_gazetter)|(df_geonames['alternatenames']==name_gazetter)]

        for i in range(geonames_slice.shape[0]):
            merged_no_loc.loc[merged_no_loc['name_gazetter'] == name_gazetter, 'lat'] = geonames_slice.iloc[i, 4]
            merged_no_loc.loc[merged_no_loc['name_gazetter'] == name_gazetter, 'lng'] = geonames_slice.iloc[i, 5]

            #print(merged_no_loc.loc[merged_no_loc['name_gazetter']==name_gazetter, ['name_gazetter', 'lat', 'lng']])

            gdf_loc_de = gpd.GeoDataFrame(merged_no_loc[merged_no_loc['name_gazetter'] == name_gazetter], geometry=gpd.points_from_xy(merged_no_loc[merged_no_loc['name_gazetter'] == name_gazetter].lng, merged_no_loc[merged_no_loc['name_gazetter'] == name_gazetter].lat))
            within = gdf_loc_de[gdf_loc_de.within(county_poly_buffered)]

            if within.shape[0]>0:
                county_merged.loc[county_merged['name_gazetter']==name_gazetter, 'lat'] = geonames_slice.iloc[i, 4]
                county_merged.loc[county_merged['name_gazetter']==name_gazetter, 'lng'] = geonames_slice.iloc[i, 5]
                county_merged.loc[county_merged['name_gazetter']==name_gazetter, 'geo_names'] = True
                break

    return county_merged

"""----------------------------------- WRITE DATA AND STATS TO FILE --------------------------------------"""

def write_merged_data(df_merged, df_lev_merged, county):
    # prepare for total output write to file
    df_merged.drop(columns=["_merge"], inplace=True)
    df_merged.sort_values(by="loc_id", inplace=True)
    df_merged.drop(columns=["lev_match"], inplace=True)
    df_merged.to_excel(os.path.join(WORKING_DIRECTORY, 'OutputCounty/', re.sub(r'\s$', '', county), 'Merged_Data_' + county + '.xlsx'),
                       index=False)
    # prepare for levenshtein matches write to file
    df_lev_merged.drop(columns=["_merge"], inplace=True)
    df_lev_merged.drop(columns=["lev_match"], inplace=True)
    df_lev_merged.sort_values(by="loc_id", inplace=True)
    df_lev_merged.to_excel(
        os.path.join(WORKING_DIRECTORY, 'OutputCounty', re.sub(r'\s$', '', county), 'Lev_Merged_Data_' + county + '.xlsx'),
        index=False)

    """ ------------------------------------ CALCULATE QUALITY STATISTICS ----------------------------------------"""

def qual_stat(exact_match_perc, df_merge_nodups, county, levenshtein_matches):
    print(f'''\n{exact_match_perc:.2f}% of locations have match with the same name''')
    match_perc = 100 * (df_merge_nodups.shape[0] - df_merge_nodups[df_merge_nodups['id'].isnull()].shape[0]) / \
                 df_merge_nodups.shape[0]
    print(f'''\n{match_perc:.2f}% of locations were matched when levenshtein distance was considered''')
    loc_perc = 100 * (df_merge_nodups[df_merge_nodups['geometry'].notnull()|df_merge_nodups['geo_names']].shape[0])/df_merge_nodups.shape[0]
    print(f'''\n{loc_perc:.2f}% of locations were matched to geocode data''')
    round1_perc = 100 * df_merge_nodups[df_merge_nodups['merge_round'] == 1].shape[0] / df_merge_nodups.shape[0]
    print(f'''\n{round1_perc:.2f}% of locations were matched by classification and the exact name''')
    round2_perc = 100 * df_merge_nodups[df_merge_nodups['merge_round'] == 2].shape[0] / df_merge_nodups.shape[0]
    print(f'''\n{round2_perc:.2f}% of locations were matched by classification and the simplified name''')
    round3_perc = 100 * df_merge_nodups[df_merge_nodups['merge_round'] == 3].shape[0] / df_merge_nodups.shape[0]
    print(f'''\n{round3_perc:.2f}% of locations were matched by exact name only''')
    round4_perc = 100 * df_merge_nodups[df_merge_nodups['merge_round'] == 4].shape[0] / df_merge_nodups.shape[0]
    print(f'''\n{round4_perc:.2f}% of locations were matched by simplified name only''')
    round5_perc = 100 * df_merge_nodups[df_merge_nodups['merge_round'] == 5].shape[0] / df_merge_nodups.shape[0]
    print(
        f'''\n{round5_perc:.2f}% of locations were matched by simplifying the name from both the census and meyers gazetter''')

    # import csv to data frame, edit it and output new data frame to csv
    df_merge_details = pd.read_excel(os.path.join(WORKING_DIRECTORY, 'OutputSummary/', 'MergeDetails.xlsx'))

    # if county has never been run, add new entry
    if df_merge_details[df_merge_details['county'] == county].empty:
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

    df_merge_details.to_excel(os.path.join(WORKING_DIRECTORY, 'OutputSummary', 'MergeDetails.xlsx'), index=False)


"""----------------------------------- CREATE FINAL PRUSSIAN CENSUS FILE  --------------------------------------"""
def update_census(df_counties):
    # create PrussianCensusMerged file
    flag = True
    for county in df_counties['orig_name']:
        county_merged_df = pd.read_excel(os.path.join(WORKING_DIRECTORY, 'OutputCounty/', re.sub(r'\s$', '', county), 'Merged_Data_' + county + '.xlsx'))

        # amalagamation code for certain cities:
        county_merged_df['amalg_flag']=False
        if county in ['trier stadtkreis', 'frankfurt am main', 'liegnitz stadtkreis','Communion-Bergamts-Bezirk Goslar']:
            county_merged_df = amalgamate_unmatched(county_merged_df)

        within_data = county_merged_df[county_merged_df['geometry'].notnull() | (county_merged_df['geo_names']) | (county_merged_df['amalg_flag'])]
        within_data = within_data.drop_duplicates(subset=['loc_id'], keep='first')
        missing = ~county_merged_df.loc_id.isin(within_data.loc_id)
        missing_data = county_merged_df[missing]
        missing_data = missing_data.drop_duplicates(subset=['loc_id'], keep='first')
        missing_data['lat'] = 0
        missing_data['lng'] = 0

        census_update_county = pd.concat([within_data, missing_data], ignore_index=True)
        headers = ['province', 'province_id', 'district', 'class', 'type_id', 'loc_id', 'orig_name', 'pop_male', 'pop_female', 'pop_tot', 'protestant', 'catholic',
                        'other_christ', 'jew', 'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate', 'province', 'alt_class', 'geo_names', 'lat', 'lng', 'amalg_flag', 'Kr', 'merge_round']
        census_update_county = census_update_county[headers]
        if flag:
            census_update = census_update_county
            flag = False
        else:
            census_update = pd.concat([census_update, census_update_county], ignore_index=True)

    census_update.to_excel(os.path.join(WORKING_DIRECTORY, 'OutputSummary/PrussianCensusUpdated.xlsx'),index=False)

def amalgamate_unmatched(df_merged):
    i = 0
    if df_merged.iloc[i,148]==0:
        i = 1
    lat = df_merged.iloc[i, 148]
    lat = np.random.normal(lat, 0.02)
    lng = df_merged.iloc[i,149]
    lng = np.random.normal(lng, 0.02)
    df_merged.loc[df_merged['geometry'].isnull()&(df_merged['geo_names']==False), 'lat'] = lat
    df_merged.loc[df_merged['geometry'].isnull()&(df_merged['geo_names']==False), 'lng'] = lng
    df_merged.loc[df_merged['geometry'].isnull() & (df_merged['geo_names'] == False), 'amalg_flag'] = True
    return df_merged

"""----------------------------------- FUNCTION INITIAITES ENTIRE CODE  --------------------------------------"""

def run_full_merge():
    # load saved data frame containing census file
    df_census = pd.read_pickle(os.path.join(WORKING_DIRECTORY, "census_df_pickle"))

    # account for two different rotenburgs:
    df_census.loc[(df_census['county']=='rotenburg')&(df_census['regbez']=='kassel'),'county'] = 'rotenburg kassel'
    df_census.loc[(df_census['county']=='rotenburg')&(df_census['regbez']=='stade'),'county'] = 'rotenburg stade'

    # extract county name data frame from census
    df_counties = extract_county_names(df_census)

    # read in map of prussia
    prussia_map = gpd.read_file(os.path.join(WORKING_DIRECTORY, "PrussianCensus1871/GIS/1871_county_shapefile-new.shp"))
    # convert to longitude and latitude for printing
    prussia_map = prussia_map.to_crs(epsg=4326)

    # find county name from map:
    map_names = extract_map_names(df_counties, prussia_map)

    # repeat this as slight changes were made in
    df_counties = extract_county_names(df_census)

    # load saved data frame
    df_gazetter = pd.read_pickle(os.path.join(WORKING_DIRECTORY, "df_pickle"))
    print(f'The number of entries in Meyer Gazetter is: {df_gazetter.shape[0]}')

    # load in GeoNames dataframe
    df_geonames_de = pd.read_csv(os.path.join(WORKING_DIRECTORY, 'GeoNames/DE/DE.txt'), sep="\t", header=None, low_memory=False)
    df_geonames_de.columns = ["geonameid", "name", 'asciiname', "alternatenames", "latitude", "longitude",
                              "feature class", "feature code", "country code ", 'cc2', 'admin1 code', 'admin2 code',
                              'admin3 code', 'admin4 code', 'population', 'elevation', 'dem', 'timezone',
                              'modification date']
    df_geonames_pl = pd.read_csv(os.path.join(WORKING_DIRECTORY, 'GeoNames/PL/PL.txt'), sep="\t", header=None, low_memory=False)
    df_geonames_pl.columns = ["geonameid", "name", 'asciiname', "alternatenames", "latitude", "longitude",
                              "feature class", "feature code", "country code ", 'cc2', 'admin1 code', 'admin2 code',
                              'admin3 code', 'admin4 code', 'population', 'elevation', 'dem', 'timezone',
                              'modification date']

    # build up list of possible county names to be searched against gazetter.
    count = 0
    for county in df_counties['orig_name']:
        count+=1
        print(count)
        if 'pr' not in county:
            continue
        current_county = df_counties.loc[df_counties['orig_name'] == county]
        current_county=current_county.reset_index()
        current_county.drop(columns=["index"], inplace=True)
        current_county_names = []
        for name in current_county.loc[0]:
            if name==None or str(name)=='nan' or name in current_county_names:
                continue
            current_county_names.append(name)

        print(f'''\nMATCHING FOR: {current_county_names[0]}\n''')
        print(current_county_names)

        # extract county name that appears in the census
        county = current_county_names[0]

        # gather and clean gazetter entires
        df_gazetter_county, county_poly_buffered = gazetter_data(current_county_names,df_gazetter, map_names[county], prussia_map)

        # gather and clean census
        df_census_county = census_data(county, df_census)

        # merge census data
        df_merged, exact_match_perc, df_join = merge_data(df_gazetter_county, df_census_county)

        # complete levenshtein distance calulations
        df_unmatched_census, levenshtein_matches = lev_dist_calc(df_census_county, df_gazetter_county, df_merged, county, df_join)

        # merge based on levenshtein distance
        df_merged, df_lev_merged = lev_merge(df_gazetter_county, df_merged, df_unmatched_census)

        # final merge by searching against GeoNames data for matched entries lacking geocode data!
        df_merged['geo_names'] = False
        df_merged = search_geonames(df_geonames_de, df_merged, county_poly_buffered)
        df_merged = search_geonames(df_geonames_pl, df_merged, county_poly_buffered)

        # write to file
        write_merged_data(df_merged, df_lev_merged, county)

        # drop duplicates from output arbitratilly. !! Need a method.
        df_merged_nodups = df_merged.drop_duplicates(subset=['loc_id'], keep='last')

        # calculate quality stats
        qual_stat(exact_match_perc, df_merged_nodups, county, levenshtein_matches)

    # update the census
    update_census(df_counties)





