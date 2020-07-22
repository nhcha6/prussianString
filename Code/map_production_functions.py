# file for producing a map of fraustadt data
import numpy as np
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from map_details import *
import mapclassify as mc

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

def extract_county_names(df_census):
    # extract all counties:
    counties = []
    for county in df_census['county']:
        if county in counties:
            if county == 'rotenberg':
                counties.append('rotenberg kassel')
                counties.append('rotenberg stade')
                counties.remove('rotenberg')
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
    return df_merged

def plot_county(county, plot_headers, prussia_map, showFlag, map_names):
    # set seed so that random numbers generate identically each time
    np.random.seed(1)

    # read in merged data
    county_merged_df = pd.read_excel("Merged_Data/" + county + "/Merged_Data_" + county + '.xlsx')

    # amalagamation code for certain cities:
    if county in ['trier stadtkreis', 'frankfurt am main', 'liegnitz stadtkreis', 'Communion-Bergamts-Bezirk Goslar']:
        county_merged_df = amalgamate_unmatched(county_merged_df)

    # drop geometry column
    county_merged_df.drop(columns=["geometry"], inplace=True)

    # extract county poly, need to loop not pop.
    map_name = map_names[county].pop()
    county_gdf = prussia_map[prussia_map['NAME']==map_name]
    county_gdf.index = range(0,county_gdf.shape[0])

    # if there are multiple regions in the map data with the same name, extract the correct one.
    county_gdf = multiple_maps(map_name, county_gdf, county)

    county_poly_buffered = county_gdf.buffer(0.05)[0]

    # add a little bit of noise to ensure that identical data points are split slightly
    county_merged_df['lat'] = np.random.normal(county_merged_df['lat'],0.01)
    county_merged_df['lng'] = np.random.normal(county_merged_df['lng'],0.01)

    # convert to geo data frame
    county_merged_gdf = gpd.GeoDataFrame(county_merged_df, geometry=gpd.points_from_xy(county_merged_df.lng,county_merged_df.lat))
    county_merged_gdf.crs = {'init': 'epsg:4326'}

    # drop those outside buffered poly
    county_merged_gdf = county_merged_gdf[county_merged_gdf.within(county_poly_buffered)]

    # if there are multiple matches, simply take the second for now. !! still need better duplicate distinction.
    county_merged_gdf = county_merged_gdf.drop_duplicates(subset=['loc_id'], keep='first')
    loc_no = county_merged_gdf.shape[0]
    print(f'''There are {loc_no} locations within the county after duplicates are dropped''')

    # merge data into a single entry for these counties where voronoi does not work
    if county in ['altona','magdeburg stadtkreis']:
        county_merged_gdf = county_merged_gdf[county_merged_gdf['loc_id']==1]

    data_headers = ['locname','type','pop_male', 'pop_female', 'pop_tot','protestant','catholic','other_christ', 'jew', 'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate', 'Kr']
    # convert all data to proportion of population
    for data in data_headers:
        if data in ['pop_tot', 'type', 'locname', 'Kr']:
            continue
        county_merged_gdf[data] = county_merged_gdf[data] / county_merged_gdf['pop_tot']
        county_merged_gdf.loc[county_merged_gdf[data] > 1, data] = 1
    # add child to mother ratio:
    county_merged_gdf['child_per_woman'] = county_merged_gdf['age_under_ten'] / county_merged_gdf['pop_female']

    # plot individual voronoi plots
    if showFlag and county_merged_gdf.shape[0] != 1:
        for header in plot_headers:
            # plot voronoi
            ax = gplt.voronoi(county_merged_gdf, hue=header, clip=county_gdf.simplify(0.001), legend = True, zorder=1, linewidth=0.5)

            # add points over the top if county is selected
            if KREIS != None:
                # convert input KREIS to plotable data
                legend = []
                for i in range(len(KREIS)):
                    county_merged_gdf.loc[county_merged_gdf['Kr'] == KREIS[i], 'kreis_hue'] = i
                    if county_merged_gdf.loc[county_merged_gdf['Kr'] == KREIS[i]].shape[0] !=0:
                        legend.append(KREIS[i])
                # presuming some relevant counties were input, continue with plot.
                if len(legend)!=0:
                    # configure legends and plot
                    scheme = mc.FisherJenks(county_merged_gdf.loc[county_merged_gdf['kreis_hue'].notnull(), 'kreis_hue'], k=len(legend))
                    gplt.pointplot(county_merged_gdf[county_merged_gdf['kreis_hue'].notnull()], ax=ax, zorder=2, hue='kreis_hue', cmap = 'Reds', scheme=scheme, legend=True, legend_labels=legend)
                    county_merged_gdf.drop(columns=["kreis_hue"], inplace=True)

            # plot county as border to reframe the image.
            gplt.polyplot(county_gdf, ax=ax,linewidth=0.8)

            # set title
            ax.set_title(county + ' - ' + header)

    return county_gdf, county_merged_gdf

def run_maps():
    # read in map of prussia
    prussia_map = gpd.read_file("GIS/1871_county_shapefile-new.shp")
    # convert to longitude and latitude for printing
    prussia_map = prussia_map.to_crs(epsg=4326)
    flag = False

    # load saved data frame containing census file
    df_census = pd.read_pickle("census_df_pickle")
    # account for two different rotenburgs:
    df_census.loc[(df_census['county'] == 'rotenburg') & (df_census['regbez'] == 'kassel'), 'county'] = 'rotenburg kassel'
    df_census.loc[(df_census['county'] == 'rotenburg') & (df_census['regbez'] == 'stade'), 'county'] = 'rotenburg stade'

    # extract county name data frame from census
    df_counties = extract_county_names(df_census)

    # find county name from map:
    map_names = extract_map_names(df_counties, prussia_map)

    # case where all counties are wanted
    if COUNTIES == 'all':
        showFlag = False
        # read in merged data
        merge_details = pd.read_excel("Merged_Data/MergeDetails.xlsx")
        counties = []
        for county in merge_details['county'].head:
            counties.append(county)
    else:
        counties = COUNTIES
        showFlag = True

    count = 0
    county_gdf_list = []
    merged_gdf_list = []
    for county in counties:
        census_no = df_census[df_census['county'] == county].shape[0]
        print(f'''\nThere are {census_no} entries in the census for {county}''')

        county_gdf, county_merged_gdf = plot_county(county, PLOT_HEADERS, prussia_map, showFlag, map_names)
        merged_gdf_list.append(county_merged_gdf)
        county_gdf_list.append(county_gdf)

    flag = True
    for merged_gdf in merged_gdf_list:
        if flag:
            concat_merged_gdf = merged_gdf
            flag = False
        else:
            concat_merged_gdf = pd.concat([concat_merged_gdf, merged_gdf], ignore_index=True)


    for header in PLOT_HEADERS:
        ax = gplt.polyplot(prussia_map, linewidth=0.8, zorder=2)
        concat_merged_gdf = concat_merged_gdf[concat_merged_gdf[header].notnull()]
        scheme = mc.HeadTailBreaks(concat_merged_gdf[header])
        for i in range(len(county_gdf_list)):
            if merged_gdf_list[i].shape[0]>1:

                gplt.voronoi(merged_gdf_list[i], hue=header, clip=county_gdf_list[i].simplify(0.001), legend=True, linewidth=0.2, zorder=1, ax=ax, scheme=scheme)
            else:
                merged_gdf_list[i]['geometry'] = county_gdf_list[i]['geometry'].iloc[0]
                gplt.choropleth(merged_gdf_list[i], linewidth=0.2, zorder=1, ax=ax, hue = header, scheme=scheme, legend=True)
        gplt.polyplot(prussia_map, linewidth=0.8, ax=ax, zorder=2)
        ax.set_title('All Counties - ' + header)
    plt.show()

run_maps()





