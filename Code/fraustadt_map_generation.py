# file for producing a map of fraustadt data
import numpy as np
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


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

    print('unmatched map')
    for county in prussia_map["NAME"]:
        flag = True
        for maps in county_map_name.values():
            if county in maps:
                flag = False
        if flag:
            print(county)

    print('double map')
    for county, maps in county_map_name.items():
        if len(maps)>1:
            print(county)
            print(maps)

    return county_map_name

def extract_county_names(df_census):
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
    df_counties.loc[df_counties['orig_name'] == 'rotenburg', 'man_name'] = 'zeven'
    df_counties.loc[df_counties['orig_name'] == 'emden', 'man_name'] = 'norden'
    df_counties.loc[df_counties['orig_name'] == 'aurich', 'man_name'] = 'wittmund'
    df_counties.loc[df_counties['orig_name'] == 'marienburg b.H.', 'man_name'] = 'alfeld'
    df_counties.loc[df_counties['orig_name'] == 'hagen', 'man_name'] = 'schwelm'
    df_counties.loc[df_counties['orig_name'] == 'celle', 'man_name'] = 'burgdorf'
    df_counties.loc[df_counties['orig_name'] == 'inowraclaw', 'man_name'] = 'inowrazlaw'
    df_counties.loc[df_counties['orig_name'] == 'langensalza', 'man_name'] = 'weimar'
    df_counties.loc[df_counties['orig_name'] == 'rinteln', 'man_name'] = 'grafschaft schaumburg'


    # strip all [name, alt_name, suffix] of white spaces
    # df_master.replace(np.nan, '', regex=True)
    for c in ['simp_name']:
        df_counties[c] = df_counties[c].str.strip()

    return df_counties

# set working directory path as location of data
wdir = '/Users/nicolaschapman/Documents/PrussianStringMatching/Data/'

def plot_county(county):
    # load saved data frame containing census file
    df_census = pd.read_pickle(wdir + "census_df_pickle")
    census_no = df_census[df_census['county']==county].shape[0]
    print(f'''There are {census_no} entries in the census''')

    # read in map of prussia
    prussia_map = gpd.read_file(wdir+"PrussianCensus1871/GIS/1871_county_shapefile-new.shp")
    # convert to longitude and latitude for printing
    prussia_map = prussia_map.to_crs(epsg=4326)

    # extract county name data frame from census
    df_counties = extract_county_names(df_census)

    # find county name from map:
    map_names = extract_map_names(df_counties, prussia_map)

    # read in merged data
    try:
        county_merged_df = pd.read_excel(wdir+"Output/" + county + "/Merged_Data_" + county + '.xlsx')
    except FileNotFoundError:
        print('nope')
        return
    county_merged_df.drop(columns=["geometry"], inplace=True)

    # we only want entries with long-lat data
    county_merged_df = county_merged_df[(county_merged_df['lat']!=0)&(county_merged_df['lat'].notnull())]

    # extract county poly, need to loop not pop.
    county_gdf = prussia_map[prussia_map['NAME']==map_names[county].pop()]
    county_gdf.index = range(0,county_gdf.shape[0])
    # if there are multiple regions in the map data with the same name, extract the correct one.
    if county_gdf.shape[0]>1:
        if county == 'koenigsberg':
            county_gdf = county_gdf[(county_gdf.index == 1) | (county_gdf.index == 2)]
        if county == 'koenigsberg in der neumark':
            county_gdf = county_gdf[county_gdf.index ==0]
    county_poly_buffered = county_gdf.buffer(0.05)[0]

    # add a little bit of noise to ensure that identical data points are split slightly
    county_merged_df['lat'] = np.random.normal(county_merged_df['lat'],0.01)
    county_merged_df['lng'] = np.random.normal(county_merged_df['lng'],0.01)

    data_headers = ['locname','type','pop_male', 'pop_female', 'pop_tot','protestant','catholic','other_christ', 'jew', 'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate']

    # convert all data to proportion of population
    for data in data_headers:
        if data in ['pop_tot', 'type', 'locname']:
            continue
        county_merged_df[data] = county_merged_df[data]/county_merged_df['pop_tot']

    # convert to geo data frame
    county_merged_gdf = gpd.GeoDataFrame(county_merged_df, geometry=gpd.points_from_xy(county_merged_df.lng,county_merged_df.lat))
    county_merged_gdf.crs = {'init': 'epsg:4326'}

    # if there are multiple matches, simply take the second for now. !! still need better duplicate distinction.
    county_merged_gdf = county_merged_gdf.drop_duplicates(subset=['loc_id'], keep='first')
    loc_no = county_merged_gdf.shape[0]
    print(f'''There are {loc_no} locations after duplicates are dropped''')

    # #drop those outside buffered poly
    county_merged_gdf = county_merged_gdf[county_merged_gdf.within(county_poly_buffered)]
    within_no = county_merged_gdf.shape[0]
    print(f'''There are {within_no} locations within the county''')

    #plot voronoi
    # ax = gplt.voronoi(county_merged_gdf, clip=county_gdf.simplify(0.001))
    # gplt.pointplot(county_merged_gdf, ax=ax)
    # gplt.voronoi(county_merged_gdf, hue='protestant', clip=county_gdf.simplify(0.001), legend = True)
    # gplt.voronoi(county_merged_gdf, hue='literate', clip=county_gdf.simplify(0.001), legend = True)

    # plot points, region, buffered region and map.
    # ax = gplt.pointplot(county_merged_gdf)
    # gplt.polyplot(county_gdf.buffer(0.05),ax=ax)
    # gplt.polyplot(county_gdf, facecolor='red',ax=ax)
    # gplt.polyplot(prussia_map, ax=ax)

    count = 0
    for map in prussia_map["NAME"]:
        count+=1
        county_gdf = prussia_map[prussia_map['NAME'] == map]
        county_gdf.index = range(0, county_gdf.shape[0])
        county_poly_buffered = county_gdf.buffer(0.05)[0]
        # drop those outside buffered poly
        if county_merged_gdf[county_merged_gdf.within(county_poly_buffered)].shape[0]>0:
            print(map)
            print(county_merged_gdf[county_merged_gdf.within(county_poly_buffered)].shape[0])
            bx = gplt.polyplot(county_gdf, facecolor='red')
            gplt.polyplot(prussia_map, ax=bx)

    plt.show()



counties = ['koenigsberg']#, 'posen landkreis', 'bromberg', 'breslau landkreis', 'liegnitz stadtkreis', 'liegnitz landkreis', 'erfurt landkreis', 'husum', 'hannover landkreis', 'rotenburg', 'muenster landkreis', 'recklinghausen', 'arnsberg', 'kassel landkreis', 'frankfurt am main', 'essen landkreis', 'solingen', 'gladbach', 'muelheim am rhein', 'koeln stadtkreis', 'trier landkreis', 'aachen landkreis', 'marienburg i. pr. ', 'neustadt i. pr. ', 'rosenberg i. pr.']

for county in counties:
    plot_county(county)



