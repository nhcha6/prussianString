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

county = "fraustadt"
county_upper = county.upper()

# set working directory path as location of data
wdir = '/Users/nicolaschapman/Documents/PrussianStringMatching/Data/'

# read in map of prussia
prussia_map = gpd.read_file(wdir+"PrussianCensus1871/GIS/1871_county_shapefile-new.shp")
# convert to longitude and latitude for printing
prussia_map = prussia_map.to_crs(epsg=4326)

# read in merged data
fraustadt_merged_df = pd.read_excel(wdir+"PrussianCensus1871/Fraustadt/Posen-Fraustadt-kreiskey-134-merged.xlsx")

# we only want entries with long-lat data
fraustadt_merged_df = fraustadt_merged_df[fraustadt_merged_df['lat']!=0]

# extract county poly
county_gdf = prussia_map[prussia_map['NAME']==county_upper]
county_gdf.index = range(0,county_gdf.shape[0])
county_poly = county_gdf.loc[0,'geometry']

# if there are multiple matches, simply take the second for now. !! still need better duplicate distinction.
fraustadt_merged_df = fraustadt_merged_df.drop_duplicates(subset=['loc_id'], keep='first')
print(f'''There are {fraustadt_merged_df.shape[0]} locations after duplicates are dropped''')

# build up set of indices for locations that are within the county. We want to include locations that are very close,
# even if they sit slightly outside.
index_in_county = set()
noisy_df = fraustadt_merged_df
for i in range(20):
    # generate small amount of noise to possibly push a point slightly outside, inside. The larger the deviation, the
    # further outside a point can be. Repeat 20 times to ensure required randomness.
    noisy_df = noisy_df.assign(lat = np.random.normal(fraustadt_merged_df['lat'],0.02))
    noisy_df = noisy_df.assign(lng = np.random.normal(fraustadt_merged_df['lng'],0.02))
    noisy_gdf = gpd.GeoDataFrame(noisy_df, geometry=gpd.points_from_xy(noisy_df.lng, noisy_df.lat))
    noisy_gdf.crs = {'init': 'epsg:4326'}
    within = noisy_gdf[noisy_gdf.within(county_poly)]
    for j in within.index:
        index_in_county.add(j)
print(f'''There are {len(index_in_county)} locations within the county''')

# update to only keep the locations deemed to be within the county
fraustadt_merged_df = fraustadt_merged_df.reindex(index_in_county)

# add a little bit of noise to ensure that identical data points are split slightly
fraustadt_merged_df['lat'] = np.random.normal(fraustadt_merged_df['lat'],0.01)
fraustadt_merged_df['lng'] = np.random.normal(fraustadt_merged_df['lng'],0.01)

data_headers = ['locname','type','pop_male', 'pop_female', 'pop_tot','protestant','catholic','other_christ', 'jew', 'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate']

# convert all data to proportion of population
for data in data_headers:
    if data in ['pop_tot', 'type', 'locname']:
        continue
    fraustadt_merged_df[data] = fraustadt_merged_df[data]/fraustadt_merged_df['pop_tot']

# convert to geo data frame
fraustadt_merged_gdf = gpd.GeoDataFrame(fraustadt_merged_df, geometry=gpd.points_from_xy(fraustadt_merged_df.lng,fraustadt_merged_df.lat))
fraustadt_merged_gdf.crs = {'init': 'epsg:4326'}

# plot
# ax = gplt.voronoi(fraustadt_merged_gdf, clip=county_gdf.simplify(0.001))
# gplt.pointplot(fraustadt_merged_gdf, ax=ax)
#
# gplt.voronoi(fraustadt_merged_gdf, hue='protestant', clip=county_gdf.simplify(0.001), legend = True)
# gplt.voronoi(fraustadt_merged_gdf, hue='literate', clip=county_gdf.simplify(0.001), legend = True)

ax = gplt.polyplot(county_gdf)
gplt.polyplot(county_gdf.buffer(0.05),ax=ax)

plt.show()

