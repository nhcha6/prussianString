# file for producing a map of fraustadt data
import numpy as np
import pandas as pd
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
from shapely.geometry import shape
from geopandas.tools import sjoin

def plot_on_map(points_gdf, map_gdf):
    # plot prussia base map and fraustadt points
    fraustadt_plot = gplt.pointplot(points_gdf, hue='class', legend=True)
    gplt.polyplot(map_gdf, ax=fraustadt_plot)
    plt.show()

county = "FRAUSTADT"

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
county_gdf = prussia_map[prussia_map['NAME']==county]
county_gdf.index = [0]
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
    noisy_gdf = noisy_gdf.set_crs(epsg=4326)
    within = noisy_gdf[noisy_gdf.within(county_poly)]
    for j in within.index:
        index_in_county.add(j)
print(f'''There are {len(index_in_county)} locations within the county''')

# add a little bit of noise to ensure that identical data points are split slightly
fraustadt_merged_df['lat'] = np.random.normal(fraustadt_merged_df['lat'],0.01)
fraustadt_merged_df['lng'] = np.random.normal(fraustadt_merged_df['lng'],0.01)

# convert to geo data frame
fraustadt_merged_gdf = gpd.GeoDataFrame(fraustadt_merged_df, geometry=gpd.points_from_xy(fraustadt_merged_df.lng,fraustadt_merged_df.lat))
fraustadt_merged_gdf = fraustadt_merged_gdf.set_crs(epsg=4326)

# update to only keen the locations deemed to be within the county
fraustadt_merged_gdf = fraustadt_merged_gdf.loc[index_in_county]

#plot_on_map(fraustadt_merged_gdf, prussia_map)

ax = gplt.voronoi(fraustadt_merged_gdf.head(10))
gplt.polyplot(county_gdf, ax=ax)
plt.show()

