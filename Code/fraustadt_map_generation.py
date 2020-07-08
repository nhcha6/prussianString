# file for producing a map of fraustadt data
import numpy as np
import pandas as pd
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt

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
# !! find fraustadt geometric object and check a point is within.
# if there are multiple matches, simply take the second for now. !! still need better duplicate distinction.
fraustadt_merged_df = fraustadt_merged_df.drop_duplicates(subset=['loc_id'], keep='last')
# add a little bit of noise to ensure that identical data points are split slightly
fraustadt_merged_df['lat'] = np.random.normal(fraustadt_merged_df['lat'],0.01)
fraustadt_merged_df['lng'] = np.random.normal(fraustadt_merged_df['lng'],0.01)


fraustadt_merged_gdf = gpd.GeoDataFrame(fraustadt_merged_df, geometry=gpd.points_from_xy(fraustadt_merged_df.lng,fraustadt_merged_df.lat))

# plot prussia base map and fraustadt points
fraustadt_plot = gplt.pointplot(fraustadt_merged_gdf, hue='class', legend=True)
gplt.polyplot(prussia_map, ax=fraustadt_plot)
plt.show()

