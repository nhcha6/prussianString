# file for producing a map of fraustadt data

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
fraustadt_merged_df = fraustadt_merged_df[fraustadt_merged_df['lat']!=0]
fraustadt_merged_gdf = gpd.GeoDataFrame(fraustadt_merged_df, geometry=gpd.points_from_xy(fraustadt_merged_df.lng,fraustadt_merged_df.lat))

# plot prussia base map
prussia_plot = gplt.polyplot(prussia_map, figsize=(8,4))
gplt.pointplot(fraustadt_merged_gdf, ax=prussia_plot)
fraustadt_plot = gplt.pointplot(fraustadt_merged_gdf)
gplt.polyplot(prussia_map, ax=fraustadt_plot)
plt.show()

