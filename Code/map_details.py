# select the counties you wish to plot: can either choose a list of counties, or range of county id numbers.
#COUNTIES =  ['posen landkreis','breslau landkreis']
# range(x, y+1) will run county_id x to y
COUNTIES = range(1,450)

# True if plots for each individual county are to be produced.
INDIV_PLOTS = False

# select the data to map.
#['pop_male', 'pop_female', 'pop_tot', 'protestant', 'catholic', 'other_christ', 'jew', 'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate', 'child_per_woman']
PLOT_HEADERS = ['catholic']

# locations within these region (from the gazetter data) will be plotted on top of the voronoi for individual counties
#KREIS = ['Memel','Lissa']
# set to None if none are wanted
KREIS = None

# option for manually setting the thresholds of the plot.
BINS = [0, 0.2, 0.4, 0.6, 0.8, 1]
# set to None if want to use the default.
#BINS = None

# Histogram Data Options: ['pop_tot', 'protestant', 'catholic', 'other_christ', 'jew', 'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate']
HISTOGRAM_DATA = ['pop_tot', 'protestant']
# Histogram Subset Options: ['all', 'total', 'stadt', 'manor', 'village']
HISTOGRAM_SUBSET = ['all', 'manor', 'village']