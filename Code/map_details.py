from itertools import chain
# SELECT THE COUNTIES YOU WISH TO PLOT: CAN EITHER CHOOSE A LIST OF COUNTIES, OR RANGE OF COUNTY ID NUMBERS.

# Here use 1871 counties
#COUNTIES =  ['fraustadt', 'koenigsberg stadt']
#COUNTIES =  ['jerichow I']

# range(x, y+1) will run county_id x to y
COUNTIES = range(125,140)

# list(range(x, y+1)) + list(range(a, b+1)) will run from county_d x to y and then from a to b.
#COUNTIES = list(range(1,40)) + list(range(250,300))

# TRUE IF PLOTS FOR EACH INDIVIDUAL COUNTY ARE TO BE PRODUCED.
# INDIV_PLOTS = False
INDIV_PLOTS = True

# select the data to map.
#['pop_male', 'pop_female', 'pop_tot', 'protestant', 'catholic', 'other_christ', 'jew', 'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate', 'child_per_woman']
PLOT_HEADERS = ['other_christ', 'jew']

# locations within these region (from the gazetter data) will be plotted on top of the voronoi for individual counties
# here use Mayers gazette county names = later counties
KREIS = ['Fraustadt','Lissa']
# set to None if none are wanted
#KREIS = None

# option for manually setting the thresholds of the plot.
# BINS = [0.2, 0.4, 0.6, 0.8, 1]
#BINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# set to None if want to use the default.
BINS = None

# Histogram Data Options: ['pop_tot', 'protestant', 'catholic', 'other_christ', 'jew', 'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate']
HISTOGRAM_DATA = ['pop_tot', 'protestant', 'catholic', 'other_christ', 'jew']
# Histogram Subset Options: ['all', 'total', 'stadt', 'manor', 'village']
HISTOGRAM_SUBSET = ['all', 'manor', 'village']