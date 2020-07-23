# select the counties you wish to plot: can either choose a list of counties, or range of county id numbers.
#COUNTIES =  ['posen landkreis','breslau landkreis']
COUNTIES = range(1,10)

# True if plots for each individual county are to be produced.
INDIV_PLOTS = True

# select the data to map.
#['pop_male', 'pop_female', 'pop_tot', 'protestant', 'catholic', 'other_christ', 'jew', 'other_relig', 'age_under_ten', 'literate', 'school_noinfo', 'illiterate', 'child_per_woman']
PLOT_HEADERS = ['protestant', 'other_christ']

# locations within these region (from the gazetter data) will be plotted on top of the voronoi for individual counties
KREIS = ['Memel','Lissa']
# set to None if none are wanted
#KREIS = None

# option for manually setting the thresholds of the plot.
#BINS = [0, 0.2, 0.4, 0.6, 0.8, 1]
# set to None if want to use the default.
BINS = None