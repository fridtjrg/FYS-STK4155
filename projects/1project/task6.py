from math import trunc

from numpy.core.numeric import True_
from regan import *
from imageio import imread



# Load the terrain
terrain = imread('./DataFiles/SRTM_data_Norway_2.tif')

run_plot_compare(terrain,'terrain in Norway', 100, N=50, n_lambdas=30, k=5,poly_degree = 20,plot=True,saveplots=True_)