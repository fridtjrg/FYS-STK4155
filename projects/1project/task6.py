from regan import *
from imageio import imread



# Load the terrain
terrain = imread('./DataFiles/SRTM_data_Norway_2.tif')

run_plot_compare(terrain,'terrain in Norway', N=50, n_lambdas=30, k=5,poly_degree = 15)