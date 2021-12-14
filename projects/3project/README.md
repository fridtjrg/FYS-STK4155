# Project 3
##	Description
This folder contains all the code used for project3, where we compare the use of machine larning methods to conventional methods when solving differential equations. For this project we chose to solve the one-dimensional diffusion equation. Below you can find the description of the folder structure. At the bottom of this file, you can find instructions on running the code.

## Folder structure

### `figures/` folder
This folder is the default save location for all plots. Here you can find all the figures that were included in the report, and some that were not.

### `src/` folder
This folder contains source code used by other programs. The python scripts in the `src/` folder is not intended to be run alone. 

### `cv.py` file
This script perfomrs cross-validation on OLS, Ridge and Lasso regression, and saves the results in the `figures/` folder. Note that this file imports the regression library from `project1`, if you did not clone the entire repository, you might have some trouble trying to run this script.

### `explicit_scheme.py` file
This script solves the diffusion equation using Eulers explicit scheme, and saves the results in the `figures/` folder.

### `linreg.py` file
This script solves the diffusion equation using Ridge linear regression. It plots the result and the difference from the analytical result and saves them in the `figures/` folder.

### `mainEigenValue.py` file
This script solves eigenvalue problems using a neural network built using tensorflow. However, it imports the core algorithm from `src/EigenValue.py`. The plots are saved in the `figures/` folder.

### `mainNN.py` file
This scripts solves the diffusion equation using a neural network. To build the neural network it imports the `src/NeuralNetworks.py` file. All plots are then saved to `figures/`. Note that this script take three arguments: Number of datapoints in x-direction, Number of datapoints in y-direction and the number of epochs. 

## Running and reproducing plots
The makefile will help you run the different scripts. Note that some of the scipts contains some random elements, which might cause your plots to differ slightly. The `mainNN.py` file is run twiche takes three arguments which are '20 20 500' and '50 50 100', this is done automatically by the makefile. The neural networks takes a long time to run, so you can also opt out of running them if you want to. The only plot that is not produced automatically is the NN-plot when using the tanh activation function, as this plot was made by changing the source code. 

If you want to reproduce all the plots, you can do this by using the following command:

`$ make all`

If you want to reproduce all the plots except the neural network, use the following commant:

`$ make all-NN`

If you only want to reproduce the neural network plots using the default paramaters, use:

`$ make NN`
