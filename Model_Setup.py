#########################################################
# Name:    Model_Setup                                  #
# Author:  Jack Toppen                                  #
# Date:    2/5/20                                       #
#########################################################
import random as r
import numpy as np
import os
from Model_Simulation import Simulation
from Model_StemCells import StemCell


# where the model will output images and cell locations ex. ("C:\\Python27\\Model")
path = "C:\\Python27\\MEGA-ARRAY"

# Parallel GPU processing? # make sure to uncomment "from numba import cuda" in Model_Simulation and the cuda functions
parallel = False

# total time step counter limit to less than 30
Run_Time = 60.0

# when does the model begin (usually 1)
Start_Time = 1.0

# change in time between time steps
Time_Step = 1.0

# number of initial GATA6 high cells
GATA6_high = 500

# number of initial NANOG high cells
NANOG_high = 500

# stochastic FGFR + ERK values?
stochastic_bool = True

# size of grid can be 3D with extra layers (layers, rows, columns)
size = (1, 1000, 1000)

# define boolean functions here ex. ("(x3+1) * (x4+1)")
funct_1 = "x5"
funct_2 = "x1 * x4"
funct_3 = "x2"
funct_4 = "x5 + 1"
funct_5 = "(x3+1) * (x4+1)"

# add functions to the list
functions = [funct_1, funct_2, funct_3, funct_4, funct_5]

# radius of each cell depending on state ex. ([state_1, state_2])
radius = 6.0

# length of time steps required for a pluripotent cell to divide
pluri_div_thresh = 12.0

# length of time steps required for a differentiated cell to divide
diff_div_thresh = 24.0

# length of time steps required for a pluripotent cell to differentiate
pluri_to_diff = 24.0

# length at which a edge is formed to create springs between cells
spring_max = 13.0

# amount of differentiated cells needed to surround a pluripotent cell and differentiate it
diff_surround = 12.0

# bounds of the simulation
bounds = [[0, 0], [0, 1000], [1000, 1000], [1000, 0]]

# spring constant value
spring_constant = 0.77

# max iterations for optimize
max_itrs = 20

# error maximum for optimize
max_error = 0.00001

# max FGF4 on a patch
max_fgf4 = 10

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def newDirect(path):
    """
    This function opens the specified save path and finds the highest folder number.
    It then returns the next highest number as a name for the currently running simulation.
    """

    files = os.listdir(path)
    file_count = len(files)
    number_files = []
    if file_count > 0:
        for i in range(file_count):
            try:
                number_files.append(float(files[i]))
            except ValueError:
                pass
        if len(number_files) > 0:
            k = max(number_files)
        else:
            k = 0
    else:
        k = 0
    return k + 1.0


# names the file
Model_ID = newDirect(path)

# initializes simulation class which holds all information about the simulation
simulation = Simulation(Model_ID, path, Start_Time, Run_Time, Time_Step, pluri_div_thresh, diff_div_thresh,
                        pluri_to_diff, size, spring_max, diff_surround, functions, max_itrs, max_error, parallel,
                        max_fgf4, bounds, spring_constant)

# loops over all NANOG_high cells and creates a stem cell object for each one with given parameters
for i in range(NANOG_high):
    ID = i
    point = [r.random() * size[1], r.random() * size[2]]
    state = "Pluripotent"
    motion = True
    if stochastic_bool:
        booleans = np.array([r.randint(0, 1), r.randint(0, 1), 0, 1])
    else:
        booleans = np.array([0, 0, 0, 1])

    diff_timer = pluri_to_diff * r.random()
    division_timer = pluri_div_thresh * r.random()

    sim_obj = StemCell(point, radius, ID, booleans, state, diff_timer, division_timer, motion)
    simulation.add_object(sim_obj)
    simulation.inc_current_ID()

# loops over all GATA6_high cells and creates a stem cell object for each one with given parameters
for i in range(GATA6_high):
    ID = i + NANOG_high
    point = [r.random() * size[1], r.random() * size[2]]
    state = "Pluripotent"
    motion = True
    if stochastic_bool:
        booleans = np.array([r.randint(0, 1), r.randint(0, 1), 1, 0])
    else:
        booleans = np.array([0, 0, 1, 0])

    diff_timer = pluri_to_diff * r.random()
    division_timer = pluri_div_thresh * r.random()

    sim_obj = StemCell(point, radius, ID, booleans, state, diff_timer, division_timer, motion)
    simulation.add_object(sim_obj)
    simulation.inc_current_ID()

# runs the model
simulation.run()