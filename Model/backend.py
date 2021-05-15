import numpy as np
import time
import math
import os
import sys
import yaml
import shutil
import re
import getopt
import igraph
from numba import jit, cuda, prange
from functools import wraps


class Graph(igraph.Graph):
    """ This class extends the Graph class from igraph adding
        additional instance variables for the bin/bucket sort.
    """
    def __init__(self, *args, **kwargs):
        # call constructor for igraph
        igraph.Graph.__init__(self, *args, **kwargs)

        # these values are used in the bin/bucket sort for finding neighbors and are frequently updated
        self.max_neighbors = 5    # the current number of neighbors that can be stored in a holder array
        self.max_agents = 5     # the current number of agents that can be stored in a bin


class Paths:
    """ Hold any important paths for a particular simulation.
    """
    def __init__(self, name, output_path):
        # how file separator
        self.separator = os.path.sep

        # some paths
        self.main_path = output_path + name + self.separator   # the path to the main directory for this simulation
        self.templates = os.path.abspath("templates") + self.separator   # the path to the .txt template directory

        # these directories are sub-directories under the main simulation directory
        general = self.main_path + name
        self.images = general + "_images" + self.separator   # the images output directory
        self.values = general + "_values" + self.separator   # the cell array values output directory
        self.gradients = general + "_gradients" + self.separator    # the gradients output directory
        self.tda = general + "_tda" + self.separator    # the topological data analysis output directory


@jit(nopython=True, cache=True)
def assign_bins_jit(number_agents, bin_locations, bins, bins_help):
    """ A just-in-time compiled function for assign_bins() that places
        the cells in their respective bins.
    """
    # go through all cells
    for index in range(number_agents):
        # get the indices of the generalized cell location
        x, y, z = bin_locations[index]

        # use the help array to get the new index for the cell in the bin
        place = bins_help[x][y][z]

        # adds the index in the cell array to the bin
        bins[x][y][z][place] = index

        # update the number of cells in a bin
        bins_help[x][y][z] += 1

    # return the arrays now filled with cell indices
    return bins, bins_help


@cuda.jit
def get_neighbors_gpu(bin_locations, locations, bins, bins_help, distance, edge_holder, if_edge, edge_count,
                      max_neighbors):
    """ A just-in-time compiled cuda kernel for the get_neighbors()
        method that performs the actual calculations.
    """
    # get the index in the array
    focus = cuda.grid(1)

    # get the starting index for writing to the edge holder array
    start = focus * max_neighbors[0]

    # double check that focus index is within the array
    if focus < bin_locations.shape[0]:
        # holds the total amount of edges for a given cell
        cell_edge_count = 0

        # get the bin location of the cell
        x, y, z = bin_locations[focus]

        # go through the surrounding bins including the bin the cell is in
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if that cell is within the search radius and only continue if the current cell
                        # has a higher index to prevent double counting edges
                        if magnitude(locations[focus], locations[current]) <= distance[0] and focus < current:
                            # if less than the max edges, add the edge
                            if cell_edge_count < max_neighbors[0]:
                                # get the index for the edge
                                index = start + cell_edge_count

                                # update the edge array and identify that this edge exists
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_edge[index] = 1

                            # increase the count of edges for a cell and the index for the next edge
                            cell_edge_count += 1

        # update the array with number of edges for the cell
        edge_count[focus] = cell_edge_count


@jit(nopython=True, parallel=True, cache=True)
def get_neighbors_cpu(number_agents, bin_locations, locations, bins, bins_help, distance, edge_holder, if_edge,
                      edge_count, max_neighbors):
    """ A just-in-time compiled function for the get_neighbors()
        method that performs the actual calculations.
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_agents):
        # get the starting index for writing to the edge holder array
        start = focus * max_neighbors

        # holds the total amount of edges for a given cell
        cell_edge_count = 0

        # get the bin location of the cell
        x, y, z = bin_locations[focus]

        # go through the surrounding bins including the bin the cell is in
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current potential neighbor
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if that cell is within the search radius and only continue if the current cell
                        # has a higher index to prevent double counting edges
                        if np.linalg.norm(locations[current] - locations[focus]) <= distance and focus < current:
                            # if less than the max edges, add the edge
                            if cell_edge_count < max_neighbors:
                                # get the index for the edge
                                index = start + cell_edge_count

                                # update the edge array and identify that this edge exists
                                edge_holder[index][0] = focus
                                edge_holder[index][1] = current
                                if_edge[index] = 1

                            # increase the count of edges for a cell and the index for the next edge
                            cell_edge_count += 1

        # update the array with number of edges for the cell
        edge_count[focus] = cell_edge_count

    return edge_holder, if_edge, edge_count


@cuda.jit
def jkr_forces_gpu(jkr_edges, delete_edges, locations, radii, jkr_forces, poisson, youngs, adhesion_const):
    """ A just-in-time compiled cuda kernel for the jkr_forces()
        method that performs the actual calculations.
    """
    # get the index in the edges array
    edge_index = cuda.grid(1)

    # double check that index is within the array
    if edge_index < jkr_edges.shape[0]:
        # get the cell indices of the edge
        cell_1 = jkr_edges[edge_index][0]
        cell_2 = jkr_edges[edge_index][1]

        # get the locations of the two cells
        location_1 = locations[cell_1]
        location_2 = locations[cell_2]

        # get the magnitude of the distance between the cells
        mag = magnitude(location_1, location_2)

        # get the overlap of the cells
        overlap = radii[cell_1] + radii[cell_2] - mag

        # get two values used for JKR calculation
        e_hat = (((1 - poisson[0] ** 2) / youngs[0]) + ((1 - poisson[0] ** 2) / youngs[0])) ** -1
        r_hat = ((1 / radii[cell_1]) + (1 / radii[cell_2])) ** -1

        # value used to calculate the max adhesive distance after bond has been already formed
        overlap_ = (((math.pi * adhesion_const[0]) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

        # get the nondimensionalized overlap
        d = overlap / overlap_

        # check to see if the cells will have a force interaction based on the nondimensionalized distance
        if d > -0.360562:
            # plug the value of d into polynomial approximation for nondimensionalized force
            f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

            # convert from the nondimensionalized force to find the JKR force
            jkr_force = f * math.pi * adhesion_const[0] * r_hat

            # loops over all directions of space
            for i in range(3):
                # get the vector by axis between the two cells
                vector = location_1[i] - location_2[i]

                # if the magnitude is 0 use the zero vector, otherwise find the normalized vector for each axis
                if mag != 0:
                    normal = vector / mag
                else:
                    normal = 0

                # adds the adhesive force as a vector in opposite directions to each cell's force holder
                jkr_forces[cell_1][i] += jkr_force * normal
                jkr_forces[cell_2][i] -= jkr_force * normal

        # remove the edge if the it fails to meet the criteria for distance, simulating that the bond is broken
        else:
            delete_edges[edge_index] = 1


@jit(nopython=True, parallel=True, cache=True)
def jkr_forces_cpu(number_edges, jkr_edges, delete_edges, locations, radii, jkr_forces, poisson, youngs,
                   adhesion_const):
    """ A just-in-time compiled function for the jkr_forces()
        method that performs the actual calculations.
    """
    # go through the edges array
    for edge_index in prange(number_edges):
        # get the cell indices of the edge
        cell_1 = jkr_edges[edge_index][0]
        cell_2 = jkr_edges[edge_index][1]

        # get the vector between the centers of the cells and the magnitude of this vector
        vector = locations[cell_1] - locations[cell_2]
        mag = np.linalg.norm(vector)

        # get the overlap of the cells
        overlap = (radii[cell_1] + radii[cell_2] - mag) / 1000000

        # get two values used for JKR calculation
        e_hat = (((1 - poisson ** 2) / youngs) + ((1 - poisson ** 2) / youngs)) ** -1
        r_hat = ((1000000 / radii[cell_1]) + (1000000 / radii[cell_2])) ** -1

        # value used to calculate the max adhesive distance after bond has been already formed
        overlap_ = (((math.pi * adhesion_const) / e_hat) ** (2 / 3)) * (r_hat ** (1 / 3))

        # get the nondimensionalized overlap
        d = overlap / overlap_

        # check to see if the cells will have a force interaction based on the nondimensionalized distance
        if d > -0.360562:
            # plug the value of d into polynomial approximation for nondimensionalized force
            f = (-0.0204 * d ** 3) + (0.4942 * d ** 2) + (1.0801 * d) - 1.324

            # convert from the nondimensionalized force to find the JKR force
            jkr_force = f * math.pi * adhesion_const * r_hat

            # if the magnitude is 0 use the zero vector, otherwise find the normalized vector for each axis. numba's
            # jit prefers a reduction instead of generating a new normalized array
            normal = np.array([0.0, 0.0, 0.0])
            if mag != 0:
                normal += vector / mag

            # adds the adhesive force as a vector in opposite directions to each cell's force holder
            jkr_forces[cell_1] += jkr_force * normal
            jkr_forces[cell_2] -= jkr_force * normal

        # remove the edge if the it fails to meet the criteria for distance, simulating that the bond is broken
        else:
            delete_edges[edge_index] = 1

    return jkr_forces, delete_edges


@cuda.jit
def apply_forces_gpu(jkr_force, motility_force, locations, radii, viscosity, size, move_dt):
    """ A just-in-time compiled cuda kernel for the apply_forces()
        method that performs the actual calculations.
    """
    # get the index in the array
    index = cuda.grid(1)

    # double check that index is within the array
    if index < locations.shape[0]:
        # stokes law for velocity based on force and fluid viscosity (friction)
        stokes_friction = 6 * math.pi * viscosity[0] * radii[index]

        # loop over all directions of space
        for i in range(3):
            # update the velocity of the cell based on stokes
            velocity = (jkr_force[index][i] + motility_force[index][i]) / stokes_friction

            # set the new location
            new_location = locations[index][i] + velocity * move_dt[0]

            # check if new location is in the simulation space, if not set values at space limits
            if new_location > size[i]:
                locations[index][i] = size[i]
            elif new_location < 0:
                locations[index][i] = 0
            else:
                locations[index][i] = new_location


@jit(nopython=True, parallel=True, cache=True)
def apply_forces_cpu(number_agents, jkr_force, motility_force, locations, radii, viscosity, size, move_dt):
    """ A just-in-time compiled function for the apply_forces()
        method that performs the actual calculations.
    """
    # loop over all cells
    for i in prange(number_agents):
        # stokes law for velocity based on force and fluid viscosity (friction)
        stokes_friction = 6 * math.pi * viscosity * radii[i] / 1000000

        # update the velocity of the cell based on stokes
        # velocity = (motility_force[i] + jkr_force[i]) / stokes_friction
        velocity = (motility_force[i] + jkr_force[i]) / stokes_friction

        # set the new location
        new_location = locations[i] + velocity * move_dt * 1000000

        # loop over all directions of space
        for j in range(0, 3):
            # check if new location is in the space, if not return it to the space limits
            if new_location[j] > size[j]:
                locations[i][j] = size[j]
            elif new_location[j] < 0:
                locations[i][j] = 0
            else:
                locations[i][j] = new_location[j]

    return locations


@jit(nopython=True, cache=True)
def update_diffusion_jit(base, steps, diffuse_dt, last_dt, diffuse_const, spat_res2):
    """ A just-in-time compiled function for update_diffusion()
        that performs the actual diffusion calculation.
    """
    # holder the following constant for faster computation, given that dx and dy match
    a = diffuse_dt * diffuse_const / spat_res2
    b = 1 - 4 * a

    # finite difference to solve laplacian diffusion equation, currently 2D
    for i in range(steps):
        # on the last step apply smaller diffuse dt if step dt doesn't divide nicely
        if i == steps - 1:
            a = last_dt * diffuse_const / spat_res2
            b = 1 - 4 * a

        # set the initial conditions by reflecting the edges of the gradient
        base[:, 0] = base[:, 1]
        base[:, -1] = base[:, -2]
        base[0, :] = base[1, :]
        base[-1, :] = base[-2, :]

        # get the morphogen addition for the diffusion points, based on other points and hold this
        temp = a * (base[2:, 1:-1] + base[:-2, 1:-1] + base[1:-1, 2:] + base[1:-1, :-2])

        # get the diffusion loss for the diffusion points
        base[1:-1, 1:-1] *= b

        # add morphogen change from the temporary array
        base[1:-1, 1:-1] += temp

    # return the gradient back without the edges
    return base[1:-1, 1:-1]


@cuda.jit(device=True)
def magnitude(vector_1, vector_2):
    """ A just-in-time compiled cuda kernel device function
        for getting the distance between two vectors.
    """
    # loop over the axes add the squared difference
    total = 0
    for i in range(0, 3):
        total += (vector_1[i] - vector_2[i]) ** 2

    # return the sqrt of the total
    return total ** 0.5


def check_direct(path):
    """ Check directory for simulation outputs.
    """
    # if it doesn't exist make directory
    if not os.path.isdir(path):
        os.mkdir(path)

    # optionally return the path
    return path


def sort_naturally(file_list):
    """ Key for sorting the file list based on the step number.
    """
    return int(re.split('(\d+)', file_list)[-2])


def progress_bar(progress, maximum):
    """ Make a progress bar because progress bars are cool.
    """
    # length of the bar
    length = 60

    # calculate bar and percent
    progress += 1    # start at 1 not 0
    fill = int(length * progress / maximum)
    bar = '#' * fill + '.' * (length - fill)
    percent = int(100 * progress / maximum)

    # output the progress bar
    print(f"\r[{bar}] {percent}%", end="")


def normal_vector(vector):
    """ Returns the normalized vector, sadly this does not
        exist in NumPy.
    """
    # get the magnitude of the vector
    mag = np.linalg.norm(vector)

    # if magnitude is 0 return zero vector, otherwise divide by the magnitude
    if mag == 0:
        return np.zeros(3)
    else:
        return vector / mag


def record_time(function):
    """ A decorator used to time individual methods.
    """
    @wraps(function)
    def wrap(simulation, *args, **kwargs):  # args and kwargs are for additional arguments
        # get the start/end time and call the method
        start = time.perf_counter()
        function(simulation, *args, **kwargs)
        end = time.perf_counter()

        # add the time to the dictionary holding these times
        simulation.method_times[function.__name__] = end - start

    return wrap


# ---------------------------------------- helper methods for user-interface ------------------------------------------
def commandline_param(flag, dtype):
    """ Returns the value for option passed at the
        command line.
    """
    # get list of command line arguments
    args = sys.argv

    # go through the arguments
    for i in range(len(args)):
        # if argument matches flag
        if args[i] == flag:
            # try to return value of
            try:
                return dtype(args[i + 1])
            # otherwise raise error
            except IndexError:
                raise Exception(f"No value for option: {args[i]}")

    # return NoneType if no value passed
    return None


def template_params(path):
    """ Return parameters as dict from YAML template file.
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


def output_dir():
    """ Read the output path from paths.yaml and if this directory
        does not exist yet, make it.
    """
    # get file separator
    separator = os.path.sep

    # open the file and load the keys
    with open("paths.yaml", "r") as file:
        keys = yaml.safe_load(file)

    # get output_path key
    output_path = keys["output_path"]

    # keep running until output directory exists
    while not os.path.isdir(output_path):
        # prompt user input
        print("\nSimulation output directory: \"" + output_path + "\" does not exist!")
        user = input("Do you want to make this directory? If \"n\", you can specify the correct path (y/n): ")
        print()

        # if not making this directory
        if user == "n":
            # get new path to output directory
            output_path = input("Correct path (absolute) to output directory: ")

            # update paths.yaml file with new output directory path
            keys["output_path"] = output_path
            with open("paths.yaml", "w") as file:
                keys = yaml.dump(keys, file)

        # if yes, make the directory
        elif user == "y":
            os.makedirs(output_path)
            break

        else:
            print("Either type \"y\" or \"n\"")

    # if path doesn't end with separator, add it
    if output_path[-1] != separator:
        output_path += separator

    # return correct path to output directory
    return output_path


def get_name_mode():
    """ This function will get the name and mode for the simulation
        either from the command line or a text-based UI.
    """
    # try to get the name and mode from the command line
    name = commandline_param("-n", str)
    mode = commandline_param("-m", int)

    # if the name variable has not been initialized by the command-line, run the text-based UI to get it
    if name is None:
        while True:
            # prompt for the name
            name = input("What is the \"name\" of the simulation? Type \"help\" for more information: ")

            # keep running if "help" is typed
            if name == "help":
                print("\nType the name of the simulation (not a path).\n")
            else:
                break

    # if the mode variable has not been initialized by the command-line, run the text-based UI to get it
    if mode is None:
        while True:
            # prompt for the mode
            mode = input("What is the \"mode\" of the simulation? Type \"help\" for more information: ")
            print()

            # keep running if "help" is typed
            if mode == "help":
                print("Here are the following modes:\n0: New simulation\n1: Continuation of past simulation\n"
                      "2: Turn simulation images to video\n3: Zip previous simulation\n")
            else:
                try:
                    # get the mode as an integer make sure mode exists, break the loop if it does
                    mode = int(mode)

                # if not an integer
                except ValueError:
                    print("Input: \"mode\" should be an integer.\n")

    # return the simulation name and mode
    return name, mode


def check_new_sim(output_path, name):
    """ Check that a new simulation can be made. """
    # get file separator
    separator = os.path.sep

    while True:
        # see if the directory exists
        if os.path.isdir(output_path + name):
            # get user input for overwriting previous simulation
            print("Simulation already exists with name: " + name)
            user = input("Would you like to overwrite that simulation? (y/n): ")
            print()

            # if no overwrite, get new simulation name
            if user == "n":
                name = input("New name: ")
                print()

            # overwrite by deleting all files/folders in previous directory
            elif user == "y":
                # clear current directory to prevent another possible future errors
                files = os.listdir(output_path + name)
                for file in files:
                    # path to each file/folder
                    path = output_path + name + separator + file

                    # delete the file/folder
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                break
            else:
                # inputs should either be "y" or "n"
                print("Either type \"y\" or \"n\"")
        else:
            # if does not exist, make directory
            os.mkdir(output_path + name)
            break


def check_previous_sim(output_path, name):
    """ Makes sure that a previous simulation exists. """
    while True:
        # if the directory exists, break loop
        if os.path.isdir(output_path + name):
            break
        else:
            # try to get correct name
            print("No directory exists with name/path: " + output_path + name)
            name = input("Please type the correct name of the simulation or type \"exit\" to exit: ")
            print()
            if name == "exit":
                exit()


def get_final_step():
    """ Gets the new last step of the simulation if using continuation
        mode.
    """
    # try get step number from commandline
    final_step = commandline_param("-fs", int)

    # if no value, then run UI until found
    if final_step is None:
        while True:
            # prompt for the final step number
            final_step = input("What is the final step of this continued simulation? Type \"help\" for more"
                               " information: ")
            print()

            # keep running if "help" is typed
            if final_step == "help":
                print("Enter the new step number that will be the last step of the simulation.\n")
            else:
                try:
                    # get the final step as an integer, break the loop if conversion is successful
                    final_step = int(final_step)
                    break

                # if not an integer
                except ValueError:
                    print("Input: \"final step\" should be an integer.\n")

    return final_step
