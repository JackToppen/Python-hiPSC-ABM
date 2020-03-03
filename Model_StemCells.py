#########################################################
# Name:    Model_StemCells                              #
# Author:  Jack Toppen                                  #
# Date:    2/15/20                                      #
#########################################################
import math as math
import numpy as np
import random as r


class StemCell(object):
    """ Every cell object in the simulation
        will have this class
    """
    def __init__(self, ID, location, motion, mass, nuclear_radius, cytoplasm_radius, booleans, state, diff_timer,
                 division_timer, death_timer):

        """ location: where the cell is located on the grid "[x,y]"
            radius: the radius of each cell
            ID: the number assigned to each cell "0-number of cells"
            booleans: array of boolean values for each boolean function
            state: whether the cell is pluripotent or differentiated
            diff_timer: counts the total steps per cell needed to differentiate
            division_timer: counts the total steps per cell needed to divide
            motion: whether the cell is in moving or not "True or False"
            spring_constant: strength of force applied based on distance
        """
        self.ID = ID
        self.location = location
        self.motion = motion
        self.mass = mass
        self.nuclear_radius = nuclear_radius
        self.cytoplasm_radius = cytoplasm_radius
        self.booleans = booleans
        self.state = state
        self.diff_timer = diff_timer
        self.division_timer = division_timer
        self.death_timer = death_timer

        # holds the compression force by a cell's neighbors
        self.compress = 0

        # holds the value of the movement vector of the cell
        self._disp_vec = np.array([0.0, 0.0])

        self.velocity = np.array([0.0, 0.0])



#######################################################################################################################

    def add_displacement_vec(self, vec):
        """ Adds a vector to the vector representing the movement of the cell
        """
        self._disp_vec += vec


    def update_constraints(self, sim):
        """ Updates the boundary constraints of the grid on the objects
            and limits the movement to a magnitude of 5
        """

        # new location is the sum of previous location and movement vector
        location = self.location + self._disp_vec

        # if there are bounds, this will check to see if new location is in the grid
        if len(sim.bounds) > 0:
            if sim.boundary.contains_point(location[0:2]):
                self.location = location

            # if the new location is not in the grid, try opposite
            else:
                new_loc = self.location - self._disp_vec
                if sim.boundary.contains_point(new_loc[0:2]):
                    self.location = new_loc
        else:
            self.location = location

        # resets the movement vector to [0,0]
        self._disp_vec = np.array([0.0, 0.0])


    def boolean_function(self, sim, fgf4_bool):
        """ preforms the functions defined in model setup as strings
        """
        # call functions from model setup
        function_list = sim.functions

        # xn is equal to the value corresponding to its function
        x1 = fgf4_bool
        x2 = self.booleans[0]
        x3 = self.booleans[1]
        x4 = self.booleans[2]
        x5 = self.booleans[3]

        # evaluate the functions by turning them from strings to math equations
        new_1 = eval(function_list[0]) % 2
        new_2 = eval(function_list[1]) % 2
        new_3 = eval(function_list[2]) % 2
        new_4 = eval(function_list[3]) % 2
        new_5 = eval(function_list[4]) % 2

        # updates self.booleans with the new boolean values
        self.booleans = np.array([new_2, new_3, new_4, new_5])

        return new_1


    def diff_surround_funct(self, sim):
        """ if there are enough differentiated cells surrounding
            a pluripotent cell then it will divide
        """
        # finds neighbors of a cell
        neighbors = np.array(list(sim.network.neighbors(self)))
        # counts neighbors that are differentiated and in the interaction distance
        counter = 0
        for i in range(len(neighbors)):
            if neighbors[i].state == "Differentiated":
                dist_vec = neighbors[i].location - self.location
                dist = Mag(dist_vec)
                if dist <= sim.neighbor_distance:
                    counter += 1

        # if there are enough cells surrounding the cell the differentiation timer will increase
        if counter >= sim.diff_surround_value:
            self.diff_timer += 1


    def cell_death(self, sim):
        neighbors = list(sim.network.neighbors(self))
        if len(neighbors) < 1:
            self.death_timer += 1


    def divide(self, sim):
        """ creates new cells if a cell divides shares
            most of the same values and places the cell
            in a random place outside
        """
        # radius of cell
        radius = self.nuclear_radius + self.cytoplasm_radius
        location = self.location

        # if there are boundaries
        if len(sim.bounds) > 0:
            count = 0
            # tries to put the cell on the grid
            while count == 0:
                location = RandomPointOnSphere() * radius * 2.0 + self.location
                if sim.boundary.contains_point(location[0:2]):
                    count = 1
        else:
            location = RandomPointOnSphere() * radius * 2.0 + self.location

        # halve the division timer
        self.division_timer *= 0.5

        # ID the cell
        ID = sim.get_ID()

        # create new cell and add it to the simulation

        sc = StemCell(ID, location, self.motion, self.mass, self.nuclear_radius, self.cytoplasm_radius, self.booleans,
                      self.state, self.diff_timer, self.division_timer, self.death_timer)

        sim.add_object_to_addition_queue(sc)


    def differentiate(self):
        """ differentiates the cell and updates the boolean values
            and sets the motion to be true
        """
        self.state = "Differentiated"
        self.booleans[2] = 1
        self.booleans[3] = 0
        self.motion = True


    def update_StemCell(self, sim):
        """ Updates the stem cell to decide whether they differentiate
            or divide, changes state, and sets motion
        """


        # if other cells are differentiated around a cell it will stop moving
        if self.state == "Differentiated":
            nbs = np.array(list(sim.network.neighbors(self)))
            for i in range(len(nbs)):
                if nbs[i].state == "Differentiated":
                    self.motion = False
                    break

        # if other cells are pluripotent, gata6 low, and nanog high they will stop moving
        if self.booleans[3] == 1 and self.booleans[2] == 0 and self.state == "Pluripotent":
            nbs = np.array(list(sim.network.neighbors(self)))
            for i in range(len(nbs)):
                if nbs[i].booleans[3] == 1 and nbs[i].booleans[2] == 0 and nbs[i].state == "Pluripotent":
                    self.motion = False
                    break

        if not self.motion and self.compress < 2.0:
            if self.state == "Differentiated" and self.division_timer >= sim.diff_div_thresh:
                self.divide(sim)

            if self.state == "Pluripotent" and self.division_timer >= sim.pluri_div_thresh:
                self.divide(sim)

            else:
                self.division_timer += 1

        # coverts position on grid into an integer for array location
        array_location_x = int(math.floor(self.location[0]))
        array_location_y = int(math.floor(self.location[1]))

        # if a certain spot of the grid is less than the max FGF4 it can hold and the cell is NANOG high increase the
        # FGF4 by 1
        if sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] < sim.max_fgf4 and \
                self.booleans[3] == 1:
            sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] += 1

        # if the FGF4 amount for the location is greater than 0, set the fgf4_bool value to be 1 for the functions
        if sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] > 0:
            fgf4_bool = 1

        else:
            fgf4_bool = 0


        # temporarily hold the FGFR value
        tempFGFR = self.booleans[0]

        # run the boolean value through the functions
        fgf4 = self.boolean_function(sim, fgf4_bool)

        # if the temporary FGFR value is 0 and the FGF4 value is 1 decrease the amount of FGF4 by 1
        # this simulates FGFR using FGF4

        if tempFGFR == 0 and fgf4 == 1 and \
                sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] >= 1:
            sim.grid[np.array([0]), np.array([array_location_x]), np.array([array_location_y])] -= 1

        # if the cell is GATA6 high and Pluripotent increase the differentiation counter by 1
        if self.booleans[2] == 1 and self.state == "Pluripotent":
            self.diff_timer += 1
            # if the differentiation counter is greater than the threshold, differentiate
            if self.diff_timer >= sim.pluri_to_diff:
                self.differentiate()


def RandomPointOnSphere():
    """ Computes a random point on a sphere
        Returns - a point on a unit sphere [x,y] at the origin
    """
    theta = r.random() * 2 * math.pi
    x = math.cos(theta)
    y = math.sin(theta)

    return np.array((x, y))

def Mag(v1):
    """ Computes the magnitude of a vector
        Returns - a float representing the vector magnitude
    """
    temp = v1[0] ** 2 + v1[1] ** 2
    return math.sqrt(temp)

