import numpy as np
import igraph


class Simulation:
    """ This object holds all of the important information about the simulation as it
        runs. The template files are read to get parameters that often change.
    """
    def __init__(self, templates_path, name, path, separator, mode):

        # read the template files and create instance variables based on their values

        # ------------- general template file -------------------------
        # open the .txt file and get a list of the lines
        with open(templates_path + "general.txt") as general_file:
            general = general_file.readlines()

        # create instance variables based on template parameters
        self.parallel = eval(general[4][2:-3])
        self.end_step = int(general[7][2:-3])
        self.num_nanog = int(general[10][2:-3])
        self.num_gata6 = int(general[13][2:-3])
        self.size = np.array(eval(general[16][2:-3]))

        # ------------- outputs template file -------------------------
        # open the .txt file and get a list of the lines
        with open(templates_path + "outputs.txt") as outputs_file:
            outputs = outputs_file.readlines()

        # create instance variables based on template parameters
        self.output_values = eval(outputs[4][2:-3])
        self.output_tda = eval(outputs[8][2:-3])
        self.output_gradients = eval(outputs[12][2:-3])
        self.output_images = eval(outputs[15][2:-3])
        self.image_quality = int(outputs[19][2:-3])
        self.fps = float(outputs[22][2:-3])
        self.color_mode = eval(outputs[26][2:-3])
        self.output_fgf4_image = eval(outputs[29][2:-3])

        # ------------- experimental template file -------------------------
        # open the .txt file and get a list of the lines
        with open(templates_path + "experimental.txt") as experimental_file:
            experimental = experimental_file.readlines()

        # create instance variables based on template parameters
        self.pluri_div_thresh = int(experimental[4][2:-3])
        self.diff_div_thresh = int(experimental[7][2:-3])
        self.pluri_to_diff = int(experimental[10][2:-3])
        self.death_thresh = int(experimental[13][2:-3])
        self.fds_thresh = int(experimental[16][2:-3])
        self.fgf4_thresh = int(experimental[19][2:-3])
        self.lonely_cell = int(experimental[22][2:-3])
        self.group = int(experimental[25][2:-3])
        self.guye_move = eval(experimental[28][2:-3])
        self.eunbi_move = eval(experimental[31][2:-3])
        self.max_concentration = float(experimental[34][2:-3])
        self.fgf4_move = eval(experimental[37][2:-3])
        self.output_tda = eval(experimental[40][2:-3])
        self.dox_value = float(experimental[43][2:-3])
        self.field = int(experimental[46][2:-3])

        # define any other instance variables that are not part of the template files

        # the temporal resolution for the simulation
        self.step_dt = 1800  # dt of each simulation step (1800 sec)
        self.move_dt = 200  # dt for incremental movement (200 sec)
        self.diffuse_dt = 0.5  # dt for stable diffusion model (0.5 sec)

        # min and max radius lengths are used to calculate linear growth of the radius over time in 2D
        self.max_radius = 0.000005  # 5 um
        self.min_radius = self.max_radius / 2 ** 0.5
        self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
        self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

        # the neighbor graph holds all nearby cells within a fixed radius, where the JKR graph is used for
        # storing adhesive bonds between cells
        self.neighbor_graph = igraph.Graph()
        self.jkr_graph = igraph.Graph()

        # add the names of the graphs below for automatic cell addition and removal
        self.graph_names = ["neighbor_graph", "jkr_graph"]

        # the diffusion constant for the molecule gradients and the radius of search for diffusion points
        # the spatial resolution of the space
        self.spat_res = 0.00001  # 10 um
        self.spat_res2 = self.spat_res ** 2
        self.diffuse = 0.00000000005    # 50 um^2/s
        self.diffuse_radius = self.spat_res * 0.707106781187

        # calculate the size of the array for the diffusion points and create gradient arrays
        self.gradient_size = np.ceil(self.size / self.spat_res).astype(int) + np.ones(3, dtype=int)
        self.fgf4_values = np.zeros(self.gradient_size)
        self.fgf4_alt = np.zeros(self.gradient_size)

        # add the names of the gradients below for automatic diffusion updating
        self.gradient_names = ["fgf4_values", "fgf4_alt"]

        ########################################################################################################
        ########################################################################################################
        ########################################################################################################
        ########################################################################################################
        ########################################################################################################
        ########################################################################################################

        # these instance variables are rarely changed and serve to keep the model running

        # hold the name of the simulation, the path to the simulation directory, and the path separator
        self.name = name
        self.path = path
        self.separator = separator
        self.mode = mode

        # create the following paths used for outputting files to simulation directory
        self.images_path = self.path + self.name + "_images" + self.separator
        self.values_path = self.path + self.name + "_values" + self.separator
        self.gradients_path = self.path + self.name + "_gradients" + self.separator
        self.tda_path = self.path + self.name + "_tda" + self.separator

        # hold the number of cells and the step to begin at (can be altered by various modes)
        self.number_cells = 0
        self.beginning_step = 1

        # arrays to store the cells set to divide or die
        self.cells_to_divide = np.array([], dtype=int)
        self.cells_to_remove = np.array([], dtype=int)

        # various other holders
        self.cell_array_names = list()    # stores the names of the cell arrays
        self.cell_types = dict()          # holds the names of the cell types defined in run.py
        self.function_times = dict()      # store the runtimes of the various methods as the model runs

    def cell_type(self, name, number):
        """ creates a new cell type for setting initial parameters
            and defines a slice in the cell arrays that corresponds
            to this cell type
        """
        # determine the bounds of the slice and update the number of cells
        begin = self.number_cells
        end = self.number_cells = begin + number

        # add the cells to the graphs
        for graph in self.graph_names:
            self.__dict__[graph].add_vertices(number)

        # add slice to general dictionary for
        self.cell_types[name] = (begin, end)

    def cell_arrays(self, *args):
        """ creates Simulation instance variables for the cell arrays
            used to represent cell values
        """
        # go through all array parameters provided
        for array_params in args:
            # add the array name to a list for automatic addition/removal when cells divide and die
            self.cell_array_names.append(array_params[0])

            # get the tuple length for the particular array parameters
            length = len(array_params)

            # get the initial size of 1D array if the length is 2, but make a 2D array if a third index is provided
            if length == 2:
                size = self.number_cells

            elif length == 3:
                size = (self.number_cells, array_params[2])

            else:
                raise Exception("Tuples for defining cell array parameters should have length 2 or 3.")

            # create an instance variable of the Simulation class for the cell array with these parameters
            self.__dict__[array_params[0]] = np.empty(size, dtype=array_params[1])

    def initials(self, array_name, func, cell_type=None):
        """ given a lambda function for the initial values
            of a cell array this updates that accordingly
        """
        # if no cell type name provided
        if cell_type is None:
            # go through all cells and give this initial parameter
            for i in range(self.number_cells):
                self.__dict__[array_name][i] = func()

        # otherwise update the slice of the array based on the
        else:
            # get the beginning and end of the slice
            begin = self.cell_types[cell_type][0]
            end = self.cell_types[cell_type][1]

            # update only this slice of the cell array
            for i in range(begin, end):
                self.__dict__[array_name][i] = func()
