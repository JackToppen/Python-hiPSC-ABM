import numpy as np
import random as r
from pythonabm import Simulation

from cell_methods import CellMethods
from cell_outputs import CellOutputs


class CellSimulation(CellMethods, CellOutputs, Simulation):
    """ This class inherits the Simulation class and adds methods from the CellMethods
        and CellOutputs mixins.
    """
    def __init__(self, name, output_path):
        # initialize the Simulation object
        Simulation.__init__(self, name, output_path)

        # read parameters from YAML file and add them to instance variables
        self.yaml_parameters("templates\\general.yaml")
        self.yaml_parameters("templates\\experimental.yaml")

        # hold these additional paths
        self.gradients_path = self.main_path + name + "_gradients" + self.separator  # gradients output directory
        self.tda_path = self.main_path + name + "_tda" + self.separator  # topological data analysis output directory

        # the temporal resolution for the simulation
        self.step_dt = 1800  # dt of each simulation step (1800 sec)
        self.move_dt = 180  # dt for incremental movement (180 sec)
        # self.diffuse_dt = 0.24  # dt for stable diffusion model (0.24 sec)
        # self.diffuse_dt = 6.24  # dt for stable diffusion model (6 sec)

        # the field for the finite dynamical system
        self.field = 2

        # probability of randomly increasing FDS value to high
        self.GATA6_prob = 0.01
        self.NANOG_prob = 0.01

        # the rates (in steps) of division, differentiation, death, and finite dynamical system updating
        self.pluri_div_thresh = 36
        self.diff_div_thresh = 72
        self.pluri_to_diff = 36
        self.death_thresh = 144
        self.fds_thresh = 1

        # min and max radius lengths are used to calculate linear growth of the radius over time
        self.max_radius = 5    # 5 um
        self.min_radius = self.max_radius / 2 ** 0.5    # half the area for max radius cell in 2D
        self.pluri_growth = (self.max_radius - self.min_radius) / self.pluri_div_thresh
        self.diff_growth = (self.max_radius - self.min_radius) / self.diff_div_thresh

        # the spatial resolution of the space, the diffusion constant for the molecule gradients, the radius of
        # search for diffusion points, and the max concentration at a diffusion point (not currently in use)
        # self.spat_res = 0.00000707106
        # self.diffuse_const = 0.00000000005    # 50 um^2/s
        # self.spat_res = 0.00001
        # self.spat_res2 = self.spat_res ** 2
        # self.diffuse_const = 0.000000000002  # 2 um^2/s
        # self.max_concentration = 2

        # calculate the size of the array for the diffusion points and create gradient array(s) (not currently in use)
        # self.gradient_size = np.ceil(self.size / self.spat_res).astype(int) + 1
        # self.fgf4_values = np.zeros(self.gradient_size, dtype=float)
        # self.gradient_names = ["fgf4_values"]  # add names for automatic CSV output of gradients
        # self.degradation = 0.1    # this will degrade the morphogen by this much at each step

        # self.fgf4_alt = np.zeros(self.gradient_size, dtype=float)    # for testing morphogen release methods
        # self.gradient_names = ["fgf4_values", "fgf4_alt"]    # add names for automatic CSV output of gradients

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # Add the specified number of NANOG/GATA6 high cells and create cell type GATA6_high.
        self.add_agents(self.num_to_start)
        self.add_agents(self.num_gata6, agent_type="GATA6_high")

        # Create the following cell arrays with initial conditions.
        self.agent_array("locations", override=np.random.rand(self.number_agents, 3) * self.size)
        self.agent_array("radii", func=lambda: self.max_radius)
        self.agent_array("FGF4", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("FGFR", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("ERK", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("GATA6", dtype=int)
        self.agent_array("NANOG", dtype=int, func=lambda: r.randrange(0, self.field))
        self.agent_array("states", dtype=int)
        self.agent_array("death_counters", dtype=int, func=lambda: r.randrange(0, self.death_thresh))
        self.agent_array("diff_counters", dtype=int, func=lambda: r.randrange(0, self.pluri_to_diff))
        self.agent_array("div_counters", dtype=int, func=lambda: r.randrange(0, self.pluri_div_thresh))
        self.agent_array("fds_counters", dtype=int, func=lambda: r.randrange(0, self.fds_thresh))
        self.agent_array("motility_forces", vector=3)
        self.agent_array("jkr_forces", vector=3)

        # Update the number of cells marked with the "GATA6_high" cell type with alternative initial conditions.
        self.agent_array("GATA6", agent_type="GATA6_high", func=lambda: r.randrange(1, self.field))
        self.agent_array("NANOG", agent_type="GATA6_high", func=lambda: 0)

        # Create graphs for holding cell neighbors
        self.agent_graph("neighbor_graph")
        self.agent_graph("jkr_graph")

        # record initial values
        self.step_image()
        self.step_values(arrays=["locations", "FGF4", "FGFR", "ERK", "GATA6", "NANOG", "states", "diff_counters",
                                 "div_counters"])
        self.step_tda()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # records step run time and prints the current step and number of agents
        self.info()

        # Finds the neighbors of each cell that are within a fixed radius and store this info in a graph.
        self.get_neighbors("neighbor_graph", 15)

        # Updates cells by adjusting trackers for differentiation, division, growth, etc. based on intracellular,
        # intercellular, and extracellular conditions through a series of separate methods.
        self.cell_division()
        self.cell_death()
        self.cell_pathway()
        self.cell_differentiate()
        # self.cell_growth()
        # self.cell_stochastic_update()
        # self.cell_diff_surround()

        # Simulates diffusion the specified extracellular gradient via the forward time centered space method.
        # self.update_diffusion("fgf4_values")
        # self.update_diffusion("fgf4_alt")    # for testing morphogen release methods

        # Calculates the direction/magnitude of a cell's movement depending on a variety of factors such as state
        # and presence of neighbors.
        self.cell_motility()

        # Through the series of methods, attempt to move the cells to a state of physical equilibrium between
        # adhesive and repulsive forces acting on the cells, while applying active motility forces.
        self.apply_forces()

        # Saves multiple forms of information about the simulation at the current step, including an image of the
        # space, CSVs with values of the cells, a temporary pickle of the Simulation object, and performance stats.
        # See the outputs.txt template file for turning off certain outputs.
        self.step_image()
        self.step_values(arrays=["locations", "FGF4", "FGFR", "ERK", "GATA6", "NANOG", "states", "diff_counters",
                                 "div_counters"])
        # self.step_gradients()
        self.step_tda()
        self.temp()
        self.data()
