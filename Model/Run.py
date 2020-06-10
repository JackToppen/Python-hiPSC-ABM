"""

This is the Python file that you run to begin the simulation. Before you begin, make sure you have
updated "Input.setup(...)" such that it represents the string that is the path pointing to the .txt file
a.k.a. the template file used for simulation information. All lines indicated with "(base)" are
necessary to for the model to work. Removing such lines will cause the model to either run incorrectly
or not at all. Feel free to modify or delete any functions that are not labeled in this fashion. Add
any necessary functions here to customize the model to your liking. Functions underneath the for loop
are run each step, while functions before or after will be run before or after all steps are executed.

See each file for a description on how each file is used. Documentation should provide lengthy explanations,
choices, and  purposes regarding the model.

"""
import Input
import Output

# setup() will create an instance of the Simulation class that holds extracellular and cell objects.
# This is done by reading a template .txt file that contains all initial parameters of the model.   (base)
simulation = Input.setup("C:\\Python37\\Seed Project\\Model\\template.txt")

# This will loop over all steps defined in the template file in addition to updating the current step
# of the simulation.   (base)
for simulation.current_step in range(simulation.beginning_step, simulation.end_step + 1):

    # Prints the current step and number of cells and records current time. Used to give an idea of the progress.
    simulation.info()

    # Updates each of the extracellular gradients by "smoothing" the points that represent the concentrations.   (base)
    simulation.update_diffusion()

    # Refreshes the graph used to represent cells as nodes and neighbor connections as edges.   (base)
    simulation.check_neighbors()

    # if any differentiated cells exist within a cell's defined search radius, this will find the closest one.
    simulation.nearest_diff()

    # A way of introducing cell death into the model by removing cells if they are without neighbors for so long.
    simulation.cell_death()

    # Represents the phenomena that differentiated neighbors of a pluripotent cell may induce its differentiation.
    simulation.cell_diff_surround()

    # Gets motility forces depending on a variety of factors involving state and presence of neighbors
    simulation.cell_motility()

    # Updates cells by adjusting trackers for differentiation and division based on intracellular, intercellular,
    # and extracellular conditions.   (base)
    simulation.update_cells()

    # Adds/removes cells to/from the simulation either all together or in desired numbers of cells. If done in
    # groups, the handle_movement() function will be used to better represent asynchronous division and death   (base)
    simulation.update_cell_queue()

    # Refreshes the graph used to represent cells as nodes and neighbor connections as edges.   (base)
    simulation.check_neighbors()

    # Moves the cells to a state of physical equilibrium so that there is minimal overlap of cells, while also
    # applying forces from the previous motility_cells() function.   (base)
    simulation.handle_movement()

    # Creates an 2D image of the space.   (base)
    Output.step_image(simulation)

    # Creates a CSV file with each line corresponding to a cell.    (base)
    Output.step_csv(simulation)

    # Adds a line to a running CSV file that includes information about the number of cells, total memory use,
    # individual step run time, and various other information.    (base)
    Output.simulation_data(simulation)

# Looks at all images produced by the simulation and turns them into a video.
Output.image_to_video(simulation)
