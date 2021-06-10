import yaml
from cell_simulation import CellSimulation

# only call start() if this file is being run directly
if __name__ == "__main__":
    # get the path to the output directory
    with open("paths.yaml", "r") as file:
        keys = yaml.safe_load(file)
    path = keys["output_dir"]

    # start the model by calling the class method of the Simulation (or child of Simulation) class
    CellSimulation.start(path)

