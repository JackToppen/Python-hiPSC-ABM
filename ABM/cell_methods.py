import random as r

from cell_backend import *
from pythonabm import record_time, normal_vector


class CellMethods:
    """ The methods in this class are meant to be inherited by the CellSimulation
        class so that CellSimulation objects can call these methods.
    """
    @record_time
    def cell_death(self):
        """ Removes any cells that meet the criteria for cell death.
        """
        # create boolean array to mark cells to be removed
        agents_to_remove = np.zeros(self.number_agents, dtype=bool)

        # determine which cells are being removed
        for index in range(self.number_agents):
            # checks to see if cell is pluripotent
            if self.states[index] == 0:
                # gets the number of neighbors for a cell, increasing the death counter if not enough neighbors
                if self.neighbor_graph.num_neighbors(index) < self.lonely_thresh:
                    self.death_counters[index] += 1

                # if sufficient neighbors, reset the death counter back to zero
                else:
                    self.death_counters[index] = 0

                # mark the cell for removal if it meets the parameters
                if self.death_counters[index] >= self.death_thresh:
                    agents_to_remove[index] = 1

        # get indices of cells to remove with a Boolean mask and count how many removed
        indices = np.arange(self.number_agents)[agents_to_remove]
        num_removed = len(indices)

        # go through the cell arrays and remove the indices
        for name in self.agent_array_names:
            # if the array is 1-dimensional, otherwise 2-dimensional
            if self.__dict__[name].ndim == 1:
                self.__dict__[name] = np.delete(self.__dict__[name], indices)
            else:
                self.__dict__[name] = np.delete(self.__dict__[name], indices, axis=0)

        # remove the indices from each graph
        for graph_name in self.graph_names:
            self.__dict__[graph_name].delete_vertices(indices)

        # change total number of cells and print to terminal
        self.number_agents -= num_removed
        print("\tRemoved " + str(num_removed) + " agents")

    @record_time
    def cell_division(self):
        """ Adjusts the cell division counters and if a cell meets
            the criteria for division, add a new cell.
        """
        # create boolean array to mark dividing cells
        agents_to_divide = np.zeros(self.number_agents, dtype=bool)

        # determine which cells are dividing
        for index in range(self.number_agents):
            # stochastically increase the division counter by either 0 or 1
            self.div_counters[index] += r.randint(0, 1)

            # if the cell is pluripotent
            if self.states[index] == 0:
                # check the division counter against the threshold, mark cell if it is
                if self.div_counters[index] >= self.pluri_div_thresh:
                    agents_to_divide[index] = 1

            # otherwise the cell is differentiated
            else:
                # check the division counter against the threshold
                if self.div_counters[index] >= self.diff_div_thresh:
                    # check for division contact inhibition if ok, mark cell for division
                    if self.neighbor_graph.num_neighbors(index) < 6:
                        agents_to_divide[index] = 1

        # get indices of the dividing cells with Boolean mask and count how many added
        indices = np.arange(self.number_agents)[agents_to_divide]
        num_added = len(indices)

        # go through the cell arrays and add indices
        for name in self.agent_array_names:
            # copy the indices of the cell array data for the dividing cells
            copies = self.__dict__[name][indices]

            # add the copies to the end of the array, handle if the array is 1-dimensional or 2-dimensional
            if self.__dict__[name].ndim == 1:
                self.__dict__[name] = np.concatenate((self.__dict__[name], copies))
            else:
                self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

        # go through each of the dividing cells, updating values for the mother and daughter cell
        for i in range(num_added):
            # get the indices of mother cell and daughter cell
            mother_index = indices[i]
            daughter_index = self.number_agents + i

            # move the cells to new positions
            division_position = self.random_vector() * (self.max_radius - self.min_radius)
            self.locations[mother_index] += division_position
            self.locations[daughter_index] -= division_position

            # set division counters to zero
            self.div_counters[mother_index] = 0
            self.div_counters[daughter_index] = 0

            # go through each graph, adding one new vertex at a time
            for graph_name in self.graph_names:
                self.__dict__[graph_name].add_vertex()

        # change total number of cells and print to terminal
        self.number_agents += num_added
        print("\tAdded " + str(num_added) + " agents")

    @record_time
    def cell_diff_surround(self):
        """ Simulates differentiated cells inducing the differentiation
            of a pluripotent cell.
        """
        for index in range(self.number_agents):
            # checks to see if cell is pluripotent and GATA6 low
            if self.states[index] == 0 and self.GATA6[index] < self.NANOG[index]:
                # get the list of neighbors for the cell
                neighbors = self.neighbor_graph.neighbors(index)

                # hold the number of differentiated neighbors
                num_diff = 0

                # if the number of differentiated neighbors is 6 or greater, set the cell to GATA6 high
                for neighbor_index in neighbors:
                    if self.states[neighbor_index] == 1:
                        num_diff += 1

                    if num_diff >= 6:
                        self.GATA6[index] = self.field - 1
                        self.NANOG[index] = 0
                        break

    @record_time
    def cell_growth(self):
        """ Simulates the growth of a cell currently linear, radius-based
            growth.
        """
        for index in range(self.number_agents):
            # increase the cell radius based on the state and whether or not it has reached the max size
            if self.radii[index] < self.max_radius:
                # pluripotent otherwise differentiated
                if self.states[index] == 0:
                    radius = self.pluri_growth * self.div_counters[index] + self.min_radius
                else:
                    radius = self.diff_growth * self.div_counters[index] + self.min_radius

                # update the radius for the index
                self.radii[index] = radius

    @record_time
    def cell_stochastic_update(self):
        """ Stochastically updates the value for GATA6 and NANOG based on
            set probabilities.
        """
        for index in range(self.number_agents):
            # if falling under threshold, raise the GATA6 value to the highest
            if r.random() < self.GATA6_prob:
                if self.GATA6[index] != self.field - 1:
                    self.GATA6[index] += 1

            # # if falling under threshold, raise the NANOG value to the highest
            # if r.random() < self.NANOG_prob:
            #     if self.NANOG[index] != self.field - 1:
            #         self.NANOG[index] += 1

    @record_time
    def cell_pathway(self):
        """ Updates the discrete dynamical system variables for each cell.
        """
        for index in range(self.number_agents):
            # only activate the pathway if doxycycline has been induced yet (after 24 hours/48 steps)
            if self.current_step >= self.dox_step:
                # get the list of neighbors for the cell and add cell to this list
                neighbors = self.neighbor_graph.neighbors(index)
                neighbors.append(index)
                num_neighbors = len(neighbors)

                # go through neighbors to get perceived FGF4 morphogen
                perceived_FGF4 = 0
                for i in range(num_neighbors):
                    # include gaussian noise
                    perceived_FGF4 += (1 + r.gauss(0, 1)) * (self.FGF4[neighbors[i]] / num_neighbors)

                # floor perceived FGF4 to nearest int, make sure it's max is field - 1 and >= 0
                perceived_FGF4 = int(perceived_FGF4)
                if perceived_FGF4 > self.field - 1:
                    perceived_FGF4 = self.field - 1
                elif perceived_FGF4 < 0:
                    perceived_FGF4 = 0

                # if updating the DDS values this step
                if self.fds_counters[index] % self.fds_thresh == 0:
                    # get the current DDS values of the cell
                    x1 = perceived_FGF4
                    x2 = self.FGFR[index]
                    x3 = self.ERK[index]
                    x4 = self.GATA6[index]
                    x5 = self.NANOG[index]

                    # if the DDS is boolean (BN_9)
                    if self.field == 2:
                        self.FGF4[index] = x5
                        self.FGFR[index] = (1 + x5 + (x4 * x5)) % 2
                        self.ERK[index] = (x1 * x2) % 2
                        self.GATA6[index] = (x3 + x4 + (x3 * x4) + (x3 * x5) + (x4 * x5) + (x3 * x4 * x5)) % 2
                        self.NANOG[index] = (x5 + (x3 * x5) + (x4 * x5) + (x3 * x4 * x5)) % 2

                    # otherwise assume ternary
                    else:
                        self.FGF4[index] = x5
                        self.FGFR[index] = (x1 * x4 * ((2*x1 + 1) * (2*x4 + 1) + x1 * x4)) % 3
                        self.ERK[index] = x2 % 3
                        self.GATA6[index] = ((x4**2) * (x5 + 1) + (x5**2) * (x4 + 1) + 2*x5 + 1) % 3
                        self.NANOG[index] = (x5**2 + x5 * (x5 + 1) * (x3 * (2*x4**2 + 2*x3 + 1) +
                                             x4*(2*x3**2 + 2*x4 + 1)) + (2*x3**2 + 1) * (2*x4**2 + 1)) % 3

                # increase the discrete dynamical system counter
                self.fds_counters[index] += 1

    @record_time
    def cell_differentiate(self):
        """ Based on GATA6 and NANOG values, stochastically increase
            differentiation counter and potentially differentiate.
        """
        for index in range(self.number_agents):
            # if the cell is GATA6 high and pluripotent
            if self.GATA6[index] > self.NANOG[index] and self.states[index] == 0:
                # increase the differentiation counter by 0 or 1
                self.diff_counters[index] += r.randint(0, 1)

                # if the differentiation counter is greater than or equal to the threshold, differentiate
                if self.diff_counters[index] >= self.pluri_to_diff:
                    self.states[index] = 1    # set state to differentiated
                    self.NANOG[index] = 0    # set NANOG to low

    @record_time
    def cell_motility(self):
        """ Gives the cells a motive force depending on set rules
            for the cell types.
        """
        # motility force for each cell
        motility_force = 0.000000002

        for index in range(self.number_agents):
            # get neighbors and if not surrounded 6 or more cells, calculate motility forces
            neighbors = self.neighbor_graph.neighbors(index)
            if len(neighbors) < 6:
                # if the cell is differentiated
                if self.states[index] == 1:
                    # add all displacement vectors of nearby NANOG high cells
                    count = 0
                    vector_holder = np.array([0.0, 0.0, 0.0])
                    for i in range(len(neighbors)):
                        if self.NANOG[neighbors[i]] > self.GATA6[neighbors[i]]:
                            count += 1
                            vector_holder += self.locations[neighbors[i]] - self.locations[index]

                    # if there is at least one nanog high cell move away from it, otherwise move randomly
                    if count > 0:
                        # get the normalized vector and move in direction opposite
                        normal = normal_vector(vector_holder)
                        random = self.random_vector()
                        self.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force
                    else:
                        self.motility_forces[index] += self.random_vector() * motility_force

                # otherwise the cell is pluripotent
                else:
                    # if the cell state is GATA6 high
                    if self.GATA6[index] > self.NANOG[index]:
                        # if guye movement, move toward differentiated cells
                        if self.guye_move:
                            # add all displacement vectors of nearby differentiated cells
                            count = 0
                            vector_holder = np.array([0.0, 0.0, 0.0])
                            for i in range(len(neighbors)):
                                if self.states[index] == 1:
                                    count += 1
                                    vector_holder += self.locations[neighbors[i]] - self.locations[index]

                            # if there is at least one differentiated cell move toward it, otherwise move randomly
                            if count > 0:
                                # get the normalized vector
                                normal = normal_vector(vector_holder)
                                random = self.random_vector()
                                self.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force
                            else:
                                self.motility_forces[index] += self.random_vector() * motility_force

                        # otherwise move away from nanog high cells
                        else:
                            # add all displacement vectors of nearby NANOG high cells
                            count = 0
                            vector_holder = np.array([0.0, 0.0, 0.0])
                            for i in range(len(neighbors)):
                                if self.NANOG[neighbors[i]] > self.GATA6[neighbors[i]]:
                                    count += 1
                                    vector_holder += self.locations[neighbors[i]] - self.locations[index]

                            # if there is at least one nanog high cell move away from it otherwise move randomly
                            if count > 0:
                                # get the normalized vector and move in direction opposite
                                normal = normal_vector(vector_holder)
                                random = self.random_vector()
                                self.motility_forces[index] += (normal * -0.8 + random * 0.2) * motility_force
                            else:
                                self.motility_forces[index] += self.random_vector() * motility_force

                    # if the cell is nanog high and gata6 low
                    elif self.GATA6[index] < self.NANOG[index]:
                        # add all displacement vectors of nearby NANOG high cells
                        count = 0
                        vector_holder = np.array([0.0, 0.0, 0.0])
                        for i in range(len(neighbors)):
                            if self.NANOG[neighbors[i]] > self.GATA6[neighbors[i]]:
                                count += 1
                                vector_holder += self.locations[neighbors[i]] - self.locations[index]

                        # if there is at least one nanog high cell move toward it, otherwise move randomly
                        if count > 0:
                            # get the normalized vector and move in direction opposite
                            normal = normal_vector(vector_holder)
                            random = self.random_vector()
                            self.motility_forces[index] += (normal * 0.8 + random * 0.2) * motility_force
                        else:
                            self.motility_forces[index] += self.random_vector() * motility_force

                    # otherwise move randomly
                    else:
                        self.motility_forces[index] += self.random_vector() * motility_force

    def calculate_jkr(self):
        """ Goes through all contacting cells and quantifies any resulting
            adhesive or repulsion forces.
        """
        # contact mechanics parameter
        adhesion_const = 0.000107    # the adhesion constant in kg/s from P Pathmanathan et al.
        poisson = 0.5    # Poisson's ratio for the cells, 0.5 means incompressible
        youngs = 1000    # Young's modulus for the cells in Pa

        # get the edges as an array, count them, and create holder used to delete edges
        jkr_edges = np.array(self.jkr_graph.get_edgelist())
        number_edges = len(jkr_edges)
        delete_edges = np.zeros(number_edges, dtype=bool)

        # only continue if edges exist
        if number_edges > 0:
            # if using CUDA GPU
            if self.cuda:
                # allow the following arrays to be passed to the GPU
                delete_edges = cuda.to_device(delete_edges)
                forces = cuda.to_device(self.jkr_forces)

                # specify threads-per-block and blocks-per-grid values
                tpb = 72
                bpg = math.ceil(number_edges / tpb)

                # call the CUDA kernel, sending arrays to GPU
                jkr_forces_gpu[bpg, tpb](cuda.to_device(jkr_edges), delete_edges, cuda.to_device(self.locations),
                                         cuda.to_device(self.radii), forces, cuda.to_device(poisson),
                                         cuda.to_device(youngs), cuda.to_device(adhesion_const))

                # return the following arrays back from the GPU
                forces = forces.copy_to_host()
                delete_edges = delete_edges.copy_to_host()

            # otherwise use parallelized JIT function
            else:
                forces, delete_edges = jkr_forces_cpu(number_edges, jkr_edges, delete_edges, self.locations, self.radii,
                                                      self.jkr_forces, poisson, youngs, adhesion_const)

            # update the graph to remove any edges that have be broken and update the JKR forces array
            self.jkr_graph.delete_edges(np.arange(number_edges)[delete_edges])
            self.jkr_forces = forces

    @record_time
    def apply_forces(self):
        """ Calls multiple methods in an attempt to move the cells to an
            equilibrium between repulsive, adhesive, and motility forces.
        """
        # constant for calculating stokes friction
        stokes = 10000

        # calculate the number of steps and the last step time if it doesn't divide nicely
        steps, last_dt = divmod(self.step_dt, self.move_dt)
        total_steps = int(steps) + 1  # add extra step for the last dt, if divides nicely last_dt will equal zero

        # go through all move steps, calculating the physical interactions and applying the forces
        for step in range(total_steps):
            # update graph for pairs of contacting cells
            self.get_neighbors("jkr_graph", 2 * self.max_radius, clear=False)

            # calculate the JKR forces based on the JKR graph edges
            self.calculate_jkr()

            # if on the last step use, that dt
            if step == total_steps - 1:
                move_dt = last_dt
            else:
                move_dt = self.move_dt

            # turn size into numpy array
            size = np.asarray(self.size)

            # if using CUDA GPU
            if self.cuda:
                # allow the following arrays to be passed to the GPU
                locations = cuda.to_device(self.locations)

                # specify threads-per-block and blocks-per-grid values
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the CUDA kernel, sending arrays to GPU
                apply_forces_gpu[bpg, tpb](cuda.to_device(self.jkr_forces), cuda.to_device(self.motility_forces),
                                           locations, cuda.to_device(self.radii), cuda.to_device(stokes),
                                           cuda.to_device(size), cuda.to_device(move_dt))

                # return the following arrays back from the GPU
                new_locations = locations.copy_to_host()

            # otherwise use parallelized JIT function
            else:
                new_locations = apply_forces_cpu(self.number_agents, self.jkr_forces, self.motility_forces,
                                                 self.locations, self.radii, stokes, size, move_dt)

            # update the locations and reset the JKR forces back to zero
            self.locations = new_locations
            self.jkr_forces[:, :] = 0

        # reset motility forces back to zero
        self.motility_forces[:, :] = 0

    @record_time
    def update_diffusion(self, gradient_name):
        """ Approximates the diffusion of the morphogen for the
            extracellular gradient specified.
        """
        # calculate the number of steps and the last step time if it doesn't divide nicely
        steps, last_dt = divmod(self.step_dt, self.diffuse_dt)
        steps = int(steps) + 1  # make sure steps is an int, add extra step for the last dt if it's less

        # all gradients are held as 3D arrays for simplicity, get the gradient as a 2D array
        gradient = self.__dict__[gradient_name][:, :, 0]

        # set max and min concentration values
        gradient[gradient > self.max_concentration] = self.max_concentration
        gradient[gradient < 0] = 0

        # pad the sides of the array with zeros for holding ghost points
        base = np.pad(gradient, 1)

        # call the JIT diffusion function, remove ghost points
        base = update_diffusion_jit(base, steps, self.diffuse_dt, last_dt, self.diffuse_const, self.spat_res2)
        gradient = base[1:-1, 1:-1]

        # degrade the morphogen concentrations
        gradient *= 1 - self.degradation

        # update the simulation with the updated gradient
        self.__dict__[gradient_name][:, :, 0] = gradient

    def get_concentration(self, gradient_name, index):
        """ Get the concentration of a gradient for a cell's
            location from the nearest diffusion point.
        """
        # get the gradient array
        gradient = self.__dict__[gradient_name]

        # find the nearest diffusion point
        half_indices = np.floor(2 * self.locations[index] / self.spat_res)
        indices = np.ceil(half_indices / 2).astype(int)
        x, y, z = indices[0], indices[1], indices[2]

        # return the value of the gradient at the diffusion point
        return gradient[x][y][z]

    def adjust_morphogens(self, gradient_name, index, amount):
        """ Adjust the concentration of the gradient based on
            the amount and the location of the cell.
        """
        # get the gradient array
        gradient = self.__dict__[gradient_name]

        # divide the location for a cell by the spatial resolution then take the floor function of it
        indices = np.floor(self.locations[index] / self.spat_res).astype(int)
        x, y, z = indices[0], indices[1], indices[2]

        # get the four nearest points to the cell in 2D and make array for holding distances
        points = np.array([[x, y, 0], [x + 1, y, 0], [x, y + 1, 0], [x + 1, y + 1, 0]], dtype=int)
        if_nearby = np.zeros(4, dtype=bool)

        # go through potential nearby diffusion points
        for i in range(4):
            # get point and make sure it's in bounds
            point = points[i]
            if point[0] < self.gradient_size[0] and point[1] < self.gradient_size[1]:
                # get location of point
                point_location = point * self.spat_res

                # see if point is in diffuse radius, if so update if_nearby index to True
                if np.linalg.norm(self.locations[index] - point_location) < self.spat_res:
                    if_nearby[i] = 1

        # get the number of points within diffuse radius
        total_nearby = np.sum(if_nearby)

        # if at least one diffusion point nearby, go back through points adding morphogen
        if total_nearby > 0:
            point_amount = amount / total_nearby
            for i in range(4):
                if if_nearby[i]:
                    x, y, z = points[i][0], points[i][1], 0
                    gradient[x][y][z] += point_amount
