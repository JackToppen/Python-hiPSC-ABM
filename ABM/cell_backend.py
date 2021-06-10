import numpy as np
import math
from numba import jit, cuda, prange

from pythonabm import cuda_magnitude


@cuda.jit
def jkr_forces_gpu(jkr_edges, delete_edges, locations, radii, jkr_forces, poisson, youngs, adhesion_const):
    """ This just-in-time compiled CUDA kernel performs the actual
        calculations for the jkr_forces() method.
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

        # get the magnitude of the distance between the cells and the overlap of the cells
        mag = cuda_magnitude(location_1, location_2)
        overlap = (radii[cell_1] + radii[cell_2] - mag) / 1e6    # convert radii from um to m

        # get two values used for JKR calculation
        e_hat = (((1 - poisson[0] ** 2) / youngs[0]) + ((1 - poisson[0] ** 2) / youngs[0])) ** -1
        r_hat = (1e6 * ((1 / radii[cell_1]) + (1 / radii[cell_2]))) ** -1    # convert radii from um to m

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
                vector_part = location_1[i] - location_2[i]

                # if the magnitude is 0 use the zero vector, otherwise find the normalized vector for each axis
                if mag != 0:
                    normal = vector_part / mag
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
    """ This just-in-time compiled method performs the actual
        calculations for the jkr_forces() method.
    """
    # go through the edges array
    for edge_index in prange(number_edges):
        # get the cell indices of the edge
        cell_1 = jkr_edges[edge_index][0]
        cell_2 = jkr_edges[edge_index][1]

        # get the vector between the centers of the cells, the magnitude of this vector and the overlap of the cells
        vector = locations[cell_1] - locations[cell_2]
        mag = np.linalg.norm(vector)
        overlap = (radii[cell_1] + radii[cell_2] - mag) / 1e6    # convert radii from um to m

        # get two values used for JKR calculation
        e_hat = (((1 - poisson ** 2) / youngs) + ((1 - poisson ** 2) / youngs)) ** -1
        r_hat = (1e6 * ((1 / radii[cell_1]) + (1 / radii[cell_2]))) ** -1    # convert radii from um to m

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
def apply_forces_gpu(jkr_force, motility_force, locations, radii, stokes, size, move_dt):
    """ This just-in-time compiled CUDA kernel performs the actual
        calculations for the apply_forces() method.
    """
    # get the index in the array
    index = cuda.grid(1)

    # double check that the index is within bounds
    if index < locations.shape[0]:
        # stokes law for velocity based on force and fluid viscosity, convert radii to m from um
        stokes_friction = 6 * math.pi * stokes[0] * (radii[index] / 1e6)

        # loop over all directions of space
        for i in range(3):
            # update the velocity of the cell based on stokes
            velocity = (jkr_force[index][i] + motility_force[index][i]) / stokes_friction

            # set the new location, convert velocity from m/s to um/s
            new_location = locations[index][i] + move_dt[0] * (velocity * 1e6)

            # check if new location is in the space
            if new_location > size[i]:
                locations[index][i] = size[i]
            elif new_location < 0:
                locations[index][i] = 0
            else:
                locations[index][i] = new_location


@jit(nopython=True, parallel=True, cache=True)
def apply_forces_cpu(number_agents, jkr_force, motility_force, locations, radii, stokes, size, move_dt):
    """ This just-in-time compiled method performs the actual
        calculations for the apply_forces() method.
    """
    for index in prange(number_agents):
        # stokes law for velocity based on force and fluid viscosity, convert radii to m from um
        stokes_friction = 6 * math.pi * stokes * (radii[index] / 1e6)

        # update the velocity of the cell based on stokes
        velocity = (motility_force[index] + jkr_force[index]) / stokes_friction

        # set the new location, convert velocity from m/s to um/s
        new_location = locations[index] + move_dt * (velocity * 1e6)

        # loop over all directions of space and check if new location is in the space
        for i in range(0, 3):
            if new_location[i] > size[i]:
                locations[index][i] = size[i]
            elif new_location[i] < 0:
                locations[index][i] = 0
            else:
                locations[index][i] = new_location[i]

    return locations


@jit(nopython=True, cache=True)
def update_diffusion_jit(base, steps, diffuse_dt, last_dt, diffuse_const, spat_res2):
    """ This just-in-time compiled method performs the actual
        calculations for the update_diffusion() method.
    """
    # holder the following constants for faster computations
    a = diffuse_dt * diffuse_const / spat_res2
    b = 1 - 4 * a

    # finite difference method to solve laplacian diffusion equation, currently 2D
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

        # get the diffusion loss for the diffusion points and add morphogen change from the temporary array
        base[1:-1, 1:-1] *= b
        base[1:-1, 1:-1] += temp

    return base
