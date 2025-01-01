__doc__ = """ Rod classes and implementation details """


import numpy as np
import functools
import numba
from elastica.rod import RodBase
from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from elastica._rotations import _inv_rotate
from elastica.rod.factory_function import allocate
from elastica.rod.knot_theory import KnotTheory
from elastica._calculus import (
    quadrature_kernel_for_block_structure,
    difference_kernel_for_block_structure,
    _difference,
    _average,
)
from elastica.reset_functions_for_block_structure._reset_ghost_vector_or_scalar import (
    _reset_vector_ghost,
)
from typing import Optional

position_difference_kernel = _difference
position_average = _average

import jax
import jax.numpy as jnp


@functools.lru_cache(maxsize=1)
def _get_z_vector():
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


def _compute_sigma_kappa_for_blockstructure(memory_block):
    """
    This function is a wrapper to call functions which computes shear stretch, strain and bending twist and strain.

    Parameters
    ----------
    memory_block : object

    Returns
    -------

    """
    _compute_shear_stretch_strains(
        memory_block.position_collection,
        memory_block.volume,
        memory_block.lengths,
        memory_block.tangents,
        memory_block.radius,
        memory_block.rest_lengths,
        memory_block.rest_voronoi_lengths,
        memory_block.dilatation,
        memory_block.voronoi_dilatation,
        memory_block.director_collection,
        memory_block.sigma,
    )

    # Compute bending twist strains for the block
    _compute_bending_twist_strains(
        memory_block.director_collection,
        memory_block.rest_voronoi_lengths,
        memory_block.kappa,
    )


class CosseratRod(RodBase, KnotTheory):
    """
    Cosserat Rod class. This is the preferred class for rods because it is derived from some
    of the essential base classes.

        Attributes
        ----------
        n_elems: int
            The number of elements of the rod.
        position_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node position vectors.
        velocity_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node velocity vectors.
        acceleration_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node acceleration vectors.
        omega_collection: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Array containing element angular velocity vectors.
        alpha_collection: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Array contining element angular acceleration vectors.
        director_collection: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Array containing element director matrices.
        rest_lengths: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths at rest configuration.
        density: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod elements densities.
        volume: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element volumes.
        mass: numpy.ndarray
            1D (n_nodes) array containing data with 'float' type.
            Rod node masses. Note that masses are stored on the nodes, not on elements.
        mass_second_moment_of_inertia: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element mass second moment of interia.
        inv_mass_second_moment_of_inertia: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element inverse mass moment of inertia.
        rest_voronoi_lengths: numpy.ndarray
            1D (n_voronoi) array containing data with 'float' type.
            Rod lengths on the voronoi domain at the rest configuration.
        internal_forces: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Rod node internal forces. Note that internal forces are stored on the node, not on elements.
        internal_torques: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element internal torques.
        external_forces: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            External forces acting on rod nodes.
        external_torques: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            External torques acting on rod elements.
        lengths: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths.
        tangents: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element tangent vectors.
        radius: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element radius.
        dilatation: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation.
        voronoi_dilatation: numpy.ndarray
            1D (n_voronoi) array containing data with 'float' type.
            Rod dilatation on voronoi domain.
        dilatation_rate: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation rates.
    """

    def __init__(
        self,
        n_elements,
        position,
        velocity,
        omega,
        acceleration,
        angular_acceleration,
        directors,
        radius,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        density,
        volume,
        mass,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        rest_lengths,
        tangents,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        rest_voronoi_lengths,
        sigma,
        kappa,
        rest_sigma,
        rest_kappa,
        internal_stress,
        internal_couple,
        ring_rod_flag,
    ):
        self.n_elems = n_elements
        self.position_collection = position
        self.velocity_collection = velocity
        self.omega_collection = omega
        self.acceleration_collection = acceleration
        self.alpha_collection = angular_acceleration
        self.director_collection = directors
        self.radius = radius
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.shear_matrix = shear_matrix
        self.bend_matrix = bend_matrix
        self.density = density
        self.volume = volume
        self.mass = mass
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques
        self.external_forces = external_forces
        self.external_torques = external_torques
        self.lengths = lengths
        self.rest_lengths = rest_lengths
        self.tangents = tangents
        self.dilatation = dilatation
        self.dilatation_rate = dilatation_rate
        self.voronoi_dilatation = voronoi_dilatation
        self.rest_voronoi_lengths = rest_voronoi_lengths
        self.sigma = sigma
        self.kappa = kappa
        self.rest_sigma = rest_sigma
        self.rest_kappa = rest_kappa
        self.internal_stress = internal_stress
        self.internal_couple = internal_couple
        self.ring_rod_flag = ring_rod_flag


        if not self.ring_rod_flag:
            # For ring rod there are no periodic elements so below code won't run.
            # We add periodic elements at the memory block construction.
            # Compute shear stretch and strains.

            self.lengths[:], self.tangents[:], self.radius[:] = _compute_geometry_from_state(
                self.position_collection,
                self.volume,
                self.lengths,
                self.tangents,
                self.radius
            )

            self.dilatation[:], self.voronoi_dilatation[:] = _compute_all_dilatations(
                self.position_collection,
                self.volume,
                self.lengths,
                self.tangents,
                self.radius,
                self.dilatation,
                self.rest_lengths,
                self.rest_voronoi_lengths,
                self.voronoi_dilatation
            )

            self.sigma[:] = _compute_shear_stretch_strains(
                self.position_collection,
                self.volume,
                self.lengths,
                self.tangents,
                self.radius,
                self.rest_lengths,
                self.rest_voronoi_lengths,
                self.dilatation,
                self.voronoi_dilatation,
                self.director_collection,
                self.sigma,
            )


            # Compute bending twist strains
            _compute_bending_twist_strains(
                self.director_collection, self.rest_voronoi_lengths, self.kappa
            )

    @classmethod
    def straight_rod(
        cls,
        n_elements: int,
        start: np.ndarray,
        direction: np.ndarray,
        normal: np.ndarray,
        base_length: float,
        base_radius: float,
        density: float,
        *,
        nu: Optional[float] = None,
        youngs_modulus: float,
        **kwargs,
    ):
        """
        Cosserat rod constructor for straight-rod geometry.


        Notes
        -----
        Since we expect the Cosserat Rod to simulate soft rod, Poisson's ratio is set to 0.5 by default.
        It is possible to give additional argument "shear_modulus" or "poisson_ratio" to specify extra modulus.


        Parameters
        ----------
        n_elements : int
            Number of element. Must be greater than 3.
            Generally recommended to start with 40-50, and adjust the resolution.
        start : NDArray[3, float]
            Starting coordinate in 3D
        direction : NDArray[3, float]
            Direction of the rod in 3D
        normal : NDArray[3, float]
            Normal vector of the rod in 3D
        base_length : float
            Total length of the rod
        base_radius : float
            Uniform radius of the rod
        density : float
            Density of the rod
        nu : float
            Damping coefficient for Rayleigh damping
        youngs_modulus : float
            Young's modulus
        **kwargs : dict, optional
            The "position" and/or "directors" can be overrided by passing "position" and "directors" argument. Remember, the shape of the "position" is (3,n_elements+1) and the shape of the "directors" is (3,3,n_elements).

        Returns
        -------
        CosseratRod

        """

        if nu is not None:
            raise ValueError(
                # Remove the option to set internal nu inside, beyond v0.4.0
                "The option to set damping coefficient (nu) for the rod during rod\n"
                "initialisation is now deprecated. Instead, for adding damping to rods,\n"
                "please derive your simulation class from the add-on Damping mixin class.\n"
                "For reference see the class elastica.dissipation.AnalyticalLinearDamper(),\n"
                "and for usage check examples/axial_stretching.py"
            )
        # Straight rod is not ring rod set flag to false
        ring_rod_flag = False
        (
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
        ) = allocate(
            n_elements,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            youngs_modulus,
            rod_origin_position=start,
            ring_rod_flag=ring_rod_flag,
            **kwargs,
        )

        return cls(
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            ring_rod_flag,
        )

    @classmethod
    def ring_rod(
        cls,
        n_elements: int,
        ring_center_position: np.ndarray,
        direction: np.ndarray,
        normal: np.ndarray,
        base_length: float,
        base_radius: float,
        density: float,
        *,
        nu: Optional[float] = None,
        youngs_modulus: float,
        **kwargs,
    ):
        """
        Cosserat rod constructor for straight-rod geometry.


        Notes
        -----
        Since we expect the Cosserat Rod to simulate soft rod, Poisson's ratio is set to 0.5 by default.
        It is possible to give additional argument "shear_modulus" or "poisson_ratio" to specify extra modulus.


        Parameters
        ----------
        n_elements : int
            Number of element. Must be greater than 3. Generarally recommended to start with 40-50, and adjust the resolution.
        ring_center_position : NDArray[3, float]
            Center coordinate for ring rod in 3D
        direction : NDArray[3, float]
            Direction of the rod in 3D
        normal : NDArray[3, float]
            Normal vector of the rod in 3D
        base_length : float
            Total length of the rod
        base_radius : float
            Uniform radius of the rod
        density : float
            Density of the rod
        nu : float
            Damping coefficient for Rayleigh damping
        youngs_modulus : float
            Young's modulus
        **kwargs : dict, optional
            The "position" and/or "directors" can be overrided by passing "position" and "directors" argument. Remember, the shape of the "position" is (3,n_elements+1) and the shape of the "directors" is (3,3,n_elements).

        Returns
        -------
        CosseratRod

        """

        if nu is not None:
            raise ValueError(
                # Remove the option to set internal nu inside, beyond v0.4.0
                "The option to set damping coefficient (nu) for the rod during rod\n"
                "initialisation is now deprecated. Instead, for adding damping to rods,\n"
                "please derive your simulation class from the add-on Damping mixin class.\n"
                "For reference see the class elastica.dissipation.AnalyticalLinearDamper(),\n"
                "and for usage check examples/axial_stretching.py"
            )
        # Straight rod is not ring rod set flag to false
        ring_rod_flag = True
        (
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
        ) = allocate(
            n_elements,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            youngs_modulus,
            rod_origin_position=ring_center_position,
            ring_rod_flag=ring_rod_flag,
            **kwargs,
        )

        return cls(
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            ring_rod_flag,
        )

    def compute_internal_forces_and_torques(self, time):
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.

        Parameters
        ----------
        time: float
            current time

        """
        self.lengths[:], self.tangents[:], self.radius[:] = _compute_geometry_from_state(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius
        )

        self.dilatation[:], self.voronoi_dilatation[:] = _compute_all_dilatations(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.dilatation,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.voronoi_dilatation
        )

        self.sigma[:] = _compute_shear_stretch_strains(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
        )

        self.internal_stress[:] = _compute_internal_shear_stretch_stresses_from_model(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
            self.rest_sigma,
            self.shear_matrix,
            self.internal_stress,
        )


        self.internal_forces[:] = _compute_internal_forces(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
            self.rest_sigma,
            self.shear_matrix,
            self.internal_stress,
            self.internal_forces,
            self.ghost_elems_idx,
        )

        self.kappa[:] = _compute_bending_twist_strains(
            self.director_collection,
            self.rest_voronoi_lengths,
            self.kappa
        )

        self.internal_couple[:] = _compute_internal_bending_twist_stresses_from_model(
            self.director_collection,
            self.rest_voronoi_lengths,
            self.internal_couple,
            self.bend_matrix,
            self.kappa,
            self.rest_kappa,
        )

        self.internal_torques[:] = _compute_internal_torques(
            self.position_collection,
            self.velocity_collection,
            self.tangents,
            self.lengths,
            self.rest_lengths,
            self.director_collection,
            self.rest_voronoi_lengths,
            self.bend_matrix,
            self.rest_kappa,
            self.kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            self.internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.internal_torques,
            self.ghost_voronoi_idx,
        )

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time):
        """
        Updates the acceleration variables

        Parameters
        ----------
        time: float
            current time

        """
        self.acceleration_collection[:], self.alpha_collection[:] = _update_accelerations(
            self.acceleration_collection,
            self.internal_forces,
            self.external_forces,
            self.mass,
            self.alpha_collection,
            self.inv_mass_second_moment_of_inertia,
            self.internal_torques,
            self.external_torques,
            self.dilatation,
        )


    def zeroed_out_external_forces_and_torques(self, time):
        self.external_forces, self.external_torques = _zeroed_out_external_forces_and_torques(
            self.external_forces, self.external_torques
        )

    def compute_translational_energy(self):
        """
        Compute total translational energy of the rod at the instance.
        """
        return (
            0.5
            * (
                self.mass
                * np.einsum(
                    "ij, ij-> j", self.velocity_collection, self.velocity_collection
                )
            ).sum()
        )

    def compute_rotational_energy(self):
        """
        Compute total rotational energy of the rod at the instance.
        """
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(self):
        """
        Compute velocity center of mass of the rod at the instance.
        """
        mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
        sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(self):
        """
        Compute position center of mass of the rod at the instance.
        """
        mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
        sum_mass_times_position = np.einsum("ij->i", mass_times_position)

        return sum_mass_times_position / self.mass.sum()

    def compute_bending_energy(self):
        """
        Compute total bending energy of the rod at the instance.
        """

        kappa_diff = self.kappa - self.rest_kappa
        bending_internal_torques = _batch_matvec(self.bend_matrix, kappa_diff)

        return (
            0.5
            * (
                _batch_dot(kappa_diff, bending_internal_torques)
                * self.rest_voronoi_lengths
            ).sum()
        )

    def compute_shear_energy(self):
        """
        Compute total shear energy of the rod at the instance.
        """

        sigma_diff = self.sigma - self.rest_sigma
        shear_internal_forces = _batch_matvec(self.shear_matrix, sigma_diff)

        return (
            0.5
            * (_batch_dot(sigma_diff, shear_internal_forces) * self.rest_lengths).sum()
        )


# Below is the numba-implementation of Cosserat Rod equations. They don't need to be visible by users.

@jax.jit
def _compute_geometry_from_state(
    position_collection, volume, lengths, tangents, radius
):
    position_diff = position_difference_kernel(position_collection)
    lengths = _batch_norm(position_diff) + 1e-14

    lim = lengths.shape[0]
    tangents = tangents.at[:lim].set(position_diff[:, :lim] / lengths[:lim])
    radius = radius.at[:lim].set(jnp.sqrt(volume[:lim] / lengths[:lim] / jnp.pi))

    return lengths, tangents, radius

@jax.jit
def _compute_all_dilatations(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    dilatation,
    rest_lengths,
    rest_voronoi_lengths,
    voronoi_dilatation,
):

    leng_lim = lengths.shape[0]
    dilatation = dilatation.at[:leng_lim].set(lengths[:leng_lim] / rest_lengths[:leng_lim])

    voronoi_lengths = position_average(lengths)
    voronoi_lim = voronoi_lengths.shape[0]
    voronoi_dilatation = voronoi_dilatation.at[:voronoi_lim].set(voronoi_lengths[:voronoi_lim] / rest_voronoi_lengths[:voronoi_lim])

    return dilatation, voronoi_dilatation


@jax.jit
def _compute_dilatation_rate(
    position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
):

    r_dot_v = _batch_dot(position_collection, velocity_collection)
    r_plus_one_dot_v = _batch_dot(
        position_collection[..., 1:], velocity_collection[..., :-1]
    )
    r_dot_v_plus_one = _batch_dot(
        position_collection[..., :-1], velocity_collection[..., 1:]
    )

    blocksize = lengths.shape[0]

    dilatation_rate = dilatation_rate.at[:blocksize].set(
        (r_dot_v[:blocksize] + r_dot_v[1:blocksize + 1] - r_dot_v_plus_one[:blocksize] - r_plus_one_dot_v[:blocksize])
        / lengths[:blocksize]
        / rest_lengths[:blocksize]
    )

@jax.jit
def _compute_shear_stretch_strains(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    voronoi_dilatation,
    director_collection,
    sigma,
):

    z_vector = np.array([0.0, 0.0, 1.0]).reshape(3, -1)
    sigma = dilatation * _batch_matvec(director_collection, tangents) - z_vector

    return sigma


@jax.jit
def _compute_internal_shear_stretch_stresses_from_model(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    voronoi_dilatation,
    director_collection,
    sigma,
    rest_sigma,
    shear_matrix,
    internal_stress,
):
    internal_stress = _batch_matvec(shear_matrix, sigma - rest_sigma)

    return internal_stress

@jax.jit
def _compute_bending_twist_strains(director_collection, rest_voronoi_lengths, kappa):
    """
    Update <curvature/twist (kappa)> given <director and rest_voronoi_length>.
    """
    temp = _inv_rotate(director_collection)
    blocksize = rest_voronoi_lengths.shape[0]

    kappa = kappa.at[:, :blocksize].set(temp[:, :blocksize] / rest_voronoi_lengths[blocksize])

    return kappa

@jax.jit
def _compute_internal_bending_twist_stresses_from_model(
    director_collection,
    rest_voronoi_lengths,
    internal_couple,
    bend_matrix,
    kappa,
    rest_kappa,
):

    blocksize = kappa.shape[1]
    temp = kappa[:3, :blocksize] - rest_kappa[:3, :blocksize]
    internal_couple = _batch_matvec(bend_matrix, temp)

    return internal_couple


@jax.jit
def _compute_internal_forces(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    voronoi_dilatation,
    director_collection,
    sigma,
    rest_sigma,
    shear_matrix,
    internal_stress,
    internal_forces,
    ghost_elems_idx,
):

    blocksize = internal_stress.shape[1]
    cosserat_internal_stress = jnp.zeros((3, blocksize))

    cosserat_internal_stress = cosserat_internal_stress.at[:3, :blocksize].set(
        jnp.einsum(
        'jik,jk->ik',
        director_collection[:3, :3, :blocksize],
        internal_stress[:3, :blocksize]
        )
    )

    cosserat_internal_stress = cosserat_internal_stress / dilatation
    internal_forces = difference_kernel_for_block_structure(
        cosserat_internal_stress, ghost_elems_idx
    )

    return internal_forces

@jax.jit
def _compute_internal_torques(
    position_collection,
    velocity_collection,
    tangents,
    lengths,
    rest_lengths,
    director_collection,
    rest_voronoi_lengths,
    bend_matrix,
    rest_kappa,
    kappa,
    voronoi_dilatation,
    mass_second_moment_of_inertia,
    omega_collection,
    internal_stress,
    internal_couple,
    dilatation,
    dilatation_rate,
    internal_torques,
    ghost_voronoi_idx,
):

    voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation ** 3

    bend_twist_couple_2D = difference_kernel_for_block_structure(
        internal_couple * voronoi_dilatation_inv_cube_cached, ghost_voronoi_idx
    )

    bend_twist_couple_3D = quadrature_kernel_for_block_structure(
        _batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths
        * voronoi_dilatation_inv_cube_cached,
        ghost_voronoi_idx,
    )

    shear_stretch_couple = (
        _batch_cross(_batch_matvec(director_collection, tangents), internal_stress)
        * rest_lengths
    )

    J_omega_upon_e = (
        _batch_matvec(mass_second_moment_of_inertia, omega_collection) / dilatation
    )

    lagrangian_transport = _batch_cross(J_omega_upon_e, omega_collection)
    unsteady_dilatation = J_omega_upon_e * dilatation_rate / dilatation

    blocksize = internal_torques.shape[1]

    internal_torques = internal_torques.at[:3 :blocksize].set(
        bend_twist_couple_2D[:3 :blocksize]
        + bend_twist_couple_3D[:3 :blocksize]
        + shear_stretch_couple[:3 :blocksize]
        + lagrangian_transport[:3 :blocksize]
        + unsteady_dilatation[:3 :blocksize]
    )

    return internal_torques


@jax.jit
def _update_accelerations(
    acceleration_collection,
    internal_forces,
    external_forces,
    mass,
    alpha_collection,
    inv_mass_second_moment_of_inertia,
    internal_torques,
    external_torques,
    dilatation,
):

    blocksize_acc = internal_forces.shape[1]
    blocksize_alpha = internal_torques.shape[1]

    acceleration_collection = acceleration_collection.at[:3, :blocksize_acc].set(
        (internal_forces[:3, :blocksize_acc] + external_forces[:3, :blocksize_acc]) / mass[:blocksize_acc]
    )

    alpha_collection = alpha_collection.at[:3, :blocksize_alpha].set(
        (
            jnp.einsum(
                "ijk,jk->ik",
                inv_mass_second_moment_of_inertia[:3, :3, :blocksize_alpha],
                internal_torques[:3, :blocksize_alpha] + external_torques[:3, :blocksize_alpha],
            ) * dilatation[:blocksize_alpha]
        )
    )

    return acceleration_collection, alpha_collection



@jax.jit
def _zeroed_out_external_forces_and_torques(external_forces, external_torques):
    """
    This function is to zeroed out external forces and torques.

    Notes
    -----
    Microbenchmark results 100 elements
    python version: 3.32 µs ± 44.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    this version: 583 ns ± 1.94 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    """
    n_nodes = external_forces.shape[1]
    n_elems = external_torques.shape[1]

    external_forces = external_forces.at[:3, :n_nodes].set(0.0)
    external_torques = external_torques.at[:3, :n_elems].set(0.0)

    return external_forces, external_torques
