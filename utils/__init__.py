"""
Utilities Module

Analysis, visualization, and security utilities.
"""

from .security import (
    get_rate_limiter,
    rate_limit,
    validate_particle_count,
    validate_time_range
)

from .coordinates import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    radec_to_xyz,
    xyz_to_radec,
    parallax_to_distance
)

from .visualization import (
    plot_particles_3d,
    plot_particles_2d,
    plot_density_profile,
    plot_energy_evolution,
    plot_velocity_dispersion,
    PLOTLY_AVAILABLE
)

from .analysis import (
    calculate_lagrange_radii,
    calculate_half_mass_radius,
    calculate_core_radius,
    calculate_system_properties,
    print_system_properties
)

__all__ = [
    'get_rate_limiter',
    'rate_limit',
    'validate_particle_count',
    'validate_time_range',
    'spherical_to_cartesian',
    'cartesian_to_spherical',
    'radec_to_xyz',
    'xyz_to_radec',
    'parallax_to_distance',
    'plot_particles_3d',
    'plot_particles_2d',
    'plot_density_profile',
    'plot_energy_evolution',
    'plot_velocity_dispersion',
    'PLOTLY_AVAILABLE',
    'calculate_lagrange_radii',
    'calculate_half_mass_radius',
    'calculate_core_radius',
    'calculate_system_properties',
    'print_system_properties'
]