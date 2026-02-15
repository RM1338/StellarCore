"""
Simulator Module

N-body simulation engine for globular clusters.
"""

from .gpu_backend import get_backend, GPUBackend, CUPY_AVAILABLE
from .physics import (
    calculate_gravitational_force,
    calculate_accelerations,
    calculate_total_energy,
    calculate_virial_ratio
)
from .initial_conditions import (
    generate_king_model,
    generate_plummer_sphere,
    generate_kroupa_masses,
    normalize_to_nbody_units
)
from .integrators import (
    hermite_step,
    leapfrog_step,
    calculate_adaptive_timestep,
    integrate_step
)
from .nbody import (
    SimulationState,
    NBodySimulation
)

__all__ = [
    'get_backend',
    'GPUBackend',
    'CUPY_AVAILABLE',
    'calculate_gravitational_force',
    'calculate_accelerations',
    'calculate_total_energy',
    'calculate_virial_ratio',
    'generate_king_model',
    'generate_plummer_sphere',
    'generate_kroupa_masses',
    'normalize_to_nbody_units',
    'hermite_step',
    'leapfrog_step',
    'calculate_adaptive_timestep',
    'integrate_step',
    'SimulationState',
    'NBodySimulation'
]