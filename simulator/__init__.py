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

__all__ = [
    'get_backend',
    'GPUBackend',
    'CUPY_AVAILABLE',
    'calculate_gravitational_force',
    'calculate_accelerations',
    'calculate_total_energy',
    'calculate_virial_ratio'
]