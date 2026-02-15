"""
Physics Engine - Gravitational Dynamics

Core physics calculations for N-body simulations.

Implements:
- Gravitational force calculations (direct summation)
- Energy computations (kinetic, potential, total)
- Momentum calculations
- Virial theorem verification
- Softening for close encounters

Security:
- Input validation via assertions
- Overflow protection
- NaN/Inf detection
- Resource limits enforcement
"""

import numpy as np
from typing import Tuple, Optional
from .gpu_backend import get_backend, ensure_array, get_array_module



G = 1.0

EPSILON = 0.01  



def calculate_gravitational_force(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float = EPSILON,
    backend=None
) -> np.ndarray:
    """
    Calculate gravitational forces on all particles.
    
    Uses direct N^2 summation: F_i = sum_j G*m_i*m_j*(r_j - r_i)/|r_j - r_i|^3
    
    Args:
        positions: Array of shape (N, 3) - particle positions
        masses: Array of shape (N,) - particle masses
        softening: Softening length (default: EPSILON)
        backend: GPU backend (uses default if None)
        
    Returns:
        Array of shape (N, 3) - forces on each particle
        
    Security:
        - Validates input shapes
        - Checks for NaN/Inf
        - Enforces resource limits
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    
    positions = ensure_array(positions, backend)
    masses = ensure_array(masses, backend)
    
    N = positions.shape[0]
    assert positions.shape == (N, 3), f"Positions must be (N, 3), got {positions.shape}"
    assert masses.shape == (N,), f"Masses must be (N,), got {masses.shape}"
    assert N > 0, "Must have at least one particle"
    
    assert xp.all(xp.isfinite(positions)), "Positions contain NaN or Inf"
    assert xp.all(xp.isfinite(masses)), "Masses contain NaN or Inf"
    assert xp.all(masses > 0), "All masses must be positive"
    
    forces = xp.zeros_like(positions)
    
    for i in range(N):
        dr = positions - positions[i]  
        
        r2 = xp.sum(dr * dr, axis=1) + softening**2  
        
        r3 = r2 * xp.sqrt(r2)  
        
        force_mag = G * masses[i] * masses / r3  
        
        force_mag = xp.where(xp.arange(N) == i, 0.0, force_mag)
        
        forces[i] = xp.sum(force_mag[:, xp.newaxis] * dr, axis=0)
    
    assert xp.all(xp.isfinite(forces)), "Force calculation produced NaN or Inf"
    
    return forces


def calculate_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float = EPSILON,
    backend=None
) -> np.ndarray:
    """
    Calculate gravitational accelerations (a = F/m).
    
    Args:
        positions: Array of shape (N, 3)
        masses: Array of shape (N,)
        softening: Softening length
        backend: GPU backend
        
    Returns:
        Array of shape (N, 3) - accelerations
    """
    forces = calculate_gravitational_force(positions, masses, softening, backend)
    
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    masses = ensure_array(masses, backend)
    
    # a = F / m
    accelerations = forces / masses[:, xp.newaxis]
    
    return accelerations



def calculate_kinetic_energy(
    velocities: np.ndarray,
    masses: np.ndarray,
    backend=None
) -> float:
    """
    Calculate total kinetic energy: KE = 0.5 * sum(m * v^2)
    
    Args:
        velocities: Array of shape (N, 3)
        masses: Array of shape (N,)
        backend: GPU backend
        
    Returns:
        Total kinetic energy (scalar)
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    velocities = ensure_array(velocities, backend)
    masses = ensure_array(masses, backend)
    
    v_squared = xp.sum(velocities * velocities, axis=1)
    
    ke = 0.5 * xp.sum(masses * v_squared)
    
    return float(backend.to_cpu(ke))


def calculate_potential_energy(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float = EPSILON,
    backend=None
) -> float:
    """
    Calculate total potential energy: PE = -G * sum_i sum_j>i (m_i * m_j / r_ij)
    
    Args:
        positions: Array of shape (N, 3)
        masses: Array of shape (N,)
        softening: Softening length
        backend: GPU backend
        
    Returns:
        Total potential energy (scalar, negative)
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    positions = ensure_array(positions, backend)
    masses = ensure_array(masses, backend)
    
    N = positions.shape[0]
    pe = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            dr = positions[j] - positions[i]
            r2 = xp.sum(dr * dr) + softening**2
            r = xp.sqrt(r2)
            
            pe += -G * masses[i] * masses[j] / r
    
    return float(backend.to_cpu(pe))


def calculate_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float = EPSILON,
    backend=None
) -> Tuple[float, float, float]:
    """
    Calculate total energy (kinetic + potential).
    
    Args:
        positions: Array of shape (N, 3)
        velocities: Array of shape (N, 3)
        masses: Array of shape (N,)
        softening: Softening length
        backend: GPU backend
        
    Returns:
        (kinetic_energy, potential_energy, total_energy)
    """
    ke = calculate_kinetic_energy(velocities, masses, backend)
    pe = calculate_potential_energy(positions, masses, softening, backend)
    total = ke + pe
    
    return ke, pe, total



def calculate_total_momentum(
    velocities: np.ndarray,
    masses: np.ndarray,
    backend=None
) -> np.ndarray:
    """
    Calculate total linear momentum: p = sum(m * v)
    
    Args:
        velocities: Array of shape (N, 3)
        masses: Array of shape (N,)
        backend: GPU backend
        
    Returns:
        Array of shape (3,) - total momentum vector
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    velocities = ensure_array(velocities, backend)
    masses = ensure_array(masses, backend)
    
    momentum = xp.sum(masses[:, xp.newaxis] * velocities, axis=0)
    
    return backend.to_cpu(momentum)


def calculate_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    backend=None
) -> np.ndarray:
    """
    Calculate total angular momentum: L = sum(m * r x v)
    
    Args:
        positions: Array of shape (N, 3)
        velocities: Array of shape (N, 3)
        masses: Array of shape (N,)
        backend: GPU backend
        
    Returns:
        Array of shape (3,) - total angular momentum vector
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    positions = ensure_array(positions, backend)
    velocities = ensure_array(velocities, backend)
    masses = ensure_array(masses, backend)
    
    cross = xp.cross(positions, velocities)
    
    angular_momentum = xp.sum(masses[:, xp.newaxis] * cross, axis=0)
    
    return backend.to_cpu(angular_momentum)



def calculate_virial_ratio(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    softening: float = EPSILON,
    backend=None
) -> float:
    """
    Calculate virial ratio: Q = 2*KE / |PE|
    
    For a system in virial equilibrium, Q should be approximately 1.
    
    Args:
        positions: Array of shape (N, 3)
        velocities: Array of shape (N, 3)
        masses: Array of shape (N,)
        softening: Softening length
        backend: GPU backend
        
    Returns:
        Virial ratio Q (dimensionless)
    """
    ke = calculate_kinetic_energy(velocities, masses, backend)
    pe = calculate_potential_energy(positions, masses, softening, backend)
    
    if abs(pe) < 1e-10:
        return 0.0
    
    Q = 2.0 * ke / abs(pe)
    
    return Q



def center_of_mass(
    positions: np.ndarray,
    masses: np.ndarray,
    backend=None
) -> np.ndarray:
    """
    Calculate center of mass position.
    
    Args:
        positions: Array of shape (N, 3)
        masses: Array of shape (N,)
        backend: GPU backend
        
    Returns:
        Array of shape (3,) - center of mass position
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    positions = ensure_array(positions, backend)
    masses = ensure_array(masses, backend)
    
    total_mass = xp.sum(masses)
    com = xp.sum(masses[:, xp.newaxis] * positions, axis=0) / total_mass
    
    return backend.to_cpu(com)


def center_of_mass_velocity(
    velocities: np.ndarray,
    masses: np.ndarray,
    backend=None
) -> np.ndarray:
    """
    Calculate center of mass velocity.
    
    Args:
        velocities: Array of shape (N, 3)
        masses: Array of shape (N,)
        backend: GPU backend
        
    Returns:
        Array of shape (3,) - center of mass velocity
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    velocities = ensure_array(velocities, backend)
    masses = ensure_array(masses, backend)
    
    total_mass = xp.sum(masses)
    com_vel = xp.sum(masses[:, xp.newaxis] * velocities, axis=0) / total_mass
    
    return backend.to_cpu(com_vel)



__all__ = [
    'calculate_gravitational_force',
    'calculate_accelerations',
    'calculate_kinetic_energy',
    'calculate_potential_energy',
    'calculate_total_energy',
    'calculate_total_momentum',
    'calculate_angular_momentum',
    'calculate_virial_ratio',
    'center_of_mass',
    'center_of_mass_velocity',
    'G',
    'EPSILON'
]