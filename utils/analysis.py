"""
Analysis Utilities

Scientific analysis tools for N-body simulations.

Implements:
- Lagrange radii (r_10%, r_50%, r_90%)
- Core radius
- Half-mass radius
- Concentration parameter
- Relaxation time
"""

import numpy as np
from typing import Tuple, Dict



def calculate_lagrange_radii(
    positions: np.ndarray,
    masses: np.ndarray,
    fractions: np.ndarray = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
) -> np.ndarray:
    """
    Calculate Lagrange radii - radii containing specified mass fractions.
    
    r_X = radius containing X% of total mass
    
    Args:
        positions: (N, 3) positions
        masses: (N,) masses
        fractions: Mass fractions (e.g., [0.1, 0.5, 0.9])
        
    Returns:
        Lagrange radii corresponding to each fraction
    """
    radii = np.linalg.norm(positions, axis=1)
    
    sorted_indices = np.argsort(radii)
    radii_sorted = radii[sorted_indices]
    masses_sorted = masses[sorted_indices]
    
    cumulative_mass = np.cumsum(masses_sorted)
    total_mass = cumulative_mass[-1]
    
    lagrange_radii = np.zeros(len(fractions))
    for i, frac in enumerate(fractions):
        target_mass = frac * total_mass
        idx = np.searchsorted(cumulative_mass, target_mass)
        if idx < len(radii_sorted):
            lagrange_radii[i] = radii_sorted[idx]
        else:
            lagrange_radii[i] = radii_sorted[-1]
    
    return lagrange_radii


def calculate_half_mass_radius(
    positions: np.ndarray,
    masses: np.ndarray
) -> float:
    """
    Calculate half-mass radius (r_50%).
    
    Args:
        positions: (N, 3) positions
        masses: (N,) masses
        
    Returns:
        Half-mass radius
    """
    lagrange = calculate_lagrange_radii(positions, masses, np.array([0.5]))
    return lagrange[0]


def calculate_core_radius(
    positions: np.ndarray,
    masses: np.ndarray,
    method: str = "king"
) -> float:
    """
    Calculate core radius.
    
    Multiple methods:
    - 'king': Radius where density drops to half central density
    - 'lagrange': r_10% (inner 10% of mass)
    
    Args:
        positions: (N, 3) positions
        masses: (N,) masses
        method: Calculation method
        
    Returns:
        Core radius
    """
    if method == "lagrange":
        lagrange = calculate_lagrange_radii(positions, masses, np.array([0.1]))
        return lagrange[0]
    elif method == "king":
        from .visualization import calculate_density_profile
        radii, densities = calculate_density_profile(positions, masses, bins=50)
        
        rho_0 = densities[0]
        
        half_density = 0.5 * rho_0
        idx = np.argmin(np.abs(densities - half_density))
        
        return radii[idx]
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_concentration(
    positions: np.ndarray,
    masses: np.ndarray
) -> float:
    """
    Calculate concentration parameter.
    
    c = log10(r_tidal / r_core)
    
    Approximates r_tidal as r_90% and r_core as r_10%
    
    Args:
        positions: (N, 3) positions
        masses: (N,) masses
        
    Returns:
        Concentration parameter
    """
    lagrange = calculate_lagrange_radii(
        positions, masses,
        np.array([0.1, 0.9])
    )
    r_core = lagrange[0]
    r_tidal = lagrange[1]
    
    return np.log10(r_tidal / r_core)



def calculate_crossing_time(
    positions: np.ndarray,
    velocities: np.ndarray
) -> float:
    """
    Calculate crossing time: t_cross = R / v_typical
    
    Args:
        positions: (N, 3) positions
        velocities: (N, 3) velocities
        
    Returns:
        Crossing time (same units as input)
    """
    # System size
    radii = np.linalg.norm(positions, axis=1)
    R = np.max(radii)
    
    # Typical velocity
    vel_mag = np.linalg.norm(velocities, axis=1)
    v_typical = np.median(vel_mag)
    
    if v_typical < 1e-10:
        return np.inf
    
    t_cross = R / v_typical
    
    return t_cross


def calculate_relaxation_time(
    N: int,
    crossing_time: float
) -> float:
    """
    Calculate two-body relaxation time.
    
    t_relax ≈ (N / 8 ln(N)) * t_cross
    
    Args:
        N: Number of particles
        crossing_time: Crossing time
        
    Returns:
        Relaxation time
    """
    if N < 2:
        return np.inf
    
    t_relax = (N / (8.0 * np.log(N))) * crossing_time
    
    return t_relax



def calculate_system_properties(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive system properties.
    
    Args:
        positions: (N, 3) positions
        velocities: (N, 3) velocities
        masses: (N,) masses
        
    Returns:
        Dictionary of properties
    """
    from simulator.physics import (
        calculate_total_energy,
        calculate_virial_ratio,
        center_of_mass
    )
    
    N = len(masses)
    
    ke, pe, E_total = calculate_total_energy(positions, velocities, masses)
    Q = calculate_virial_ratio(positions, velocities, masses)
    
    lagrange = calculate_lagrange_radii(
        positions, masses,
        np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    )
    r_core = calculate_core_radius(positions, masses, method="lagrange")
    r_half = lagrange[2]  # r_50%
    
    concentration = calculate_concentration(positions, masses)
    
    t_cross = calculate_crossing_time(positions, velocities)
    t_relax = calculate_relaxation_time(N, t_cross)
    
    com = center_of_mass(positions, masses)
    com_distance = np.linalg.norm(com)
    
    return {
        'N': N,
        'total_mass': masses.sum(),
        'kinetic_energy': ke,
        'potential_energy': pe,
        'total_energy': E_total,
        'virial_ratio': Q,
        'r_core': r_core,
        'r_10': lagrange[0],
        'r_25': lagrange[1],
        'r_50': r_half,
        'r_75': lagrange[3],
        'r_90': lagrange[4],
        'concentration': concentration,
        'crossing_time': t_cross,
        'relaxation_time': t_relax,
        'com_distance': com_distance
    }


def print_system_properties(properties: Dict[str, float]):
    """
    Print system properties in readable format.
    
    Args:
        properties: Dictionary from calculate_system_properties
    """
    print("\n" + "=" * 60)
    print("SYSTEM PROPERTIES")
    print("=" * 60)
    
    print(f"\nParticles: {properties['N']}")
    print(f"Total Mass: {properties['total_mass']:.2e} M_sun")
    
    print(f"\nEnergy:")
    print(f"  Kinetic:   {properties['kinetic_energy']:12.6f}")
    print(f"  Potential: {properties['potential_energy']:12.6f}")
    print(f"  Total:     {properties['total_energy']:12.6f}")
    print(f"  Virial Q:  {properties['virial_ratio']:12.6f}")
    
    print(f"\nStructure:")
    print(f"  Core radius (r_10%):  {properties['r_core']:.3f} pc")
    print(f"  Quarter-mass (r_25%): {properties['r_25']:.3f} pc")
    print(f"  Half-mass (r_50%):    {properties['r_50']:.3f} pc")
    print(f"  Three-quarter (r_75%): {properties['r_75']:.3f} pc")
    print(f"  Tidal radius (r_90%): {properties['r_90']:.3f} pc")
    print(f"  Concentration c:      {properties['concentration']:.3f}")
    
    print(f"\nTime Scales:")
    print(f"  Crossing time:    {properties['crossing_time']:.3f} Gyr")
    print(f"  Relaxation time:  {properties['relaxation_time']:.3f} Gyr")
    
    print(f"\nCenter of Mass:")
    print(f"  Distance from origin: {properties['com_distance']:.6f} pc")
    
    print("=" * 60)



__all__ = [
    'calculate_lagrange_radii',
    'calculate_half_mass_radius',
    'calculate_core_radius',
    'calculate_concentration',
    'calculate_crossing_time',
    'calculate_relaxation_time',
    'calculate_system_properties',
    'print_system_properties'
]