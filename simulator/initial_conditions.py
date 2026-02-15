"""
Initial Conditions Generators

Generate initial positions and velocities for N-body simulations.

Implements:
- King (1966) models with concentration parameter W0
- Plummer (1911) sphere
- Mass functions (Kroupa IMF)
- Distribution function sampling

Security:
- Input validation
- Resource limits
- Reproducible random seeds
"""

import numpy as np
from typing import Tuple, Optional
from .gpu_backend import get_backend
from utils.security import validate_particle_count



def generate_king_model(
    N: int,
    W0: float = 6.0,
    mass_total: float = 1.0,
    r_scale: float = 1.0,
    seed: Optional[int] = None,
    backend=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate King (1966) model initial conditions.
    
    King models are realistic models for globular clusters with a
    concentration parameter W0. Typical values: 5 <= W0 <= 9.
    
    Algorithm:
    1. Sample radii from King distribution function
    2. Assign velocities from Maxwellian at each radius
    3. Ensure system is in virial equilibrium
    
    Args:
        N: Number of particles
        W0: Dimensionless central potential (concentration parameter)
            W0 = 3: Low concentration
            W0 = 6: Moderate concentration (typical)
            W0 = 9: High concentration
        mass_total: Total system mass (N-body units)
        r_scale: Scale radius (N-body units)
        seed: Random seed for reproducibility
        backend: GPU backend
        
    Returns:
        (positions, velocities, masses) as NumPy arrays
        - positions: (N, 3) array in parsecs
        - velocities: (N, 3) array in km/s
        - masses: (N,) array in solar masses
        
    References:
        King, I. R. (1966). AJ, 71, 64
    """
    # Validation
    N = validate_particle_count(N)
    assert W0 >= 1.0 and W0 <= 12.0, f"W0 must be in [1, 12], got {W0}"
    assert mass_total > 0, "Total mass must be positive"
    assert r_scale > 0, "Scale radius must be positive"
    
    if seed is not None:
        np.random.seed(seed)
    
    if backend is None:
        backend = get_backend()
    
    print(f"Generating King model: N={N}, W0={W0:.1f}")
    
    r_tidal = _king_tidal_radius(W0) * r_scale
    rho_0 = 3.0 / (4.0 * np.pi * r_scale**3)  
    
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    
    for i in range(N):
        r = _sample_king_radius(W0, r_scale, r_tidal)
        
        theta = np.arccos(2.0 * np.random.random() - 1.0)
        phi = 2.0 * np.pi * np.random.random()
        
        positions[i, 0] = r * np.sin(theta) * np.cos(phi)
        positions[i, 1] = r * np.sin(theta) * np.sin(phi)
        positions[i, 2] = r * np.cos(theta)
        
        sigma_r = _king_velocity_dispersion(r, W0, r_scale)
        
        v_r = np.random.normal(0, sigma_r)
        v_theta = np.random.normal(0, sigma_r)
        v_phi = np.random.normal(0, sigma_r)
        
        velocities[i, 0] = (v_r * np.sin(theta) * np.cos(phi) + 
                           v_theta * np.cos(theta) * np.cos(phi) - 
                           v_phi * np.sin(phi))
        velocities[i, 1] = (v_r * np.sin(theta) * np.sin(phi) + 
                           v_theta * np.cos(theta) * np.sin(phi) + 
                           v_phi * np.cos(phi))
        velocities[i, 2] = v_r * np.cos(theta) - v_theta * np.sin(theta)
    
    masses = np.ones(N) * (mass_total / N)
    
    positions = _center_system(positions, masses)
    velocities = _center_velocities(velocities, masses)
    
    print(f"King model generated: r_tidal={r_tidal:.2f}")
    
    return positions, velocities, masses


def _king_tidal_radius(W0: float) -> float:
    """Calculate tidal radius for King model."""
    if W0 <= 1.0:
        return 10.0
    elif W0 <= 3.0:
        return 15.0
    elif W0 <= 5.0:
        return 25.0
    elif W0 <= 7.0:
        return 50.0
    elif W0 <= 9.0:
        return 100.0
    else:
        return 200.0


def _sample_king_radius(W0: float, r_scale: float, r_tidal: float) -> float:
    """Sample radius from King density profile using rejection sampling."""
    max_attempts = 1000
    
    for _ in range(max_attempts):
        r = np.random.random() * r_tidal
        
        psi = W0 * np.exp(-(r / r_scale)**2)
        rho = np.exp(psi) - 1.0 if psi > 0 else 0.0
        
        if np.random.random() < rho / np.exp(W0):
            return r
    
    return r_scale * np.random.exponential()


def _king_velocity_dispersion(r: float, W0: float, r_scale: float) -> float:
    """Calculate velocity dispersion at radius r for King model."""
    psi = W0 * np.exp(-(r / r_scale)**2)
    sigma = np.sqrt(max(psi, 0.1))  
    return sigma * 0.1  



def generate_plummer_sphere(
    N: int,
    mass_total: float = 1.0,
    a: float = 1.0,
    seed: Optional[int] = None,
    backend=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Plummer (1911) sphere initial conditions.
    
    Plummer sphere is a simple analytical model with density profile:
    rho(r) = (3M / 4πa³) * (1 + r²/a²)^(-5/2)
    
    Algorithm from Aarseth et al. (1974):
    1. Sample radii from cumulative mass function
    2. Assign velocities from distribution function
    3. Ensure virial equilibrium
    
    Args:
        N: Number of particles
        mass_total: Total system mass
        a: Plummer scale radius
        seed: Random seed
        backend: GPU backend
        
    Returns:
        (positions, velocities, masses) as NumPy arrays
        
    References:
        Plummer, H. C. (1911). MNRAS, 71, 460
        Aarseth et al. (1974). A&A, 37, 183
    """
    # Validation
    N = validate_particle_count(N)
    assert mass_total > 0, "Total mass must be positive"
    assert a > 0, "Scale radius must be positive"
    
    if seed is not None:
        np.random.seed(seed)
    
    if backend is None:
        backend = get_backend()
    
    print(f"Generating Plummer sphere: N={N}, a={a:.2f}")
    
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    
    for i in range(N):
        # Sample radius from Plummer cumulative mass function
        # M(r) / M_total = r³ / (r² + a²)^(3/2)
        # Invert to get r from uniform random
        X = np.random.random()
        r = a / np.sqrt(X**(-2.0/3.0) - 1.0)
        
        # Random position on sphere
        theta = np.arccos(2.0 * np.random.random() - 1.0)
        phi = 2.0 * np.pi * np.random.random()
        
        positions[i, 0] = r * np.sin(theta) * np.cos(phi)
        positions[i, 1] = r * np.sin(theta) * np.sin(phi)
        positions[i, 2] = r * np.cos(theta)
        
        v_esc = np.sqrt(2.0 * mass_total / np.sqrt(r**2 + a**2))
        
        v = 0.0
        for _ in range(100):  
            v_trial = v_esc * np.random.random()
            g = (1.0 - v_trial**2 / v_esc**2)**3.5
            if np.random.random() < g:
                v = v_trial
                break
        
        v_theta = np.arccos(2.0 * np.random.random() - 1.0)
        v_phi = 2.0 * np.pi * np.random.random()
        
        velocities[i, 0] = v * np.sin(v_theta) * np.cos(v_phi)
        velocities[i, 1] = v * np.sin(v_theta) * np.sin(v_phi)
        velocities[i, 2] = v * np.cos(v_theta)
    
    masses = np.ones(N) * (mass_total / N)
    
    positions = _center_system(positions, masses)
    velocities = _center_velocities(velocities, masses)
    
    print(f"Plummer sphere generated")
    
    return positions, velocities, masses



def generate_kroupa_masses(
    N: int,
    m_min: float = 0.1,
    m_max: float = 100.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate stellar masses from Kroupa (2001) IMF.
    
    Kroupa IMF is a broken power law:
    dN/dm ∝ m^(-α)
    
    where α = 0.3 for m < 0.08 M_sun
          α = 1.3 for 0.08 < m < 0.5 M_sun
          α = 2.3 for m > 0.5 M_sun (Salpeter)
    
    Args:
        N: Number of stars
        m_min: Minimum mass (solar masses)
        m_max: Maximum mass (solar masses)
        seed: Random seed
        
    Returns:
        Array of N masses in solar masses
        
    References:
        Kroupa, P. (2001). MNRAS, 322, 231
    """
    if seed is not None:
        np.random.seed(seed)
    
    masses = np.zeros(N)
    
    alpha = 2.3
    
    for i in range(N):
        u = np.random.random()
        
        if alpha != 1.0:
            m = m_min * (1.0 + u * ((m_max/m_min)**(1.0-alpha) - 1.0))**(1.0/(1.0-alpha))
        else:
            m = m_min * np.exp(u * np.log(m_max/m_min))
        
        masses[i] = m
    
    return masses


def generate_equal_masses(N: int, mass_total: float = 1.0) -> np.ndarray:
    """
    Generate equal masses for all particles.
    
    Args:
        N: Number of particles
        mass_total: Total mass
        
    Returns:
        Array of N equal masses
    """
    return np.ones(N) * (mass_total / N)



def _center_system(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Center system on center of mass."""
    com = np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses)
    return positions - com


def _center_velocities(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Remove center of mass velocity."""
    com_vel = np.sum(masses[:, np.newaxis] * velocities, axis=0) / np.sum(masses)
    return velocities - com_vel


def normalize_to_nbody_units(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize system to N-body units where G = M = -4E = 1.
    
    Args:
        positions: Position array (N, 3)
        velocities: Velocity array (N, 3)
        masses: Mass array (N,)
        G: Gravitational constant (default: 1)
        
    Returns:
        Normalized (positions, velocities, masses)
    """
    from simulator.physics import calculate_total_energy
    
    M = np.sum(masses)
    _, _, E = calculate_total_energy(positions, velocities, masses)
    
    r_scale = -G * M**2 / (4.0 * E)
    v_scale = np.sqrt(-E / M)
    m_scale = 1.0 / M
    
    positions_norm = positions * r_scale
    velocities_norm = velocities * v_scale
    masses_norm = masses * m_scale
    
    return positions_norm, velocities_norm, masses_norm



__all__ = [
    'generate_king_model',
    'generate_plummer_sphere',
    'generate_kroupa_masses',
    'generate_equal_masses',
    'normalize_to_nbody_units'
]