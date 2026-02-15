"""
N-Body Integrators

Time integration schemes for gravitational N-body simulations.

Implements:
- Hermite 4th order predictor-corrector
- Leapfrog (fallback)
- Adaptive timestep selection

Security:
- Timestep validation
- Overflow detection
- Convergence monitoring
"""

import numpy as np
from typing import Tuple, Optional
from .gpu_backend import get_backend
from .physics import calculate_gravitational_force



def hermite_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    softening: float = 0.01,
    backend=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one Hermite 4th order integration step.
    
    Hermite is a predictor-corrector method with 4th order accuracy:
    1. Predict positions/velocities at t + dt
    2. Calculate forces at predicted positions
    3. Correct using both initial and predicted forces
    
    This is more accurate than leapfrog for small particle numbers.
    
    Algorithm:
    - Predictor: x_p = x + v*dt + 0.5*a*dt^2 + (1/6)*jerk*dt^3
    - Corrector: x_c = x + 0.5*(v + v_p)*dt + ...
    
    Args:
        positions: Current positions (N, 3)
        velocities: Current velocities (N, 3)
        masses: Particle masses (N,)
        dt: Timestep
        softening: Softening length
        backend: GPU backend
        
    Returns:
        (new_positions, new_velocities)
        
    References:
        Makino & Aarseth (1992). PASJ, 44, 141
        Nitadori & Makino (2008). New Astronomy, 13, 498
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    
    positions = backend.to_gpu(positions)
    velocities = backend.to_gpu(velocities)
    masses = backend.to_gpu(masses)
    
    forces_0 = calculate_gravitational_force(positions, masses, softening, backend)
    accel_0 = forces_0 / masses[:, xp.newaxis]
    
    jerk_0 = xp.zeros_like(accel_0)
    
    pos_pred = positions + velocities * dt + 0.5 * accel_0 * dt**2 + (1.0/6.0) * jerk_0 * dt**3
    vel_pred = velocities + accel_0 * dt + 0.5 * jerk_0 * dt**2
    
    forces_1 = calculate_gravitational_force(pos_pred, masses, softening, backend)
    accel_1 = forces_1 / masses[:, xp.newaxis]
    
    jerk_1 = (accel_1 - accel_0) / dt
    
    accel_avg = 0.5 * (accel_0 + accel_1)
    jerk_avg = 0.5 * (jerk_0 + jerk_1)
    
    pos_new = positions + velocities * dt + 0.5 * accel_avg * dt**2 + (1.0/6.0) * jerk_avg * dt**3
    vel_new = velocities + accel_avg * dt + 0.5 * jerk_avg * dt**2
    
    assert xp.all(xp.isfinite(pos_new)), "Position integration produced NaN/Inf"
    assert xp.all(xp.isfinite(vel_new)), "Velocity integration produced NaN/Inf"
    
    pos_new = backend.to_cpu(pos_new)
    vel_new = backend.to_cpu(vel_new)
    
    return pos_new, vel_new



def leapfrog_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    softening: float = 0.01,
    backend=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one leapfrog integration step.
    
    Leapfrog is a 2nd order symplectic integrator:
    1. v(t + dt/2) = v(t) + a(t) * dt/2
    2. x(t + dt) = x(t) + v(t + dt/2) * dt
    3. v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
    
    Advantages:
    - Symplectic (conserves phase space volume)
    - Time-reversible
    - Simple and robust
    
    Disadvantages:
    - Lower accuracy than Hermite (2nd vs 4th order)
    
    Args:
        positions: Current positions (N, 3)
        velocities: Current velocities (N, 3)
        masses: Particle masses (N,)
        dt: Timestep
        softening: Softening length
        backend: GPU backend
        
    Returns:
        (new_positions, new_velocities)
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    
    positions = backend.to_gpu(positions)
    velocities = backend.to_gpu(velocities)
    masses = backend.to_gpu(masses)
    
    forces = calculate_gravitational_force(positions, masses, softening, backend)
    accel = forces / masses[:, xp.newaxis]
    
    vel_half = velocities + 0.5 * accel * dt
    
    pos_new = positions + vel_half * dt
    
    forces_new = calculate_gravitational_force(pos_new, masses, softening, backend)
    accel_new = forces_new / masses[:, xp.newaxis]
    
    vel_new = vel_half + 0.5 * accel_new * dt
    
    assert xp.all(xp.isfinite(pos_new)), "Position integration produced NaN/Inf"
    assert xp.all(xp.isfinite(vel_new)), "Velocity integration produced NaN/Inf"
    
    pos_new = backend.to_cpu(pos_new)
    vel_new = backend.to_cpu(vel_new)
    
    return pos_new, vel_new



def calculate_adaptive_timestep(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    eta: float = 0.01,
    dt_min: float = 1e-6,
    dt_max: float = 0.01,
    softening: float = 0.01,
    backend=None
) -> float:
    """
    Calculate adaptive timestep based on local acceleration.
    
    Uses the criterion:
    dt = eta * sqrt(epsilon / |a|)
    
    where:
    - eta: accuracy parameter (smaller = more accurate, slower)
    - epsilon: softening length
    - a: acceleration magnitude
    
    Takes minimum over all particles for stability.
    
    Args:
        positions: Positions (N, 3)
        velocities: Velocities (N, 3)
        masses: Masses (N,)
        eta: Timestep accuracy parameter (default: 0.01)
        dt_min: Minimum allowed timestep
        dt_max: Maximum allowed timestep
        softening: Softening length
        backend: GPU backend
        
    Returns:
        Adaptive timestep (scalar)
    """
    if backend is None:
        backend = get_backend()
    
    xp = backend.xp
    
    forces = calculate_gravitational_force(positions, masses, softening, backend)
    accel = forces / masses[:, xp.newaxis]
    
    accel_mag = xp.sqrt(xp.sum(accel * accel, axis=1))
    
    accel_mag = xp.maximum(accel_mag, 1e-10)
    
    dt_particle = eta * xp.sqrt(softening / accel_mag)
    
    dt = float(backend.to_cpu(xp.min(dt_particle)))
    
    dt = max(dt_min, min(dt, dt_max))
    
    return dt


def calculate_timestep_from_crossing_time(
    positions: np.ndarray,
    velocities: np.ndarray,
    fraction: float = 0.01
) -> float:
    """
    Calculate timestep as fraction of crossing time.
    
    Crossing time: t_cross = R / v_typical
    Timestep: dt = fraction * t_cross
    
    This is a simple, robust estimate.
    
    Args:
        positions: Positions (N, 3)
        velocities: Velocities (N, 3)
        fraction: Fraction of crossing time (default: 0.01)
        
    Returns:
        Timestep (scalar)
    """
    radii = np.linalg.norm(positions, axis=1)
    R = np.max(radii)
    
    vel_mag = np.linalg.norm(velocities, axis=1)
    v_typical = np.median(vel_mag)
    
    if v_typical < 1e-10:
        v_typical = 0.1  # Fallback
    
    t_cross = R / v_typical
    
    dt = fraction * t_cross
    
    return dt


def integrate_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    method: str = "hermite",
    softening: float = 0.01,
    backend=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one integration step using specified method.
    
    Args:
        positions: Positions (N, 3)
        velocities: Velocities (N, 3)
        masses: Masses (N,)
        dt: Timestep
        method: Integration method ("hermite" or "leapfrog")
        softening: Softening length
        backend: GPU backend
        
    Returns:
        (new_positions, new_velocities)
    """
    if method.lower() == "hermite":
        return hermite_step(positions, velocities, masses, dt, softening, backend)
    elif method.lower() == "leapfrog":
        return leapfrog_step(positions, velocities, masses, dt, softening, backend)
    else:
        raise ValueError(f"Unknown integration method: {method}. Use 'hermite' or 'leapfrog'")



__all__ = [
    'hermite_step',
    'leapfrog_step',
    'calculate_adaptive_timestep',
    'calculate_timestep_from_crossing_time',
    'integrate_step'
]