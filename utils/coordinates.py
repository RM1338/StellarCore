"""
Coordinate Transformations

Utilities for astronomical coordinate conversions.

Transformations:
- Spherical ↔ Cartesian
- Equatorial (RA/Dec) ↔ Galactic (l/b)
- Distance/velocity unit conversions
"""

import numpy as np
from typing import Tuple



def spherical_to_cartesian(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical to Cartesian coordinates.
    
    Convention:
    - r: radial distance
    - theta: polar angle from z-axis (0 to π)
    - phi: azimuthal angle in xy-plane (0 to 2π)
    
    Args:
        r: Radial distance
        theta: Polar angle (radians)
        phi: Azimuthal angle (radians)
        
    Returns:
        (x, y, z) Cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return x, y, z


def cartesian_to_spherical(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian to spherical coordinates.
    
    Args:
        x, y, z: Cartesian coordinates
        
    Returns:
        (r, theta, phi) spherical coordinates
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    
    return r, theta, phi



def radec_to_xyz(
    ra: np.ndarray,
    dec: np.ndarray,
    distance: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert equatorial coordinates to Cartesian.
    
    Args:
        ra: Right ascension (degrees)
        dec: Declination (degrees)
        distance: Distance (any units)
        
    Returns:
        (x, y, z) in same units as distance
    """
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    x = distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance * np.cos(dec_rad) * np.sin(ra_rad)
    z = distance * np.sin(dec_rad)
    
    return x, y, z


def xyz_to_radec(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian to equatorial coordinates.
    
    Args:
        x, y, z: Cartesian coordinates
        
    Returns:
        (ra, dec, distance)
        - ra: degrees (0 to 360)
        - dec: degrees (-90 to 90)
        - distance: same units as input
    """
    distance = np.sqrt(x**2 + y**2 + z**2)
    
    dec = np.degrees(np.arcsin(z / distance))
    ra = np.degrees(np.arctan2(y, x))
    
    ra = np.where(ra < 0, ra + 360, ra)
    
    return ra, dec, distance



def parallax_to_distance(parallax_mas: float) -> float:
    """
    Convert parallax to distance.
    
    Args:
        parallax_mas: Parallax in milliarcseconds
        
    Returns:
        Distance in parsecs
    """
    return 1000.0 / parallax_mas


def distance_to_parallax(distance_pc: float) -> float:
    """
    Convert distance to parallax.
    
    Args:
        distance_pc: Distance in parsecs
        
    Returns:
        Parallax in milliarcseconds
    """
    return 1000.0 / distance_pc


def proper_motion_to_velocity(
    pm_mas_yr: float,
    distance_pc: float
) -> float:
    """
    Convert proper motion to tangential velocity.
    
    v_tan = 4.74 * μ * d
    
    where:
    - μ: proper motion (arcsec/yr)
    - d: distance (pc)
    - v_tan: velocity (km/s)
    
    Args:
        pm_mas_yr: Proper motion (mas/yr)
        distance_pc: Distance (pc)
        
    Returns:
        Tangential velocity (km/s)
    """
    pm_arcsec_yr = pm_mas_yr / 1000.0
    v_tan = 4.74 * pm_arcsec_yr * distance_pc
    return v_tan



__all__ = [
    'spherical_to_cartesian',
    'cartesian_to_spherical',
    'radec_to_xyz',
    'xyz_to_radec',
    'parallax_to_distance',
    'distance_to_parallax',
    'proper_motion_to_velocity'
]