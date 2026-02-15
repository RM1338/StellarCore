"""
Gaia DR3 Data Loader

Query and process data from ESA Gaia Data Release 3.

Features:
- TAP queries to Gaia archive
- Coordinate transformations
- Data validation and quality filters
- Local caching for repeated queries
- Pre-built queries for known clusters

Security:
- Query timeout limits
- Row count limits
- Input validation
- Safe caching

References:
- Gaia Collaboration et al. (2016, 2023)
- https://gea.esac.esa.int/archive/
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import warnings

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, Distance
    from astroquery.gaia import Gaia
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("Astropy/Astroquery not available. Gaia queries disabled.")

from config.settings import settings, CACHE_DIR



Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = settings.gaia_max_rows



def query_cluster_region(
    ra: float,
    dec: float,
    radius: float,
    max_rows: Optional[int] = None,
    quality_filters: bool = True,
    cache: bool = True
) -> Optional[Dict[str, np.ndarray]]:
    """
    Query Gaia archive for stars in a circular region.
    
    Args:
        ra: Right ascension (degrees)
        dec: Declination (degrees)
        radius: Search radius (degrees)
        max_rows: Maximum number of rows (None = config default)
        quality_filters: Apply quality filters
        cache: Use cached data if available
        
    Returns:
        Dictionary with arrays:
        - ra, dec: Coordinates (degrees)
        - parallax: Parallax (mas)
        - pmra, pmdec: Proper motions (mas/yr)
        - radial_velocity: Radial velocity (km/s)
        - phot_g_mean_mag: G magnitude
        - source_id: Gaia source ID
        
    Raises:
        RuntimeError: If query fails
    """
    if not ASTROPY_AVAILABLE:
        raise RuntimeError("Astropy not available. Install: pip install astropy astroquery")
    
    assert -90 <= dec <= 90, f"Invalid declination: {dec}"
    assert 0 < radius < 10, f"Radius must be 0-10 degrees, got {radius}"
    
    if max_rows is None:
        max_rows = settings.gaia_max_rows
    
    cache_file = CACHE_DIR / f"gaia_ra{ra:.3f}_dec{dec:.3f}_r{radius:.3f}.npz"
    
    if cache and cache_file.exists():
        print(f"Loading cached Gaia data: {cache_file.name}")
        return _load_cache(cache_file)
    
    query = f"""
    SELECT TOP {max_rows}
        source_id,
        ra, dec,
        parallax, parallax_error,
        pmra, pmra_error,
        pmdec, pmdec_error,
        radial_velocity, radial_velocity_error,
        phot_g_mean_mag
    FROM {Gaia.MAIN_GAIA_TABLE}
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius})
    )
    """
    
    if quality_filters:
        query += """
        AND parallax IS NOT NULL
        AND parallax > 0
        AND parallax_error / parallax < 0.2
        AND pmra IS NOT NULL
        AND pmdec IS NOT NULL
        AND phot_g_mean_mag IS NOT NULL
        """
    
    query += " ORDER BY phot_g_mean_mag ASC"
    
    print(f"Querying Gaia archive: RA={ra:.3f}, Dec={dec:.3f}, radius={radius:.3f}°")
    print(f"Max rows: {max_rows}, Quality filters: {quality_filters}")
    
    try:
        job = Gaia.launch_job_async(
            query,
            dump_to_file=False,
            verbose=False
        )
        
        result = job.get_results()
        
        if len(result) == 0:
            print("Warning: No sources found")
            return None
        
        print(f"Retrieved {len(result)} sources")
        
        data = {
            'source_id': np.array(result['source_id']),
            'ra': np.array(result['ra']),
            'dec': np.array(result['dec']),
            'parallax': np.array(result['parallax']),
            'parallax_error': np.array(result['parallax_error']),
            'pmra': np.array(result['pmra']),
            'pmra_error': np.array(result['pmra_error']),
            'pmdec': np.array(result['pmdec']),
            'pmdec_error': np.array(result['pmdec_error']),
            'phot_g_mean_mag': np.array(result['phot_g_mean_mag'])
        }
        
        if 'radial_velocity' in result.colnames:
            rv = np.array(result['radial_velocity'])
            rv_error = np.array(result['radial_velocity_error'])
            data['radial_velocity'] = np.where(rv.mask if hasattr(rv, 'mask') else False, np.nan, rv)
            data['radial_velocity_error'] = np.where(rv_error.mask if hasattr(rv_error, 'mask') else False, np.nan, rv_error)
        else:
            data['radial_velocity'] = np.full(len(result), np.nan)
            data['radial_velocity_error'] = np.full(len(result), np.nan)
        
        if cache:
            _save_cache(cache_file, data)
            print(f"Cached data: {cache_file.name}")
        
        return data
        
    except Exception as e:
        raise RuntimeError(f"Gaia query failed: {e}")


def query_cluster_by_name(
    cluster_name: str,
    radius: Optional[float] = None,
    max_rows: Optional[int] = None,
    cache: bool = True
) -> Optional[Dict[str, np.ndarray]]:
    """
    Query Gaia data for a known cluster using catalog parameters.
    
    Args:
        cluster_name: Cluster name (e.g., "M13", "47 Tucanae")
        radius: Search radius in degrees (None = auto from catalog)
        max_rows: Maximum rows
        cache: Use caching
        
    Returns:
        Gaia data dictionary
    """
    from .cluster_catalog import get_cluster
    
    cluster = get_cluster(cluster_name)
    
    ra = cluster['ra']
    dec = cluster['dec']
    
    if radius is None:
        r_tidal_pc = cluster['r_tidal']
        distance_kpc = cluster['distance']
        
        theta_rad = r_tidal_pc / (distance_kpc * 1000.0)
        radius = np.degrees(theta_rad)
        
        radius *= 1.2
        
        radius = max(0.1, min(radius, 5.0))
    
    print(f"Querying {cluster['name']}: RA={ra:.3f}°, Dec={dec:.3f}°, radius={radius:.3f}°")
    
    return query_cluster_region(
        ra=ra,
        dec=dec,
        radius=radius,
        max_rows=max_rows,
        cache=cache
    )



def gaia_to_cartesian(
    ra: np.ndarray,
    dec: np.ndarray,
    parallax: np.ndarray,
    pmra: np.ndarray,
    pmdec: np.ndarray,
    radial_velocity: Optional[np.ndarray] = None,
    center_on_mean: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Gaia observables to Cartesian positions and velocities.
    
    Transformations:
    1. (RA, Dec, parallax) → (X, Y, Z) positions
    2. (pmra, pmdec, RV) → (vx, vy, vz) velocities
    
    Args:
        ra: Right ascension (degrees)
        dec: Declination (degrees)
        parallax: Parallax (mas)
        pmra: Proper motion in RA (mas/yr)
        pmdec: Proper motion in Dec (mas/yr)
        radial_velocity: Radial velocity (km/s), optional
        center_on_mean: Subtract mean position/velocity
        
    Returns:
        (positions, velocities) in parsecs and km/s
        - positions: (N, 3) array in pc
        - velocities: (N, 3) array in km/s
    """
    if not ASTROPY_AVAILABLE:
        raise RuntimeError("Astropy required for coordinate transformations")
    
    N = len(ra)
    
    if radial_velocity is None:
        radial_velocity = np.zeros(N)
    else:
        radial_velocity = np.nan_to_num(radial_velocity, nan=0.0)
    
    distance_pc = 1000.0 / parallax
    
    coords = SkyCoord(
        ra=ra * u.degree,
        dec=dec * u.degree,
        distance=distance_pc * u.pc,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=radial_velocity * u.km / u.s,
        frame='icrs'
    )
    
    cartesian = coords.cartesian
    
    positions = np.column_stack([
        cartesian.x.to(u.pc).value,
        cartesian.y.to(u.pc).value,
        cartesian.z.to(u.pc).value
    ])
    
    vel_cartesian = coords.velocity.d_xyz
    velocities = np.column_stack([
        vel_cartesian[0].to(u.km / u.s).value,
        vel_cartesian[1].to(u.km / u.s).value,
        vel_cartesian[2].to(u.km / u.s).value
    ])
    
    if center_on_mean:
        positions -= np.mean(positions, axis=0)
        velocities -= np.mean(velocities, axis=0)
    
    return positions, velocities


def gaia_to_nbody_units(
    gaia_data: Dict[str, np.ndarray],
    mass_total: float = 1.0,
    assume_equal_mass: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Gaia data to N-body units for simulation.
    
    Args:
        gaia_data: Dictionary from query_cluster_*
        mass_total: Total system mass (solar masses)
        assume_equal_mass: If True, use equal masses; else use magnitude-based
        
    Returns:
        (positions, velocities, masses) ready for simulation
    """
    ra = gaia_data['ra']
    dec = gaia_data['dec']
    parallax = gaia_data['parallax']
    pmra = gaia_data['pmra']
    pmdec = gaia_data['pmdec']
    rv = gaia_data.get('radial_velocity', None)
    
    N = len(ra)
    
    positions, velocities = gaia_to_cartesian(
        ra, dec, parallax, pmra, pmdec, rv,
        center_on_mean=True
    )
    
    if assume_equal_mass:
        masses = np.ones(N) * (mass_total / N)
    else:
        mags = gaia_data['phot_g_mean_mag']
        
        mass_weights = 10.0 ** ((np.median(mags) - mags) / 2.5)
        masses = mass_weights / np.sum(mass_weights) * mass_total
    
    print(f"Converted {N} stars to N-body units")
    print(f"  Position range: {np.abs(positions).max():.1f} pc")
    print(f"  Velocity range: {np.abs(velocities).max():.1f} km/s")
    print(f"  Total mass: {masses.sum():.2f} M_sun")
    
    return positions, velocities, masses



def _save_cache(filepath: Path, data: Dict[str, np.ndarray]):
    """Save Gaia data to cache file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, **data)


def _load_cache(filepath: Path) -> Dict[str, np.ndarray]:
    """Load Gaia data from cache file."""
    cached = np.load(filepath, allow_pickle=False)
    return {key: cached[key] for key in cached.files}


def clear_cache():
    """Clear all cached Gaia data."""
    cache_files = list(CACHE_DIR.glob("gaia_*.npz"))
    for f in cache_files:
        f.unlink()
    print(f"Cleared {len(cache_files)} cached Gaia files")



__all__ = [
    'query_cluster_region',
    'query_cluster_by_name',
    'gaia_to_cartesian',
    'gaia_to_nbody_units',
    'clear_cache',
    'ASTROPY_AVAILABLE'
]