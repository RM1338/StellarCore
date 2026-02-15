"""
Globular Cluster Catalog

Physical parameters for real globular clusters.

Data sources:
- Harris (1996, 2010 edition) catalog
- Baumgardt & Hilker (2018) updated parameters
- Gaia DR3 measurements

Clusters included:
- M13 (NGC 6205): Bright northern cluster
- 47 Tucanae (NGC 104): Massive southern cluster
- M4 (NGC 6121): Nearest globular cluster
- NGC 6397: Metal-poor, well-studied cluster
"""

from typing import Dict, Any
import numpy as np



CLUSTER_CATALOG: Dict[str, Dict[str, Any]] = {
    "M13": {
        "name": "M13",
        "ngc": "NGC 6205",
        "other_names": ["Hercules Cluster"],
        
        "ra": 250.421,  # degrees
        "dec": 36.461,  # degrees
        "l": 59.01,  # Galactic longitude
        "b": 40.91,  # Galactic latitude
        
        "distance": 7.1,  # kpc
        "distance_error": 0.3,  # kpc
        
        "r_core": 0.36,  # pc - core radius
        "r_half": 2.83,  # pc - half-light radius
        "r_tidal": 38.0,  # pc - tidal radius
        "concentration": 1.58,  # log(r_tidal/r_core)
        
        "mass": 6.0e5,  # solar masses
        "n_stars": 3.0e5,  # approximate
        "metallicity": -1.53,  # [Fe/H]
        "age": 11.5,  # Gyr
        
        "v_radial": -244.2,  # km/s - radial velocity
        "pm_ra": -3.44,  # mas/yr - proper motion RA
        "pm_dec": -17.08,  # mas/yr - proper motion Dec
        
        "references": [
            "Harris (1996, 2010 edition)",
            "Baumgardt & Hilker (2018)"
        ]
    },
    
    "47 Tucanae": {
        "name": "47 Tucanae",
        "ngc": "NGC 104",
        "other_names": ["47 Tuc"],
        
        "ra": 6.024,  # degrees
        "dec": -72.081,  # degrees
        "l": 305.89,
        "b": -44.89,
        
        "distance": 4.5,  # kpc
        "distance_error": 0.15,
        
        "r_core": 0.36,  # pc
        "r_half": 3.17,  # pc
        "r_tidal": 42.0,  # pc
        "concentration": 2.07,
        
        "mass": 7.0e5,  # solar masses
        "n_stars": 1.0e6,
        "metallicity": -0.76,  # [Fe/H]
        "age": 12.0,  # Gyr
        
        "v_radial": -18.7,  # km/s
        "pm_ra": 5.26,  # mas/yr
        "pm_dec": -2.53,  # mas/yr
        
        "references": [
            "Harris (1996, 2010 edition)",
            "Baumgardt & Hilker (2018)"
        ]
    },
    
    "M4": {
        "name": "M4",
        "ngc": "NGC 6121",
        "other_names": [],
        
        "ra": 245.897,  # degrees
        "dec": -26.526,  # degrees
        "l": 350.97,
        "b": 15.97,
        
        "distance": 1.8,  # kpc - nearest globular cluster
        "distance_error": 0.1,
        
        "r_core": 0.60,  # pc
        "r_half": 2.93,  # pc
        "r_tidal": 23.0,  # pc
        "concentration": 1.58,
        
        "mass": 6.7e4,  # solar masses
        "n_stars": 1.0e5,
        "metallicity": -1.18,  # [Fe/H]
        "age": 12.2,  # Gyr
        
        "v_radial": 70.4,  # km/s
        "pm_ra": -12.54,  # mas/yr
        "pm_dec": -19.07,  # mas/yr
        
        "references": [
            "Harris (1996, 2010 edition)",
            "Baumgardt & Hilker (2018)"
        ]
    },
    
    "NGC 6397": {
        "name": "NGC 6397",
        "ngc": "NGC 6397",
        "other_names": [],
        
        "ra": 265.175,  # degrees
        "dec": -53.674,  # degrees
        "l": 338.17,
        "b": -11.96,
        
        "distance": 2.5,  # kpc - one of the nearest
        "distance_error": 0.1,
        
        "r_core": 0.05,  # pc - post-core-collapse
        "r_half": 2.06,  # pc
        "r_tidal": 23.0,  # pc
        "concentration": 2.66,
        
        "mass": 9.0e4,  # solar masses
        "n_stars": 4.0e5,
        "metallicity": -2.02,  # [Fe/H] - very metal-poor
        "age": 13.4,  # Gyr - one of the oldest
        
        "v_radial": 18.4,  # km/s
        "pm_ra": 3.27,  # mas/yr
        "pm_dec": -17.59,  # mas/yr
        
        "references": [
            "Harris (1996, 2010 edition)",
            "Baumgardt & Hilker (2018)",
            "Arnold et al. (2025) - validation data"
        ]
    }
}



def get_cluster(name: str) -> Dict[str, Any]:
    """
    Get cluster parameters by name.
    
    Args:
        name: Cluster name (e.g., "M13", "47 Tucanae", "NGC 6205")
        
    Returns:
        Dictionary of cluster parameters
        
    Raises:
        KeyError: If cluster not found
    """
    name_clean = name.strip().upper()
    
    for key in CLUSTER_CATALOG:
        if key.upper() == name_clean:
            return CLUSTER_CATALOG[key].copy()
    
    for key, cluster in CLUSTER_CATALOG.items():
        if cluster.get("ngc", "").upper() == name_clean:
            return cluster.copy()
    
    for key, cluster in CLUSTER_CATALOG.items():
        for other_name in cluster.get("other_names", []):
            if other_name.upper() == name_clean:
                return cluster.copy()
    
    available = list(CLUSTER_CATALOG.keys())
    raise KeyError(f"Cluster '{name}' not found. Available: {available}")


def list_clusters() -> list:
    """
    Get list of available clusters.
    
    Returns:
        List of cluster names
    """
    return list(CLUSTER_CATALOG.keys())


def get_cluster_names_formatted() -> list:
    """
    Get formatted cluster names for display in UI.
    
    Returns:
        List of formatted strings like "M13 (NGC 6205)"
    """
    formatted = []
    for key, cluster in CLUSTER_CATALOG.items():
        ngc = cluster.get("ngc", "")
        if ngc:
            formatted.append(f"{cluster['name']} ({ngc})")
        else:
            formatted.append(cluster['name'])
    return formatted


def print_cluster_info(name: str):
    """
    Print detailed information about a cluster.
    
    Args:
        name: Cluster name
    """
    cluster = get_cluster(name)
    
    print("=" * 60)
    print(f"Cluster: {cluster['name']} ({cluster.get('ngc', 'N/A')})")
    print("=" * 60)
    
    print(f"\nPosition:")
    print(f"  RA: {cluster['ra']:.3f}°")
    print(f"  Dec: {cluster['dec']:.3f}°")
    print(f"  Distance: {cluster['distance']:.2f} ± {cluster['distance_error']:.2f} kpc")
    
    print(f"\nStructure:")
    print(f"  Core radius: {cluster['r_core']:.2f} pc")
    print(f"  Half-light radius: {cluster['r_half']:.2f} pc")
    print(f"  Tidal radius: {cluster['r_tidal']:.1f} pc")
    print(f"  Concentration: {cluster['concentration']:.2f}")
    
    print(f"\nStellar Population:")
    print(f"  Mass: {cluster['mass']:.1e} M_sun")
    print(f"  Number of stars: {cluster['n_stars']:.1e}")
    print(f"  Metallicity [Fe/H]: {cluster['metallicity']:.2f}")
    print(f"  Age: {cluster['age']:.1f} Gyr")
    
    print(f"\nKinematics:")
    print(f"  Radial velocity: {cluster['v_radial']:.1f} km/s")
    print(f"  Proper motion (RA): {cluster['pm_ra']:.2f} mas/yr")
    print(f"  Proper motion (Dec): {cluster['pm_dec']:.2f} mas/yr")
    
    print(f"\nReferences:")
    for ref in cluster.get('references', []):
        print(f"  - {ref}")
    
    print("=" * 60)



__all__ = [
    'CLUSTER_CATALOG',
    'get_cluster',
    'list_clusters',
    'get_cluster_names_formatted',
    'print_cluster_info'
]