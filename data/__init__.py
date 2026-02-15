"""
Data Module

Cluster catalog and Gaia data management.
"""

from .cluster_catalog import (
    CLUSTER_CATALOG,
    get_cluster,
    list_clusters,
    get_cluster_names_formatted,
    print_cluster_info
)

from .gaia_loader import (
    query_cluster_region,
    query_cluster_by_name,
    gaia_to_cartesian,
    gaia_to_nbody_units,
    clear_cache,
    ASTROPY_AVAILABLE
)

__all__ = [
    'CLUSTER_CATALOG',
    'get_cluster',
    'list_clusters',
    'get_cluster_names_formatted',
    'print_cluster_info',
    'query_cluster_region',
    'query_cluster_by_name',
    'gaia_to_cartesian',
    'gaia_to_nbody_units',
    'clear_cache',
    'ASTROPY_AVAILABLE'
]