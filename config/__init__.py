"""
Configuration Module
Centralized configuration management for StellarCore
"""

from .settings import (
    settings,
    PhysicsConstants,
    SimulationDefaults,
    BASE_DIR,
    DATA_DIR,
    LOG_DIR
)

__all__ = [
    'settings',
    'PhysicsConstants',
    'SimulationDefaults',
    'BASE_DIR',
    'DATA_DIR',
    'CACHE_DIR'
    'LOG_DIR'
]