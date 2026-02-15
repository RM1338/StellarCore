"""
StellarCore Global Configuration

Secure configuration management with environment variable support.

Security Features:
- Environment-based configuration (12-factor app)
- Input validation using Pydantic
- Secure defaults
- No hardcoded secrets
- Type checking
"""

import os
import secrets
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "precomputed"
LOG_DIR = BASE_DIR / "logs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """
    Application settings with secure defaults and validation.
    
    All sensitive values loaded from environment variables.
    Implements OWASP Configuration Management best practices.
    """
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'  
    )
    
    app_env: Literal['development', 'production', 'testing'] = Field(
        default='development',
        description="Application environment"
    )
    
    debug_mode: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        min_length=32,
        description="Secret key for session management"
    )
    
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    
    rate_limit_per_minute: int = Field(
        default=30,
        ge=1,
        le=1000,
        description="Max requests per minute per IP"
    )
    
    rate_limit_per_hour: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Max requests per hour per IP"
    )
    
    gaia_archive_url: str = Field(
        default="https://gea.esac.esa.int/tap-server/tap",
        description="Gaia TAP service URL"
    )
    
    gaia_max_rows: int = Field(
        default=50000,
        ge=100,
        le=100000,
        description="Maximum rows to download (prevent abuse)"
    )
    
    gaia_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Gaia query timeout in seconds"
    )
    
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file upload size in MB"
    )
    
    max_particles: int = Field(
        default=100000,
        ge=100,
        le=1000000,
        description="Maximum N-body particles"
    )
    
    max_simulation_time_gyr: float = Field(
        default=20.0,
        ge=0.1,
        le=100.0,
        description="Maximum simulation time in Gyr"
    )
    
    max_concurrent_simulations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent simulations per session"
    )
    
    session_timeout_minutes: int = Field(
        default=60,
        ge=5,
        le=480,
        description="Session timeout in minutes"
    )
    
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        description="Logging level"
    )
    
    enable_audit_log: bool = Field(
        default=True,
        description="Enable security audit logging"
    )
    
    allowed_origins: str = Field(
        default="http://localhost:8501",
        description="Comma-separated allowed origins"
    )
    
    enable_security_headers: bool = Field(
        default=True,
        description="Enable security headers"
    )
    
    enable_csrf_protection: bool = Field(
        default=True,
        description="Enable CSRF protection"
    )
    
    @field_validator('app_env')
    @classmethod
    def validate_production_settings(cls, v, info):
        """Ensure secure settings in production."""
        if v == 'production':
            # Check debug mode is off
            if info.data.get('debug_mode', False):
                raise ValueError("DEBUG_MODE must be False in production!")
        return v
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v):
        """Ensure secret key is cryptographically secure."""
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters!")
        if v == "your-secret-key-min-32-chars-change-this-in-production":
            raise ValueError("SECRET_KEY must be changed from default value!")
        return v
    
    def get_allowed_origins_list(self) -> list[str]:
        """Parse allowed origins from comma-separated string."""
        return [origin.strip() for origin in self.allowed_origins.split(',')]


try:
    settings = Settings()
except Exception as e:
    print(f"Configuration Error: {e}")
    print("Using fallback safe defaults...")
    settings = Settings(
        app_env='development',
        debug_mode=False,
        rate_limit_enabled=True
    )


class PhysicsConstants:
    """Fundamental physics constants for N-body simulations."""
    
    G = 1.0
    
    SOLAR_MASS_KG = 1.989e30
    
    PARSEC_M = 3.086e16
    
    YEAR_S = 3.156e7
    
    KPC = 1000.0
    
    GYR = 1e9
    
    C = 2.998e8
    
    KM_S_TO_PC_GYR = 1.0227  # km/s to pc/Gyr


class SimulationDefaults:
    """Default parameters for N-body simulations."""
    
    DEFAULT_DT = 0.001  
    MIN_DT = 1e-6  
    MAX_DT = 0.01  
    
    ETA = 0.01  
    EPSILON = 0.01  
    
    DEFAULT_N_PARTICLES = 10000
    MIN_N_PARTICLES = 100
    
    SNAPSHOTS_PER_GYR = 10


__all__ = [
    'settings',
    'PhysicsConstants',
    'SimulationDefaults',
    'BASE_DIR',
    'DATA_DIR',
    'CACHE_DIR',
    'LOG_DIR'
]