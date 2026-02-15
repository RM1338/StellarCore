"""
Security Utilities

Rate limiting, input validation, and security helpers.

Implements OWASP best practices:
- Rate limiting (DoS prevention)
- Input sanitization
- Resource limits enforcement
- Audit logging
"""

import time
import functools
from typing import Callable, Dict, Any, Optional
from collections import defaultdict
import threading



class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    Tracks requests per key (e.g., IP address, user ID) and enforces limits.
    Thread-safe for concurrent access.
    """
    
    def __init__(
        self,
        max_per_minute: int = 30,
        max_per_hour: int = 500
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_per_minute: Maximum requests per minute
            max_per_hour: Maximum requests per hour
        """
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        
        self._minute_requests: Dict[str, list] = defaultdict(list)
        self._hour_requests: Dict[str, list] = defaultdict(list)
        
        self._lock = threading.Lock()
    
    def is_allowed(self, key: str) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed.
        
        Args:
            key: Identifier (IP address, user ID, etc.)
            
        Returns:
            (allowed, reason) - (True, None) if allowed, (False, reason) if blocked
        """
        with self._lock:
            now = time.time()
            
            self._cleanup(key, now)
            
            minute_ago = now - 60
            recent_minute = [t for t in self._minute_requests[key] if t > minute_ago]
            
            if len(recent_minute) >= self.max_per_minute:
                return False, f"Rate limit exceeded: {self.max_per_minute} requests per minute"
            
            hour_ago = now - 3600
            recent_hour = [t for t in self._hour_requests[key] if t > hour_ago]
            
            if len(recent_hour) >= self.max_per_hour:
                return False, f"Rate limit exceeded: {self.max_per_hour} requests per hour"
            
            self._minute_requests[key].append(now)
            self._hour_requests[key].append(now)
            
            return True, None
    
    def _cleanup(self, key: str, now: float):
        """Remove timestamps older than 1 hour."""
        hour_ago = now - 3600
        self._minute_requests[key] = [t for t in self._minute_requests[key] if t > hour_ago]
        self._hour_requests[key] = [t for t in self._hour_requests[key] if t > hour_ago]
    
    def reset(self, key: Optional[str] = None):
        """
        Reset rate limit counters.
        
        Args:
            key: Specific key to reset, or None to reset all
        """
        with self._lock:
            if key is None:
                self._minute_requests.clear()
                self._hour_requests.clear()
            else:
                self._minute_requests.pop(key, None)
                self._hour_requests.pop(key, None)


_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter."""
    global _global_limiter
    if _global_limiter is None:
        from config.settings import settings
        _global_limiter = RateLimiter(
            max_per_minute=settings.rate_limit_per_minute,
            max_per_hour=settings.rate_limit_per_hour
        )
    return _global_limiter


def rate_limit(key_func: Optional[Callable] = None):
    """
    Decorator to apply rate limiting to functions.
    
    Usage:
        @rate_limit(lambda: "user_123")
        def expensive_operation():
            pass
    
    Args:
        key_func: Function that returns rate limit key (default: "default")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if key_func is not None:
                key = key_func(*args, **kwargs)
            else:
                key = "default"
            
            limiter = get_rate_limiter()
            allowed, reason = limiter.is_allowed(key)
            
            if not allowed:
                raise RuntimeError(f"Rate limit exceeded: {reason}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator



def validate_particle_count(n: int, max_particles: int = 100000) -> int:
    """
    Validate particle count against resource limits.
    
    Args:
        n: Requested particle count
        max_particles: Maximum allowed (from config)
        
    Returns:
        Validated particle count
        
    Raises:
        ValueError: If count is invalid
    """
    if not isinstance(n, int):
        raise ValueError(f"Particle count must be integer, got {type(n)}")
    
    if n < 1:
        raise ValueError(f"Particle count must be positive, got {n}")
    
    if n > max_particles:
        raise ValueError(
            f"Particle count {n:,} exceeds maximum {max_particles:,}"
        )
    
    return n


def validate_time_range(t_start: float, t_end: float, max_time: float = 20.0) -> tuple[float, float]:
    """
    Validate simulation time range.
    
    Args:
        t_start: Start time (Gyr)
        t_end: End time (Gyr)
        max_time: Maximum allowed time (from config)
        
    Returns:
        (validated_t_start, validated_t_end)
        
    Raises:
        ValueError: If time range is invalid
    """
    if not isinstance(t_start, (int, float)):
        raise ValueError(f"Start time must be numeric, got {type(t_start)}")
    
    if not isinstance(t_end, (int, float)):
        raise ValueError(f"End time must be numeric, got {type(t_end)}")
    
    if t_start < 0:
        raise ValueError(f"Start time must be non-negative, got {t_start}")
    
    if t_end <= t_start:
        raise ValueError(f"End time {t_end} must be greater than start time {t_start}")
    
    duration = t_end - t_start
    if duration > max_time:
        raise ValueError(
            f"Simulation duration {duration:.2f} Gyr exceeds maximum {max_time:.2f} Gyr"
        )
    
    return float(t_start), float(t_end)


def sanitize_string(s: str, max_length: int = 200) -> str:
    """
    Sanitize user input string.
    
    Args:
        s: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValueError: If string is invalid
    """
    if not isinstance(s, str):
        raise ValueError(f"Expected string, got {type(s)}")
    
    s = s.replace('\x00', '')
    
    s = s.strip()
    
    if len(s) > max_length:
        raise ValueError(f"String length {len(s)} exceeds maximum {max_length}")
    
    return s



def log_security_event(event_type: str, details: Dict[str, Any]):
    """
    Log security-relevant event.
    
    Args:
        event_type: Event type (e.g., "rate_limit_exceeded")
        details: Event details
    """
    # TODO: Implement proper audit logging in SET 8
    # For now, just print
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[SECURITY] {timestamp} - {event_type}: {details}")



__all__ = [
    'RateLimiter',
    'get_rate_limiter',
    'rate_limit',
    'validate_particle_count',
    'validate_time_range',
    'sanitize_string',
    'log_security_event'
]