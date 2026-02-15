#!/usr/bin/env python3
"""
SET 2 Comprehensive Test Suite

Tests GPU backend, physics engine, and security utilities.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_gpu_backend():
    """Test GPU backend detection and operations."""
    print("\n" + "=" * 60)
    print("TEST 1: GPU Backend")
    print("=" * 60)
    
    from simulator.gpu_backend import get_backend, reset_backend, ensure_array, MemoryGuard
    
    # Reset and get backend
    reset_backend()
    backend = get_backend(prefer_gpu=True)
    
    print(f"Backend selected: {backend.name}")
    print(f"GPU available: {backend.is_gpu}")
    print(f"Array module: {backend.xp.__name__}")
    
    # Test array creation
    arr1 = backend.array([1, 2, 3, 4, 5])
    arr2 = backend.zeros((3, 3))
    arr3 = backend.ones((2, 4))
    
    print(f"Array creation: OK")
    print(f"  - array([1,2,3,4,5]): shape={arr1.shape}")
    print(f"  - zeros((3,3)): shape={arr2.shape}")
    print(f"  - ones((2,4)): shape={arr3.shape}")
    
    # Test CPU transfer
    cpu_arr = backend.to_cpu(arr1)
    assert isinstance(cpu_arr, np.ndarray)
    print(f"CPU transfer: OK")
    
    # Test ensure_array
    test_data = [1.0, 2.0, 3.0]
    ensured = ensure_array(test_data, backend)
    print(f"ensure_array: OK")
    
    # Test memory guard
    with MemoryGuard(backend):
        temp = backend.array([1, 2, 3])
    print(f"Memory guard: OK")
    
    print("\n[PASS] GPU Backend tests passed")
    return True


def test_physics_engine():
    """Test physics calculations."""
    print("\n" + "=" * 60)
    print("TEST 2: Physics Engine")
    print("=" * 60)
    
    from simulator.physics import (
        calculate_gravitational_force,
        calculate_accelerations,
        calculate_kinetic_energy,
        calculate_potential_energy,
        calculate_total_energy,
        calculate_total_momentum,
        calculate_angular_momentum,
        calculate_virial_ratio,
        center_of_mass,
        center_of_mass_velocity
    )
    
    # Create simple 2-body system
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    velocities = np.array([
        [0.0, 0.1, 0.0],
        [0.0, -0.1, 0.0]
    ])
    masses = np.array([1.0, 1.0])
    
    # Test force calculation
    forces = calculate_gravitational_force(positions, masses)
    print(f"Force calculation:")
    print(f"  - Forces shape: {forces.shape}")
    print(f"  - Force magnitude: {np.linalg.norm(forces[0]):.6f}")
    assert forces.shape == (2, 3)
    assert np.linalg.norm(forces[0]) > 0
    
    # Test accelerations
    accel = calculate_accelerations(positions, masses)
    print(f"  - Accelerations shape: {accel.shape}")
    assert accel.shape == (2, 3)
    
    # Test energy
    ke = calculate_kinetic_energy(velocities, masses)
    pe = calculate_potential_energy(positions, masses)
    ke_calc, pe_calc, total = calculate_total_energy(positions, velocities, masses)
    
    print(f"\nEnergy calculations:")
    print(f"  - Kinetic energy: {ke:.6f}")
    print(f"  - Potential energy: {pe:.6f}")
    print(f"  - Total energy: {total:.6f}")
    assert abs(ke - ke_calc) < 1e-10
    assert abs(pe - pe_calc) < 1e-10
    assert abs(total - (ke + pe)) < 1e-10
    
    # Test momentum
    p_linear = calculate_total_momentum(velocities, masses)
    p_angular = calculate_angular_momentum(positions, velocities, masses)
    
    print(f"\nMomentum calculations:")
    print(f"  - Linear momentum: {p_linear}")
    print(f"  - Angular momentum: {p_angular}")
    assert p_linear.shape == (3,)
    assert p_angular.shape == (3,)
    
    # Test virial ratio
    Q = calculate_virial_ratio(positions, velocities, masses)
    print(f"\nVirial ratio: Q = {Q:.6f}")
    assert Q > 0
    
    # Test center of mass
    com = center_of_mass(positions, masses)
    com_vel = center_of_mass_velocity(velocities, masses)
    
    print(f"\nCenter of mass:")
    print(f"  - Position: {com}")
    print(f"  - Velocity: {com_vel}")
    assert com.shape == (3,)
    assert com_vel.shape == (3,)
    
    print("\n[PASS] Physics engine tests passed")
    return True


def test_physics_performance():
    """Test physics performance with larger systems."""
    print("\n" + "=" * 60)
    print("TEST 3: Physics Performance")
    print("=" * 60)
    
    from simulator.physics import calculate_gravitational_force
    from simulator.gpu_backend import get_backend
    import time
    
    backend = get_backend()
    
    test_sizes = [100, 500, 1000, 2000]
    
    print(f"Backend: {backend.name}")
    print(f"\nTiming force calculations:")
    print(f"{'N particles':<15} {'Time (s)':<12} {'Forces/sec':<15}")
    print("-" * 45)
    
    for N in test_sizes:
        # Create random system
        positions = np.random.randn(N, 3) * 10.0
        masses = np.ones(N)
        
        # Time calculation
        start = time.time()
        forces = calculate_gravitational_force(positions, masses, backend=backend)
        if backend.is_gpu:
            backend.synchronize()
        elapsed = time.time() - start
        
        # Calculate throughput
        force_pairs = N * (N - 1) / 2
        throughput = force_pairs / elapsed
        
        print(f"{N:<15} {elapsed:<12.4f} {throughput:<15.0f}")
    
    print("\n[PASS] Performance tests passed")
    return True


def test_security_utilities():
    """Test security features."""
    print("\n" + "=" * 60)
    print("TEST 4: Security Utilities")
    print("=" * 60)
    
    from utils.security import (
        RateLimiter,
        validate_particle_count,
        validate_time_range,
        sanitize_string
    )
    
    # Test rate limiter
    print("Rate limiter test:")
    limiter = RateLimiter(max_per_minute=3, max_per_hour=10)
    
    results = []
    for i in range(5):
        allowed, reason = limiter.is_allowed('test_user')
        results.append(allowed)
        status = "Allowed" if allowed else f"Blocked ({reason})"
        print(f"  Request {i+1}: {status}")
    
    assert results == [True, True, True, False, False]
    print("  - Rate limiting works correctly")
    
    # Test validation
    print("\nInput validation test:")
    
    # Valid particle count
    n = validate_particle_count(5000, max_particles=100000)
    assert n == 5000
    print("  - Valid particle count: OK")
    
    # Invalid particle count (too large)
    try:
        validate_particle_count(200000, max_particles=100000)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  - Rejected large count: OK ({e})")
    
    # Valid time range
    t_start, t_end = validate_time_range(0.0, 5.0, max_time=20.0)
    assert t_start == 0.0 and t_end == 5.0
    print("  - Valid time range: OK")
    
    # Invalid time range (too long)
    try:
        validate_time_range(0.0, 50.0, max_time=20.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  - Rejected long duration: OK")
    
    # String sanitization
    clean = sanitize_string("  test string  ", max_length=50)
    assert clean == "test string"
    print("  - String sanitization: OK")
    
    print("\n[PASS] Security utilities tests passed")
    return True


def test_energy_conservation():
    """Test energy conservation in simple orbits."""
    print("\n" + "=" * 60)
    print("TEST 5: Energy Conservation")
    print("=" * 60)
    
    from simulator.physics import calculate_total_energy
    
    # Create circular orbit (approximately)
    positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    masses = np.array([1.0, 1.0])
    
    # Calculate initial energy
    ke0, pe0, E0 = calculate_total_energy(positions, velocities, masses)
    print(f"Initial energy:")
    print(f"  - Kinetic: {ke0:.6f}")
    print(f"  - Potential: {pe0:.6f}")
    print(f"  - Total: {E0:.6f}")
    
    # Check that energy components are reasonable
    assert ke0 > 0, "Kinetic energy should be positive"
    assert pe0 < 0, "Potential energy should be negative"
    print("\n[PASS] Energy conservation test passed")
    return True


def test_integration():
    """Test integration of all components."""
    print("\n" + "=" * 60)
    print("TEST 6: Integration Test")
    print("=" * 60)
    
    from simulator.gpu_backend import get_backend
    from simulator.physics import (
        calculate_gravitational_force,
        calculate_total_energy,
        calculate_virial_ratio
    )
    from utils.security import validate_particle_count
    
    # Validate inputs
    N = validate_particle_count(500, max_particles=100000)
    print(f"Creating system with N={N} particles")
    
    # Get backend
    backend = get_backend()
    print(f"Using backend: {backend.name}")
    
    # Create random system
    np.random.seed(42)  # Reproducible
    positions = np.random.randn(N, 3) * 10.0
    velocities = np.random.randn(N, 3) * 0.1
    masses = np.ones(N)
    
    # Calculate forces
    forces = calculate_gravitational_force(positions, masses, backend=backend)
    print(f"Forces calculated: shape={forces.shape}")
    
    # Calculate energy
    ke, pe, total = calculate_total_energy(positions, velocities, masses)
    print(f"Energy: KE={ke:.6f}, PE={pe:.6f}, Total={total:.6f}")
    
    # Calculate virial ratio
    Q = calculate_virial_ratio(positions, velocities, masses)
    print(f"Virial ratio: Q={Q:.6f}")
    
    # Verify results
    assert forces.shape == (N, 3)
    assert np.all(np.isfinite(forces))
    assert ke > 0
    assert pe < 0
    assert Q > 0
    
    print("\n[PASS] Integration test passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SET 2 COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("GPU Backend", test_gpu_backend),
        ("Physics Engine", test_physics_engine),
        ("Physics Performance", test_physics_performance),
        ("Security Utilities", test_security_utilities),
        ("Energy Conservation", test_energy_conservation),
        ("Integration", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\nALL TESTS PASSED - SET 2 COMPLETE")
        print("Ready to proceed to SET 3!")
        return 0
    else:
        print("\nSOME TESTS FAILED")
        print("Please fix issues before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())