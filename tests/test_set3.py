#!/usr/bin/env python3
"""
SET 3 Comprehensive Test Suite
===============================
Tests initial conditions generators and cluster catalog.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_king_model():
    """Test King model generation."""
    print("\n" + "=" * 60)
    print("TEST 1: King Model Generation")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_king_model
    from simulator.physics import calculate_total_energy, calculate_virial_ratio
    
    # Test different concentrations
    W0_values = [3.0, 6.0, 9.0]
    
    for W0 in W0_values:
        print(f"\nTesting W0={W0}")
        pos, vel, mass = generate_king_model(N=500, W0=W0, seed=42)
        
        # Check shapes
        assert pos.shape == (500, 3)
        assert vel.shape == (500, 3)
        assert mass.shape == (500,)
        print(f"  Shapes: OK")
        
        # Check total mass
        total_mass = mass.sum()
        assert abs(total_mass - 1.0) < 1e-10
        print(f"  Total mass: {total_mass:.6f}")
        
        # Check center of mass (should be near zero)
        com = np.sum(mass[:, np.newaxis] * pos, axis=0) / total_mass
        com_dist = np.linalg.norm(com)
        assert com_dist < 1e-10
        print(f"  COM distance from origin: {com_dist:.2e}")
        
        # Check center of mass velocity (should be near zero)
        com_vel = np.sum(mass[:, np.newaxis] * vel, axis=0) / total_mass
        com_vel_mag = np.linalg.norm(com_vel)
        assert com_vel_mag < 1e-10
        print(f"  COM velocity: {com_vel_mag:.2e}")
        
        # Check energy
        ke, pe, total = calculate_total_energy(pos, vel, mass)
        print(f"  Kinetic energy: {ke:.6f}")
        print(f"  Potential energy: {pe:.6f}")
        print(f"  Total energy: {total:.6f}")
        assert ke > 0
        assert pe < 0
        
        # Check virial ratio (should be around 0.5 for equilibrium)
        Q = calculate_virial_ratio(pos, vel, mass)
        print(f"  Virial ratio Q: {Q:.6f}")
        # Note: May not be exactly 0.5 for small N
        
        # Check radial distribution
        radii = np.linalg.norm(pos, axis=1)
        print(f"  Radial extent: {radii.min():.3f} - {radii.max():.3f}")
        print(f"  Median radius: {np.median(radii):.3f}")
    
    print("\n[PASS] King model tests passed")
    return True


def test_plummer_sphere():
    """Test Plummer sphere generation."""
    print("\n" + "=" * 60)
    print("TEST 2: Plummer Sphere Generation")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_plummer_sphere
    from simulator.physics import calculate_total_energy, calculate_virial_ratio
    
    # Test with different scale radii
    scale_radii = [0.5, 1.0, 2.0]
    
    for a in scale_radii:
        print(f"\nTesting scale radius a={a}")
        pos, vel, mass = generate_plummer_sphere(N=500, a=a, seed=42)
        
        # Check shapes
        assert pos.shape == (500, 3)
        assert vel.shape == (500, 3)
        assert mass.shape == (500,)
        print(f"  Shapes: OK")
        
        # Check total mass
        total_mass = mass.sum()
        assert abs(total_mass - 1.0) < 1e-10
        print(f"  Total mass: {total_mass:.6f}")
        
        # Check center of mass
        com = np.sum(mass[:, np.newaxis] * pos, axis=0) / total_mass
        com_dist = np.linalg.norm(com)
        assert com_dist < 1e-10
        print(f"  COM distance: {com_dist:.2e}")
        
        # Check energy
        ke, pe, total = calculate_total_energy(pos, vel, mass)
        print(f"  Total energy: {total:.6f}")
        assert ke > 0
        assert pe < 0
        
        # Check virial ratio
        Q = calculate_virial_ratio(pos, vel, mass)
        print(f"  Virial ratio Q: {Q:.6f}")
        
        # Check that scale affects distribution
        radii = np.linalg.norm(pos, axis=1)
        median_r = np.median(radii)
        print(f"  Median radius: {median_r:.3f}")
        # Median should scale with 'a'
        if a > 0.5:
            assert median_r > 0.5
    
    print("\n[PASS] Plummer sphere tests passed")
    return True


def test_mass_functions():
    """Test mass function generators."""
    print("\n" + "=" * 60)
    print("TEST 3: Mass Functions")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_kroupa_masses, generate_equal_masses
    
    # Test Kroupa IMF
    print("\nKroupa IMF:")
    masses = generate_kroupa_masses(N=10000, m_min=0.1, m_max=100.0, seed=42)
    
    assert len(masses) == 10000
    assert masses.min() >= 0.1
    assert masses.max() <= 100.0
    print(f"  N: {len(masses)}")
    print(f"  Range: {masses.min():.2f} - {masses.max():.2f} M_sun")
    print(f"  Mean: {masses.mean():.3f} M_sun")
    print(f"  Median: {np.median(masses):.3f} M_sun")
    print(f"  Std dev: {masses.std():.3f} M_sun")
    
    # Check that distribution is skewed (most stars are low mass)
    assert np.median(masses) < masses.mean()
    print(f"  Distribution skewed toward low masses: OK")
    
    # Check mass bins
    low_mass = np.sum(masses < 0.5) / len(masses)
    mid_mass = np.sum((masses >= 0.5) & (masses < 2.0)) / len(masses)
    high_mass = np.sum(masses >= 2.0) / len(masses)
    
    print(f"\n  Mass distribution:")
    print(f"    < 0.5 M_sun: {low_mass*100:.1f}%")
    print(f"    0.5-2 M_sun: {mid_mass*100:.1f}%")
    print(f"    > 2 M_sun: {high_mass*100:.1f}%")
    
    # Most stars should be low mass
    assert low_mass > 0.5
    
    # Test equal masses
    print("\nEqual masses:")
    equal_masses = generate_equal_masses(N=1000, mass_total=500.0)
    
    assert len(equal_masses) == 1000
    assert np.allclose(equal_masses, 0.5)
    assert abs(equal_masses.sum() - 500.0) < 1e-10
    print(f"  All masses equal: {equal_masses[0]:.6f} M_sun")
    print(f"  Total mass: {equal_masses.sum():.6f} M_sun")
    
    print("\n[PASS] Mass function tests passed")
    return True


def test_cluster_catalog():
    """Test cluster catalog functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: Cluster Catalog")
    print("=" * 60)
    
    from data.cluster_catalog import (
        CLUSTER_CATALOG,
        get_cluster,
        list_clusters,
        get_cluster_names_formatted
    )
    
    # Test listing clusters
    clusters = list_clusters()
    print(f"\nAvailable clusters: {clusters}")
    assert len(clusters) == 4
    assert "M13" in clusters
    assert "47 Tucanae" in clusters
    assert "M4" in clusters
    assert "NGC 6397" in clusters
    
    # Test formatted names
    formatted = get_cluster_names_formatted()
    print(f"Formatted names: {formatted}")
    assert len(formatted) == 4
    
    # Test getting each cluster
    for cluster_name in clusters:
        print(f"\nTesting {cluster_name}:")
        cluster = get_cluster(cluster_name)
        
        # Check required fields
        required_fields = [
            'name', 'ngc', 'ra', 'dec', 'distance',
            'r_core', 'r_half', 'r_tidal', 'mass',
            'metallicity', 'age'
        ]
        
        for field in required_fields:
            assert field in cluster, f"Missing field: {field}"
        
        print(f"  Distance: {cluster['distance']} kpc")
        print(f"  Mass: {cluster['mass']:.2e} M_sun")
        print(f"  Age: {cluster['age']} Gyr")
        print(f"  Metallicity: {cluster['metallicity']} [Fe/H]")
        
        # Check physical constraints
        assert cluster['distance'] > 0
        assert cluster['mass'] > 0
        assert cluster['age'] > 0
        assert cluster['r_core'] < cluster['r_half'] < cluster['r_tidal']
    
    # Test NGC number lookup
    print("\nTesting NGC lookup:")
    m13_by_ngc = get_cluster("NGC 6205")
    m13_by_name = get_cluster("M13")
    assert m13_by_ngc['name'] == m13_by_name['name']
    print(f"  NGC 6205 = {m13_by_ngc['name']}: OK")
    
    # Test error handling
    print("\nTesting error handling:")
    try:
        get_cluster("NonExistent")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"  Invalid cluster rejected: OK")
    
    print("\n[PASS] Cluster catalog tests passed")
    return True


def test_physical_properties():
    """Test physical properties of generated systems."""
    print("\n" + "=" * 60)
    print("TEST 5: Physical Properties")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_king_model
    from simulator.physics import (
        calculate_total_momentum,
        calculate_angular_momentum,
        center_of_mass,
        center_of_mass_velocity
    )
    
    # Generate system
    print("\nGenerating King model (N=1000)...")
    pos, vel, mass = generate_king_model(N=1000, W0=6.0, seed=42)
    
    # Test momentum conservation
    p_linear = calculate_total_momentum(vel, mass)
    p_magnitude = np.linalg.norm(p_linear)
    print(f"\nLinear momentum:")
    print(f"  Magnitude: {p_magnitude:.2e}")
    assert p_magnitude < 1e-10, "Linear momentum should be conserved (near zero)"
    print(f"  Conservation: OK")
    
    # Test angular momentum
    L = calculate_angular_momentum(pos, vel, mass)
    L_magnitude = np.linalg.norm(L)
    print(f"\nAngular momentum:")
    print(f"  Magnitude: {L_magnitude:.6f}")
    # Small random component is OK
    
    # Test center of mass
    com = center_of_mass(pos, mass)
    com_dist = np.linalg.norm(com)
    print(f"\nCenter of mass:")
    print(f"  Distance from origin: {com_dist:.2e}")
    assert com_dist < 1e-10
    
    # Test center of mass velocity
    com_vel = center_of_mass_velocity(vel, mass)
    com_vel_mag = np.linalg.norm(com_vel)
    print(f"\nCenter of mass velocity:")
    print(f"  Magnitude: {com_vel_mag:.2e}")
    assert com_vel_mag < 1e-10
    
    # Test radial density profile
    radii = np.linalg.norm(pos, axis=1)
    bins = np.logspace(-2, 2, 20)
    hist, _ = np.histogram(radii, bins=bins)
    
    print(f"\nRadial distribution:")
    print(f"  Min radius: {radii.min():.4f}")
    print(f"  Max radius: {radii.max():.2f}")
    print(f"  Mean radius: {radii.mean():.3f}")
    print(f"  Median radius: {np.median(radii):.3f}")
    
    # Should have central concentration
    inner_count = np.sum(radii < 1.0)
    outer_count = np.sum(radii > 10.0)
    print(f"  Particles r < 1.0: {inner_count} ({inner_count/1000*100:.1f}%)")
    print(f"  Particles r > 10.0: {outer_count} ({outer_count/1000*100:.1f}%)")
    
    print("\n[PASS] Physical properties tests passed")
    return True


def test_reproducibility():
    """Test that random seeds produce reproducible results."""
    print("\n" + "=" * 60)
    print("TEST 6: Reproducibility")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_king_model, generate_plummer_sphere
    
    # Test King model
    print("\nKing model reproducibility:")
    pos1, vel1, mass1 = generate_king_model(N=100, W0=6.0, seed=123)
    pos2, vel2, mass2 = generate_king_model(N=100, W0=6.0, seed=123)
    
    assert np.allclose(pos1, pos2)
    assert np.allclose(vel1, vel2)
    assert np.allclose(mass1, mass2)
    print("  Same seed produces identical results: OK")
    
    # Different seeds should give different results
    pos3, vel3, mass3 = generate_king_model(N=100, W0=6.0, seed=456)
    assert not np.allclose(pos1, pos3)
    print("  Different seed produces different results: OK")
    
    # Test Plummer sphere
    print("\nPlummer sphere reproducibility:")
    pos1, vel1, mass1 = generate_plummer_sphere(N=100, a=1.0, seed=123)
    pos2, vel2, mass2 = generate_plummer_sphere(N=100, a=1.0, seed=123)
    
    assert np.allclose(pos1, pos2)
    assert np.allclose(vel1, vel2)
    assert np.allclose(mass1, mass2)
    print("  Same seed produces identical results: OK")
    
    print("\n[PASS] Reproducibility tests passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SET 3 COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("King Model", test_king_model),
        ("Plummer Sphere", test_plummer_sphere),
        ("Mass Functions", test_mass_functions),
        ("Cluster Catalog", test_cluster_catalog),
        ("Physical Properties", test_physical_properties),
        ("Reproducibility", test_reproducibility)
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
        print("\nALL TESTS PASSED - SET 3 COMPLETE")
        print("Ready to proceed to SET 4!")
        return 0
    else:
        print("\nSOME TESTS FAILED")
        print("Please fix issues before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())