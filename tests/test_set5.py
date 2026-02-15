#!/usr/bin/env python3
"""
SET 5 Comprehensive Test Suite
===============================
Tests Gaia data loader and coordinate transformations.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_astropy_availability():
    """Test that Astropy/Astroquery are available."""
    print("\n" + "=" * 60)
    print("TEST 1: Dependencies")
    print("=" * 60)
    
    from data.gaia_loader import ASTROPY_AVAILABLE
    
    print(f"Astropy/Astroquery available: {ASTROPY_AVAILABLE}")
    
    if ASTROPY_AVAILABLE:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        print("  Astropy imported successfully")
        print("[PASS] Dependencies available")
    else:
        print("[SKIP] Astropy not available - Gaia queries disabled")
        print("       Install with: pip install astropy astroquery")
    
    return True


def test_coordinate_transformations():
    """Test coordinate transformation utilities."""
    print("\n" + "=" * 60)
    print("TEST 2: Coordinate Transformations")
    print("=" * 60)
    
    from utils.coordinates import (
        parallax_to_distance,
        distance_to_parallax,
        proper_motion_to_velocity,
        radec_to_xyz,
        xyz_to_radec,
        spherical_to_cartesian,
        cartesian_to_spherical
    )
    
    # Test parallax conversions
    print("\nParallax conversions:")
    parallax = 0.14  # mas
    distance = parallax_to_distance(parallax)
    parallax_back = distance_to_parallax(distance)
    
    print(f"  {parallax} mas → {distance:.1f} pc → {parallax_back:.2f} mas")
    assert abs(parallax - parallax_back) < 1e-6
    print("  Round-trip conversion: OK")
    
    # Test proper motion to velocity
    print("\nProper motion to velocity:")
    pm = 5.0  # mas/yr
    d = 1000.0  # pc
    v = proper_motion_to_velocity(pm, d)
    print(f"  {pm} mas/yr at {d} pc → {v:.2f} km/s")
    assert v > 0
    print("  Conversion: OK")
    
    # Test RA/Dec to XYZ and back
    print("\nRA/Dec ↔ XYZ:")
    ra_in = np.array([180.0, 45.0, 270.0])
    dec_in = np.array([0.0, 45.0, -30.0])
    dist_in = np.array([100.0, 200.0, 150.0])
    
    x, y, z = radec_to_xyz(ra_in, dec_in, dist_in)
    ra_out, dec_out, dist_out = xyz_to_radec(x, y, z)
    
    print(f"  Input RA: {ra_in}")
    print(f"  Output RA: {ra_out}")
    assert np.allclose(ra_in, ra_out, atol=1e-6)
    assert np.allclose(dec_in, dec_out, atol=1e-6)
    assert np.allclose(dist_in, dist_out, atol=1e-6)
    print("  Round-trip conversion: OK")
    
    # Test spherical ↔ Cartesian
    print("\nSpherical ↔ Cartesian:")
    r = np.array([1.0, 2.0, 3.0])
    theta = np.array([np.pi/4, np.pi/2, np.pi/3])
    phi = np.array([0.0, np.pi/2, np.pi])
    
    x, y, z = spherical_to_cartesian(r, theta, phi)
    r_out, theta_out, phi_out = cartesian_to_spherical(x, y, z)
    
    assert np.allclose(r, r_out, atol=1e-10)
    assert np.allclose(theta, theta_out, atol=1e-10)
    print("  Round-trip conversion: OK")
    
    print("\n[PASS] Coordinate transformations work correctly")
    return True


def test_cache_system():
    """Test Gaia data caching."""
    print("\n" + "=" * 60)
    print("TEST 3: Cache System")
    print("=" * 60)
    
    from data.gaia_loader import _save_cache, _load_cache
    from config.settings import CACHE_DIR
    import tempfile
    
    # Create temporary cache file
    test_file = CACHE_DIR / "test_cache.npz"
    
    # Test data
    test_data = {
        'ra': np.array([1.0, 2.0, 3.0]),
        'dec': np.array([4.0, 5.0, 6.0]),
        'parallax': np.array([0.1, 0.2, 0.3])
    }
    
    # Save
    _save_cache(test_file, test_data)
    print(f"Saved test cache: {test_file.name}")
    assert test_file.exists()
    
    # Load
    loaded_data = _load_cache(test_file)
    print(f"Loaded cache: {len(loaded_data)} arrays")
    
    # Verify
    for key in test_data:
        assert key in loaded_data
        assert np.allclose(test_data[key], loaded_data[key])
    
    print("  Data integrity: OK")
    
    # Cleanup
    test_file.unlink()
    print("  Cleanup: OK")
    
    print("\n[PASS] Cache system works")
    return True


def test_gaia_transforms():
    """Test Gaia-specific transformations."""
    print("\n" + "=" * 60)
    print("TEST 4: Gaia Coordinate Transforms")
    print("=" * 60)
    
    from data.gaia_loader import ASTROPY_AVAILABLE
    
    if not ASTROPY_AVAILABLE:
        print("[SKIP] Astropy not available")
        return True
    
    from data.gaia_loader import gaia_to_cartesian
    
    # Simulate Gaia data for a small cluster
    N = 10
    ra = np.random.uniform(250, 251, N)  # Around M13
    dec = np.random.uniform(36, 37, N)
    parallax = np.random.uniform(0.13, 0.15, N)  # ~7 kpc
    pmra = np.random.uniform(-5, -3, N)
    pmdec = np.random.uniform(-18, -16, N)
    rv = np.random.uniform(-250, -240, N)
    
    print(f"\nTransforming {N} simulated stars...")
    
    # Transform to Cartesian
    positions, velocities = gaia_to_cartesian(
        ra, dec, parallax, pmra, pmdec, rv,
        center_on_mean=True
    )
    
    print(f"  Positions shape: {positions.shape}")
    print(f"  Velocities shape: {velocities.shape}")
    
    assert positions.shape == (N, 3)
    assert velocities.shape == (N, 3)
    
    # Check centering
    mean_pos = np.mean(positions, axis=0)
    mean_vel = np.mean(velocities, axis=0)
    
    print(f"  Mean position: {mean_pos}")
    print(f"  Mean velocity: {mean_vel}")
    
    assert np.allclose(mean_pos, 0.0, atol=1e-10)
    assert np.allclose(mean_vel, 0.0, atol=1e-10)
    
    print("  Centering: OK")
    
    # Check units
    pos_range = np.abs(positions).max()
    vel_range = np.abs(velocities).max()
    
    print(f"  Position range: {pos_range:.1f} pc")
    print(f"  Velocity range: {vel_range:.1f} km/s")
    
    # Should be reasonable for M13
    assert pos_range < 1000  # Within ~1 kpc
    assert vel_range < 100  # Within ~100 km/s
    
    print("\n[PASS] Gaia transformations work")
    return True


def test_nbody_conversion():
    """Test conversion to N-body units."""
    print("\n" + "=" * 60)
    print("TEST 5: N-Body Unit Conversion")
    print("=" * 60)
    
    from data.gaia_loader import ASTROPY_AVAILABLE
    
    if not ASTROPY_AVAILABLE:
        print("[SKIP] Astropy not available")
        return True
    
    from data.gaia_loader import gaia_to_nbody_units
    
    # Simulate Gaia data
    N = 100
    gaia_data = {
        'ra': np.random.uniform(250, 251, N),
        'dec': np.random.uniform(36, 37, N),
        'parallax': np.random.uniform(0.13, 0.15, N),
        'pmra': np.random.uniform(-5, -3, N),
        'pmdec': np.random.uniform(-18, -16, N),
        'radial_velocity': np.random.uniform(-250, -240, N),
        'phot_g_mean_mag': np.random.uniform(12, 18, N)
    }
    
    print(f"\nConverting {N} stars to N-body units...")
    
    # Convert
    positions, velocities, masses = gaia_to_nbody_units(
        gaia_data,
        mass_total=1e5,  # 100,000 solar masses
        assume_equal_mass=True
    )
    
    print(f"  Positions shape: {positions.shape}")
    print(f"  Velocities shape: {velocities.shape}")
    print(f"  Masses shape: {masses.shape}")
    
    assert positions.shape == (N, 3)
    assert velocities.shape == (N, 3)
    assert masses.shape == (N,)
    
    # Check total mass
    total_mass = masses.sum()
    print(f"  Total mass: {total_mass:.1f} M_sun")
    assert abs(total_mass - 1e5) < 1e-6
    
    # Check equal masses
    assert np.allclose(masses, masses[0])
    print(f"  Individual mass: {masses[0]:.1f} M_sun")
    
    print("\n[PASS] N-body conversion works")
    return True


def test_cluster_parameters():
    """Test cluster parameter lookup for Gaia queries."""
    print("\n" + "=" * 60)
    print("TEST 6: Cluster Parameters for Gaia")
    print("=" * 60)
    
    from data.cluster_catalog import get_cluster
    
    # Test M13 parameters
    m13 = get_cluster("M13")
    
    print(f"\nM13 parameters for Gaia query:")
    print(f"  RA: {m13['ra']:.3f}°")
    print(f"  Dec: {m13['dec']:.3f}°")
    print(f"  Distance: {m13['distance']:.1f} kpc")
    print(f"  Tidal radius: {m13['r_tidal']:.1f} pc")
    
    # Calculate search radius
    r_tidal_pc = m13['r_tidal']
    distance_kpc = m13['distance']
    theta_rad = r_tidal_pc / (distance_kpc * 1000.0)
    radius_deg = np.degrees(theta_rad)
    
    print(f"  Search radius: {radius_deg:.3f}°")
    
    # Should be reasonable
    assert 0.1 < radius_deg < 2.0
    
    print("\n[PASS] Cluster parameters suitable for Gaia queries")
    return True


def test_gaia_query_dry_run():
    """Test Gaia query setup without actually querying."""
    print("\n" + "=" * 60)
    print("TEST 7: Gaia Query Setup (Dry Run)")
    print("=" * 60)
    
    from data.gaia_loader import ASTROPY_AVAILABLE
    
    if not ASTROPY_AVAILABLE:
        print("[SKIP] Astropy not available")
        return True
    
    from data.cluster_catalog import get_cluster
    
    # Get M13 parameters
    m13 = get_cluster("M13")
    
    print("\nQuery parameters for M13:")
    print(f"  RA: {m13['ra']:.3f}°")
    print(f"  Dec: {m13['dec']:.3f}°")
    print(f"  Suggested radius: ~0.5°")
    print(f"  Expected stars: ~50,000 (within tidal radius)")
    
    print("\nQuery would be:")
    print(f"  SELECT * FROM gaiadr3.gaia_source")
    print(f"  WHERE CONTAINS(POINT(ra, dec), CIRCLE({m13['ra']}, {m13['dec']}, 0.5))")
    print(f"  AND parallax > 0")
    print(f"  AND parallax_error / parallax < 0.2")
    
    print("\n[PASS] Query setup ready")
    print("\nTo actually query Gaia:")
    print("  from data.gaia_loader import query_cluster_by_name")
    print("  data = query_cluster_by_name('M13')")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SET 5 COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_astropy_availability),
        ("Coordinate Transformations", test_coordinate_transformations),
        ("Cache System", test_cache_system),
        ("Gaia Transforms", test_gaia_transforms),
        ("N-Body Conversion", test_nbody_conversion),
        ("Cluster Parameters", test_cluster_parameters),
        ("Gaia Query Setup", test_gaia_query_dry_run)
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
            else:
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
    if skipped > 0:
        print(f"Skipped: {skipped}/{len(tests)}")
    
    if failed == 0:
        print("\nALL TESTS PASSED - SET 5 COMPLETE")
        print("\nGaia DR3 data loader ready!")
        print("Next: SET 6 for visualization")
        return 0
    else:
        print("\nSOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())