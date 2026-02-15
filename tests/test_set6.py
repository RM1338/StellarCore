#!/usr/bin/env python3
"""
SET 6 Comprehensive Test Suite
===============================
Tests visualization and analysis utilities.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_plotly_availability():
    """Test Plotly availability."""
    print("\n" + "=" * 60)
    print("TEST 1: Dependencies")
    print("=" * 60)
    
    from utils.visualization import PLOTLY_AVAILABLE
    
    if PLOTLY_AVAILABLE:
        print("  Plotly: Available")
        print("[PASS] Dependencies available")
    else:
        print("  Plotly: NOT available")
        print("[SKIP] Install with: pip install plotly")
    
    return True


def test_analysis_utilities():
    """Test analysis calculations."""
    print("\n" + "=" * 60)
    print("TEST 2: Analysis Utilities")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_plummer_sphere
    from utils.analysis import (
        calculate_lagrange_radii,
        calculate_half_mass_radius,
        calculate_core_radius,
        calculate_concentration,
        calculate_crossing_time,
        calculate_relaxation_time
    )
    
    # Generate test system
    pos, vel, mass = generate_plummer_sphere(N=500, a=1.0, seed=42)
    print(f"Generated {len(mass)} particles")
    
    # Test Lagrange radii
    lagrange = calculate_lagrange_radii(
        pos, mass,
        np.array([0.1, 0.5, 0.9])
    )
    print(f"\nLagrange radii:")
    print(f"  r_10%: {lagrange[0]:.3f} pc")
    print(f"  r_50%: {lagrange[1]:.3f} pc")
    print(f"  r_90%: {lagrange[2]:.3f} pc")
    
    # Should be monotonically increasing
    assert lagrange[0] < lagrange[1] < lagrange[2]
    print("  Monotonic: OK")
    
    # Test half-mass radius
    r_half = calculate_half_mass_radius(pos, mass)
    print(f"\nHalf-mass radius: {r_half:.3f} pc")
    assert abs(r_half - lagrange[1]) < 1e-10
    
    # Test core radius
    r_core = calculate_core_radius(pos, mass, method="lagrange")
    print(f"Core radius: {r_core:.3f} pc")
    assert r_core > 0
    assert r_core < r_half
    
    # Test concentration
    c = calculate_concentration(pos, mass)
    print(f"Concentration: {c:.3f}")
    assert c > 0
    assert c < 3  # Typical range for globular clusters
    
    # Test time scales
    t_cross = calculate_crossing_time(pos, vel)
    t_relax = calculate_relaxation_time(len(mass), t_cross)
    print(f"\nCrossing time: {t_cross:.3f}")
    print(f"Relaxation time: {t_relax:.3f}")
    assert t_cross > 0
    assert t_relax > t_cross  # Relaxation should be longer
    
    print("\n[PASS] Analysis utilities work correctly")
    return True


def test_system_properties():
    """Test comprehensive system property calculation."""
    print("\n" + "=" * 60)
    print("TEST 3: System Properties")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_king_model
    from utils.analysis import calculate_system_properties
    
    # Generate King model
    pos, vel, mass = generate_king_model(N=500, W0=6.0, seed=42)
    
    # Calculate properties
    props = calculate_system_properties(pos, vel, mass)
    
    print(f"\nSystem properties calculated:")
    print(f"  N: {props['N']}")
    print(f"  Total mass: {props['total_mass']:.6f}")
    print(f"  Energy: {props['total_energy']:.6f}")
    print(f"  Virial Q: {props['virial_ratio']:.6f}")
    print(f"  r_core: {props['r_core']:.3f} pc")
    print(f"  r_half: {props['r_50']:.3f} pc")
    print(f"  Concentration: {props['concentration']:.3f}")
    
    # Validate properties
    assert props['N'] == 500
    assert abs(props['total_mass'] - 1.0) < 1e-10
    assert props['kinetic_energy'] > 0
    assert props['potential_energy'] < 0
    assert props['r_core'] < props['r_50']
    assert props['r_50'] < props['r_90']
    
    print("\n[PASS] System properties calculated correctly")
    return True


def test_visualization_3d():
    """Test 3D visualization."""
    print("\n" + "=" * 60)
    print("TEST 4: 3D Visualization")
    print("=" * 60)
    
    from utils.visualization import PLOTLY_AVAILABLE
    
    if not PLOTLY_AVAILABLE:
        print("[SKIP] Plotly not available")
        return True
    
    from utils.visualization import plot_particles_3d
    from simulator.initial_conditions import generate_plummer_sphere
    
    # Generate particles
    pos, vel, mass = generate_plummer_sphere(N=200, seed=42)
    
    # Create 3D plot
    fig = plot_particles_3d(
        pos, mass,
        title="Test 3D Plot"
    )
    
    print("  3D scatter plot created")
    print(f"  Data points: {len(fig.data)}")
    print(f"  Layout configured: {fig.layout.title.text}")
    
    assert len(fig.data) > 0
    assert fig.layout.title.text == "Test 3D Plot"
    
    # Test with colors
    colors = np.linalg.norm(vel, axis=1)
    fig_colored = plot_particles_3d(
        pos, mass, colors=colors,
        title="Colored by Velocity"
    )
    
    print("  Colored plot created")
    
    print("\n[PASS] 3D visualization works")
    return True


def test_density_profile():
    """Test density profile calculation and plotting."""
    print("\n" + "=" * 60)
    print("TEST 5: Density Profile")
    print("=" * 60)
    
    from utils.visualization import PLOTLY_AVAILABLE
    
    if not PLOTLY_AVAILABLE:
        print("[SKIP] Plotly not available")
        return True
    
    from utils.visualization import (
        calculate_density_profile,
        plot_density_profile
    )
    from simulator.initial_conditions import generate_plummer_sphere
    
    # Generate system
    pos, vel, mass = generate_plummer_sphere(N=1000, a=1.0, seed=42)
    
    # Calculate density profile
    radii, densities = calculate_density_profile(pos, mass, bins=30)
    
    print(f"  Calculated {len(radii)} radial bins")
    print(f"  Radius range: {radii.min():.3f} - {radii.max():.3f} pc")
    print(f"  Density range: {densities.min():.2e} - {densities.max():.2e}")
    
    # Check properties
    assert len(radii) == 30
    assert len(densities) == 30
    assert np.all(radii > 0)
    assert np.any(densities > 0)
    assert densities[0] > 0
    
    # Density should generally decrease with radius
    # (though may have fluctuations)
    assert densities[0] > densities[-1]
    
    # Create plot
    fig = plot_density_profile(pos, mass, title="Test Density")
    
    print("  Density profile plot created")
    assert len(fig.data) > 0
    
    print("\n[PASS] Density profile works")
    return True


def test_energy_plot():
    """Test energy evolution plotting."""
    print("\n" + "=" * 60)
    print("TEST 6: Energy Evolution Plot")
    print("=" * 60)
    
    from utils.visualization import PLOTLY_AVAILABLE
    
    if not PLOTLY_AVAILABLE:
        print("[SKIP] Plotly not available")
        return True
    
    from utils.visualization import plot_energy_evolution
    
    # Simulate energy history
    times = np.linspace(0, 1.0, 100)
    ke = 0.5 + 0.01 * np.sin(times * 10)
    pe = -1.0 + 0.02 * np.cos(times * 10)
    total = ke + pe
    
    # Create plot
    fig = plot_energy_evolution(
        times, ke, pe, total,
        title="Test Energy Evolution"
    )
    
    print(f"  Energy plot created")
    print(f"  Subplots: {len(fig.data)} traces")
    
    # Should have traces for KE, PE, Total, and Drift
    assert len(fig.data) >= 4
    
    print("\n[PASS] Energy plotting works")
    return True


def test_velocity_dispersion():
    """Test velocity dispersion profile."""
    print("\n" + "=" * 60)
    print("TEST 7: Velocity Dispersion")
    print("=" * 60)
    
    from utils.visualization import PLOTLY_AVAILABLE
    
    if not PLOTLY_AVAILABLE:
        print("[SKIP] Plotly not available")
        return True
    
    from utils.visualization import (
        calculate_velocity_dispersion_profile,
        plot_velocity_dispersion
    )
    from simulator.initial_conditions import generate_king_model
    
    # Generate system
    pos, vel, mass = generate_king_model(N=500, W0=6.0, seed=42)
    
    # Calculate dispersion
    radii, dispersions = calculate_velocity_dispersion_profile(
        pos, vel, bins=20
    )
    
    print(f"  Calculated {len(radii)} bins")
    print(f"  Dispersion range: {dispersions.min():.3f} - {dispersions.max():.3f} km/s")
    
    assert len(radii) == 20
    assert len(dispersions) == 20
    assert np.all(dispersions >= 0)
    
    # Create plot
    fig = plot_velocity_dispersion(pos, vel)
    
    print("  Dispersion plot created")
    
    print("\n[PASS] Velocity dispersion works")
    return True


def test_integration():
    """Test full analysis pipeline."""
    print("\n" + "=" * 60)
    print("TEST 8: Full Analysis Pipeline")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_king_model
    from simulator.nbody import NBodySimulation
    from utils.analysis import calculate_system_properties
    from utils.visualization import PLOTLY_AVAILABLE
    
    # Generate initial conditions
    print("\nGenerating King model (N=200)...")
    pos, vel, mass = generate_king_model(N=200, W0=6.0, seed=42)
    
    # Run short simulation
    print("Running simulation...")
    sim = NBodySimulation(pos, vel, mass)
    sim.evolve(
        t_end=0.01,
        adaptive=True,
        progress_interval=0
    )
    
    # Analyze final state
    print("\nAnalyzing final state...")
    final_pos = sim.get_positions()
    final_vel = sim.get_velocities()
    final_mass = sim.get_masses()
    
    props = calculate_system_properties(final_pos, final_vel, final_mass)
    
    print(f"  Final r_half: {props['r_50']:.3f} pc")
    print(f"  Energy drift: {sim.energy_drift():.6e}")
    
    # Create visualizations if available
    if PLOTLY_AVAILABLE:
        from utils.visualization import (
            plot_particles_3d,
            plot_density_profile,
            plot_energy_evolution
        )
        
        fig_3d = plot_particles_3d(final_pos, final_mass)
        fig_density = plot_density_profile(final_pos, final_mass)
        
        # Energy evolution
        history = sim.get_energy_history()
        fig_energy = plot_energy_evolution(
            history[:, 0],  # times
            history[:, 1],  # KE
            history[:, 2],  # PE
            history[:, 3]   # Total
        )
        
        print("\n  Created 3 visualizations:")
        print("    - 3D particle distribution")
        print("    - Density profile")
        print("    - Energy evolution")
    
    print("\n[PASS] Full pipeline works")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SET 6 COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_plotly_availability),
        ("Analysis Utilities", test_analysis_utilities),
        ("System Properties", test_system_properties),
        ("3D Visualization", test_visualization_3d),
        ("Density Profile", test_density_profile),
        ("Energy Plot", test_energy_plot),
        ("Velocity Dispersion", test_velocity_dispersion),
        ("Full Pipeline", test_integration)
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
        print("\nALL TESTS PASSED - SET 6 COMPLETE")
        print("\nVisualization and analysis ready!")
        print("Next: SET 7 to wire everything to the webapp")
        return 0
    else:
        print("\nSOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())