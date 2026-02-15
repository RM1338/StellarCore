#!/usr/bin/env python3
"""
SET 4 Comprehensive Test Suite
===============================
Tests N-body integrators and simulation engine.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_hermite_integrator():
    """Test Hermite 4th order integrator."""
    print("\n" + "=" * 60)
    print("TEST 1: Hermite Integrator")
    print("=" * 60)
    
    from simulator.integrators import hermite_step
    from simulator.physics import calculate_total_energy
    
    # Simple 2-body circular orbit
    positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    masses = np.array([1.0, 1.0])
    
    # Initial energy
    _, _, E0 = calculate_total_energy(positions, velocities, masses)
    print(f"Initial energy: {E0:.6f}")
    
    # Integrate for 10 steps
    dt = 0.01
    for i in range(10):
        positions, velocities = hermite_step(
            positions, velocities, masses, dt
        )
    
    # Final energy
    _, _, E1 = calculate_total_energy(positions, velocities, masses)
    print(f"Final energy after 10 steps: {E1:.6f}")
    
    # Check energy conservation
    drift = abs(E1 - E0) / abs(E0)
    print(f"Energy drift: {drift:.6e} ({drift*100:.4f}%)")
    
    assert drift < 0.01, f"Energy drift too large: {drift*100:.2f}%"
    
    print("[PASS] Hermite integrator conserves energy")
    return True


def test_leapfrog_integrator():
    """Test Leapfrog integrator."""
    print("\n" + "=" * 60)
    print("TEST 2: Leapfrog Integrator")
    print("=" * 60)
    
    from simulator.integrators import leapfrog_step
    from simulator.physics import calculate_total_energy
    
    # Simple 2-body system
    positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    masses = np.array([1.0, 1.0])
    
    # Initial energy
    _, _, E0 = calculate_total_energy(positions, velocities, masses)
    print(f"Initial energy: {E0:.6f}")
    
    # Integrate
    dt = 0.01
    for i in range(10):
        positions, velocities = leapfrog_step(
            positions, velocities, masses, dt
        )
    
    # Final energy
    _, _, E1 = calculate_total_energy(positions, velocities, masses)
    print(f"Final energy: {E1:.6f}")
    
    drift = abs(E1 - E0) / abs(E0)
    print(f"Energy drift: {drift:.6e}")
    
    # Leapfrog is symplectic but 2nd order, so slightly larger drift OK
    assert drift < 0.05
    
    print("[PASS] Leapfrog integrator works")
    return True


def test_adaptive_timestep():
    """Test adaptive timestep calculation."""
    print("\n" + "=" * 60)
    print("TEST 3: Adaptive Timestep")
    print("=" * 60)
    
    from simulator.integrators import calculate_adaptive_timestep
    from simulator.initial_conditions import generate_plummer_sphere
    
    # Generate system
    pos, vel, mass = generate_plummer_sphere(N=100, seed=42)
    
    # Calculate adaptive timestep
    dt = calculate_adaptive_timestep(
        pos, vel, mass,
        eta=0.01,
        dt_min=1e-6,
        dt_max=0.01
    )
    
    print(f"Adaptive timestep: {dt:.6e}")
    
    # Should be within bounds
    assert 1e-6 <= dt <= 0.01
    assert dt > 0
    
    # Test with different eta
    dt_small = calculate_adaptive_timestep(pos, vel, mass, eta=0.001)
    dt_large = calculate_adaptive_timestep(pos, vel, mass, eta=0.1)
    
    print(f"  eta=0.001: dt={dt_small:.6e}")
    print(f"  eta=0.01:  dt={dt:.6e}")
    print(f"  eta=0.1:   dt={dt_large:.6e}")
    
    # Smaller eta should give smaller timestep
    assert dt_small < dt < dt_large
    
    print("[PASS] Adaptive timestep calculation works")
    return True


def test_nbody_simulation():
    """Test NBodySimulation class."""
    print("\n" + "=" * 60)
    print("TEST 4: NBodySimulation Class")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_plummer_sphere
    from simulator.nbody import NBodySimulation
    
    # Generate initial conditions
    print("\nGenerating N=200 Plummer sphere...")
    pos, vel, mass = generate_plummer_sphere(N=200, seed=42)
    
    # Create simulation
    sim = NBodySimulation(pos, vel, mass)
    
    # Check initialization
    assert sim.N == 200
    assert sim.time == 0.0
    assert sim.n_steps == 0
    assert len(sim.snapshots) == 1  # Initial snapshot
    
    print(f"Simulation initialized: N={sim.N}")
    
    # Get initial energy
    ke0, pe0, E0 = sim.total_energy()
    print(f"Initial energy: {E0:.6f}")
    
    # Single step
    sim.step(dt=0.001, method="hermite")
    assert sim.time == 0.001
    assert sim.n_steps == 1
    
    print(f"Single step: t={sim.time:.6f}")
    
    # Evolve
    print("\nEvolving to t=0.01 Gyr...")
    sim.evolve(
        t_end=0.01,
        adaptive=True,
        method="hermite",
        progress_interval=0  # No progress output
    )
    
    assert sim.time >= 0.01
    print(f"Evolution complete: {sim.n_steps} steps")
    
    # Check energy conservation
    drift = sim.energy_drift()
    print(f"Energy drift: {drift:.6e} ({drift*100:.4f}%)")
    assert drift < 0.01, f"Energy drift too large: {drift*100:.2f}%"
    
    # Check snapshots
    assert len(sim.snapshots) >= 2
    print(f"Snapshots saved: {len(sim.snapshots)}")
    
    print("[PASS] NBodySimulation class works")
    return True


def test_energy_conservation():
    """Test energy conservation over longer integration."""
    print("\n" + "=" * 60)
    print("TEST 5: Energy Conservation")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_plummer_sphere
    from simulator.nbody import NBodySimulation
    
    # Small system for speed
    print("\nGenerating N=100 system...")
    pos, vel, mass = generate_plummer_sphere(N=100, seed=42)
    
    # Create simulation
    sim = NBodySimulation(pos, vel, mass)
    
    # Initial energy
    _, _, E0 = sim.total_energy()
    print(f"Initial energy: {E0:.6f}")
    
    # Evolve for longer
    print("\nEvolving to t=0.05 Gyr...")
    sim.evolve(
        t_end=0.05,
        adaptive=True,
        method="hermite",
        snapshot_interval=0.01,
        progress_interval=0
    )
    
    # Final energy
    _, _, E1 = sim.total_energy()
    print(f"Final energy: {E1:.6f}")
    
    # Check drift
    drift = sim.energy_drift()
    print(f"Energy drift: {drift:.6e} ({drift*100:.4f}%)")
    
    # Should be < 1% for decent integration
    assert drift < 0.01, f"Energy drift too large: {drift*100:.2f}%"
    
    # Check energy history
    history = sim.get_energy_history()
    print(f"Energy history: {len(history)} snapshots")
    
    assert len(history) >= 5
    
    print("[PASS] Energy conservation verified")
    return True


def test_different_methods():
    """Compare Hermite vs Leapfrog."""
    print("\n" + "=" * 60)
    print("TEST 6: Integration Method Comparison")
    print("=" * 60)
    
    from simulator.initial_conditions import generate_plummer_sphere
    from simulator.nbody import NBodySimulation
    
    # Same initial conditions
    pos, vel, mass = generate_plummer_sphere(N=100, seed=123)
    
    # Test Hermite
    print("\nTesting Hermite integrator...")
    sim_hermite = NBodySimulation(pos.copy(), vel.copy(), mass.copy())
    sim_hermite.evolve(
        t_end=0.02,
        adaptive=False,
        dt=0.001,
        method="hermite",
        progress_interval=0
    )
    drift_hermite = sim_hermite.energy_drift()
    print(f"  Hermite energy drift: {drift_hermite:.6e}")
    
    # Test Leapfrog
    print("\nTesting Leapfrog integrator...")
    sim_leapfrog = NBodySimulation(pos.copy(), vel.copy(), mass.copy())
    sim_leapfrog.evolve(
        t_end=0.02,
        adaptive=False,
        dt=0.001,
        method="leapfrog",
        progress_interval=0
    )
    drift_leapfrog = sim_leapfrog.energy_drift()
    print(f"  Leapfrog energy drift: {drift_leapfrog:.6e}")
    
    # Hermite should be more accurate (lower drift)
    # But both should be reasonable
    assert drift_hermite < 0.05
    assert drift_leapfrog < 0.1
    
    print(f"\nComparison:")
    print(f"  Hermite is {drift_leapfrog/drift_hermite:.1f}x more accurate")
    
    print("[PASS] Both integration methods work")
    return True


def test_simulation_state():
    """Test SimulationState class."""
    print("\n" + "=" * 60)
    print("TEST 7: SimulationState")
    print("=" * 60)
    
    from simulator.nbody import SimulationState
    
    # Create state
    pos = np.random.randn(10, 3)
    vel = np.random.randn(10, 3)
    mass = np.ones(10)
    
    state = SimulationState(
        time=1.5,
        positions=pos,
        velocities=vel,
        masses=mass
    )
    
    print(f"State created: t={state.time}")
    
    # Compute energies
    state.compute_energies()
    
    assert state.kinetic_energy is not None
    assert state.potential_energy is not None
    assert state.total_energy is not None
    assert state.virial_ratio is not None
    
    print(f"  KE: {state.kinetic_energy:.6f}")
    print(f"  PE: {state.potential_energy:.6f}")
    print(f"  E:  {state.total_energy:.6f}")
    print(f"  Q:  {state.virial_ratio:.6f}")
    
    print("[PASS] SimulationState works")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SET 4 COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Hermite Integrator", test_hermite_integrator),
        ("Leapfrog Integrator", test_leapfrog_integrator),
        ("Adaptive Timestep", test_adaptive_timestep),
        ("NBodySimulation Class", test_nbody_simulation),
        ("Energy Conservation", test_energy_conservation),
        ("Method Comparison", test_different_methods),
        ("SimulationState", test_simulation_state)
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
        print("\nALL TESTS PASSED - SET 4 COMPLETE")
        print("\nYou now have a working N-body simulator!")
        print("Next: SET 6 for visualization, SET 7 for UI integration")
        return 0
    else:
        print("\nSOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())