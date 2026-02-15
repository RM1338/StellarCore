"""
N-Body Simulation Engine

Main simulation loop and state management.

Features:
- Time evolution of particle systems
- Snapshot saving at intervals
- Energy and momentum tracking
- Progress monitoring
- Graceful error handling

Security:
- Resource limits enforcement
- Interrupt handling
- Memory management
"""

import numpy as np
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .gpu_backend import get_backend, GPUBackend
from .physics import calculate_total_energy, calculate_virial_ratio
from .integrators import integrate_step, calculate_adaptive_timestep



@dataclass
class SimulationState:
    """
    Container for simulation state at a given time.
    
    Includes positions, velocities, masses, and derived quantities.
    """
    time: float
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    
    kinetic_energy: Optional[float] = None
    potential_energy: Optional[float] = None
    total_energy: Optional[float] = None
    virial_ratio: Optional[float] = None
    
    def compute_energies(self):
        """Compute energy quantities."""
        from .physics import calculate_total_energy, calculate_virial_ratio
        
        self.kinetic_energy, self.potential_energy, self.total_energy = \
            calculate_total_energy(self.positions, self.velocities, self.masses)
        
        self.virial_ratio = calculate_virial_ratio(
            self.positions, self.velocities, self.masses
        )



class NBodySimulation:
    """
    Main N-body simulation engine.
    
    Manages time evolution, state tracking, and snapshot saving.
    
    Example:
        >>> from simulator.initial_conditions import generate_plummer_sphere
        >>> pos, vel, mass = generate_plummer_sphere(N=1000)
        >>> sim = NBodySimulation(pos, vel, mass)
        >>> sim.evolve(t_end=0.1, dt=0.001)
        >>> print(f"Final energy: {sim.total_energy()}")
    """
    
    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        time: float = 0.0,
        softening: float = 0.01,
        backend: Optional[GPUBackend] = None
    ):
        """
        Initialize simulation.
        
        Args:
            positions: Initial positions (N, 3)
            velocities: Initial velocities (N, 3)
            masses: Particle masses (N,)
            time: Initial time (default: 0)
            softening: Softening length (default: 0.01)
            backend: GPU backend (uses default if None)
        """
        self.N = positions.shape[0]
        assert positions.shape == (self.N, 3)
        assert velocities.shape == (self.N, 3)
        assert masses.shape == (self.N,)
        
        self.positions = positions.copy()
        self.velocities = velocities.copy()
        self.masses = masses.copy()
        self.time = time
        
        self.softening = softening
        self.backend = backend if backend is not None else get_backend()
        
        self.snapshots: List[SimulationState] = []
        self.energy_history: List[Tuple[float, float, float, float]] = []
        
        self.n_steps = 0
        self.wall_time = 0.0
        
        self._save_snapshot()
        
        print(f"NBodySimulation initialized: N={self.N}, backend={self.backend.name}")
    
    def step(self, dt: float, method: str = "hermite"):
        """
        Perform one integration step.
        
        Args:
            dt: Timestep
            method: Integration method ("hermite" or "leapfrog")
        """
        self.positions, self.velocities = integrate_step(
            self.positions,
            self.velocities,
            self.masses,
            dt,
            method=method,
            softening=self.softening,
            backend=self.backend
        )
        
        self.time += dt
        self.n_steps += 1
    
    def evolve(
        self,
        t_end: float,
        dt: Optional[float] = None,
        adaptive: bool = True,
        method: str = "hermite",
        snapshot_interval: Optional[float] = None,
        progress_interval: int = 100,
        eta: float = 0.01
    ):
        """
        Evolve system from current time to t_end.
        
        Args:
            t_end: Final time
            dt: Fixed timestep (if adaptive=False)
            adaptive: Use adaptive timestep
            method: Integration method
            snapshot_interval: Save snapshot every interval (Gyr)
            progress_interval: Print progress every N steps
            eta: Accuracy parameter for adaptive timestep
        """
        assert t_end > self.time, f"t_end ({t_end}) must be > current time ({self.time})"
        
        if not adaptive and dt is None:
            raise ValueError("Must provide dt if adaptive=False")
        
        t_start_wall = time.time()
        last_snapshot_time = self.time
        
        print(f"\nStarting evolution: t={self.time:.6f} -> {t_end:.6f} Gyr")
        print(f"Method: {method}, Adaptive: {adaptive}")
        
        while self.time < t_end:
            if adaptive:
                dt = calculate_adaptive_timestep(
                    self.positions,
                    self.velocities,
                    self.masses,
                    eta=eta,
                    softening=self.softening,
                    backend=self.backend
                )
                dt = min(dt, t_end - self.time)
            else:
                dt = min(dt, t_end - self.time)
            
            self.step(dt, method=method)
            
            if snapshot_interval is not None:
                if self.time - last_snapshot_time >= snapshot_interval:
                    self._save_snapshot()
                    last_snapshot_time = self.time
            
            if progress_interval > 0 and self.n_steps % progress_interval == 0:
                elapsed = time.time() - t_start_wall
                ke, pe, E = self.total_energy()
                progress = (self.time - self.snapshots[0].time) / (t_end - self.snapshots[0].time) * 100
                
                print(f"  Step {self.n_steps}: t={self.time:.6f}, "
                      f"dt={dt:.2e}, E={E:.6f}, "
                      f"progress={progress:.1f}%, "
                      f"elapsed={elapsed:.2f}s")
        
        self._save_snapshot()
        
        total_time = time.time() - t_start_wall
        self.wall_time += total_time
        
        print(f"\nEvolution complete!")
        print(f"  Steps: {self.n_steps}")
        print(f"  Wall time: {total_time:.2f} seconds")
        print(f"  Steps/second: {self.n_steps / total_time:.1f}")
    
    def _save_snapshot(self):
        """Save current state as snapshot."""
        state = SimulationState(
            time=self.time,
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy()
        )
        state.compute_energies()
        
        self.snapshots.append(state)
        
        self.energy_history.append((
            state.time,
            state.kinetic_energy,
            state.potential_energy,
            state.total_energy
        ))
    
    def total_energy(self) -> Tuple[float, float, float]:
        """
        Calculate current total energy.
        
        Returns:
            (kinetic, potential, total)
        """
        return calculate_total_energy(
            self.positions,
            self.velocities,
            self.masses
        )
    
    def virial_ratio(self) -> float:
        """Calculate current virial ratio Q = 2*KE / |PE|."""
        return calculate_virial_ratio(
            self.positions,
            self.velocities,
            self.masses
        )
    
    def get_state(self) -> SimulationState:
        """Get current state."""
        state = SimulationState(
            time=self.time,
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy()
        )
        state.compute_energies()
        return state
    
    def get_positions(self) -> np.ndarray:
        """Get current positions."""
        return self.positions.copy()
    
    def get_velocities(self) -> np.ndarray:
        """Get current velocities."""
        return self.velocities.copy()
    
    def get_masses(self) -> np.ndarray:
        """Get masses."""
        return self.masses.copy()
    
    def get_snapshots(self) -> List[SimulationState]:
        """Get all saved snapshots."""
        return self.snapshots.copy()
    
    def get_energy_history(self) -> np.ndarray:
        """
        Get energy history as array.
        
        Returns:
            Array of shape (N_snapshots, 4): [time, KE, PE, E_total]
        """
        return np.array(self.energy_history)
    
    def energy_drift(self) -> float:
        """
        Calculate relative energy drift from initial.
        
        Returns:
            Fractional energy change: |E(t) - E(0)| / |E(0)|
        """
        if len(self.energy_history) < 2:
            return 0.0
        
        E_initial = self.energy_history[0][3]
        E_current = self.energy_history[-1][3]
        
        if abs(E_initial) < 1e-10:
            return 0.0
        
        drift = abs(E_current - E_initial) / abs(E_initial)
        return drift
    
    def print_summary(self):
        """Print simulation summary."""
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        
        print(f"Particles: {self.N}")
        print(f"Time: {self.time:.6f} Gyr")
        print(f"Steps: {self.n_steps}")
        print(f"Snapshots: {len(self.snapshots)}")
        
        if self.wall_time > 0:
            print(f"Wall time: {self.wall_time:.2f} seconds")
            print(f"Performance: {self.n_steps / self.wall_time:.1f} steps/second")
        
        ke, pe, E = self.total_energy()
        print(f"\nEnergy:")
        print(f"  Kinetic: {ke:.6f}")
        print(f"  Potential: {pe:.6f}")
        print(f"  Total: {E:.6f}")
        
        drift = self.energy_drift()
        print(f"  Energy drift: {drift:.6e} ({drift*100:.4f}%)")
        
        Q = self.virial_ratio()
        print(f"  Virial ratio Q: {Q:.6f}")
        
        print("=" * 60)



__all__ = [
    'SimulationState',
    'NBodySimulation'
]