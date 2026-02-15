"""
Visualization Utilities

Create scientific plots for N-body simulations.

Implements:
- 3D particle visualization (Plotly)
- Density profiles
- Energy evolution plots
- Velocity dispersion profiles

Style:
- Professional scientific aesthetic
- Dark theme consistent with webapp
- Publication-ready quality
"""

import numpy as np
from typing import Optional, Tuple, List
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install: pip install plotly")

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install: pip install matplotlib")



def plot_particles_3d(
    positions: np.ndarray,
    masses: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    title: str = "Particle Distribution",
    size_scale: float = 2.0,
    show_axes: bool = True
):
    """
    Create interactive 3D scatter plot of particles.
    
    Args:
        positions: (N, 3) array of positions
        masses: (N,) array of masses (for sizing points)
        colors: (N,) array for coloring (e.g., velocity, energy)
        title: Plot title
        size_scale: Point size scaling
        show_axes: Show axis lines
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly required. Install: pip install plotly")
    
    N = positions.shape[0]
    
    if masses is not None:
        sizes = np.cbrt(masses / np.mean(masses)) * size_scale
    else:
        sizes = np.ones(N) * size_scale
    
    if colors is None:
        colors = np.zeros(N)
        colorscale = [[0, '#3B82F6'], [1, '#3B82F6']]  # Solid blue
        showscale = False
    else:
        colorscale = 'Viridis'
        showscale = True
    
    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=colorscale,
            showscale=showscale,
            colorbar=dict(title="Value") if showscale else None,
            opacity=0.7,
            line=dict(width=0)
        ),
        text=[f"Particle {i}" for i in range(N)],
        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#E6EDF3')),
        scene=dict(
            xaxis=dict(
                title='X (pc)',
                backgroundcolor='#0A0E14',
                gridcolor='#30363D',
                showbackground=show_axes,
                color='#8B949E'
            ),
            yaxis=dict(
                title='Y (pc)',
                backgroundcolor='#0A0E14',
                gridcolor='#30363D',
                showbackground=show_axes,
                color='#8B949E'
            ),
            zaxis=dict(
                title='Z (pc)',
                backgroundcolor='#0A0E14',
                gridcolor='#30363D',
                showbackground=show_axes,
                color='#8B949E'
            ),
            bgcolor='#0A0E14'
        ),
        paper_bgcolor='#0A0E14',
        plot_bgcolor='#0A0E14',
        font=dict(color='#E6EDF3', family='system-ui'),
        width=800,
        height=700
    )
    
    return fig


def plot_particles_2d(
    positions: np.ndarray,
    projection: str = 'xy',
    masses: Optional[np.ndarray] = None,
    title: Optional[str] = None
):
    """
    Create 2D projection plot of particles.
    
    Args:
        positions: (N, 3) array
        projection: 'xy', 'xz', or 'yz'
        masses: (N,) for point sizing
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly required")
    
    if projection == 'xy':
        x, y = positions[:, 0], positions[:, 1]
        xlabel, ylabel = 'X (pc)', 'Y (pc)'
    elif projection == 'xz':
        x, y = positions[:, 0], positions[:, 2]
        xlabel, ylabel = 'X (pc)', 'Z (pc)'
    elif projection == 'yz':
        x, y = positions[:, 1], positions[:, 2]
        xlabel, ylabel = 'Y (pc)', 'Z (pc)'
    else:
        raise ValueError(f"Invalid projection: {projection}")
    
    if title is None:
        title = f"Particle Distribution ({projection.upper()} plane)"
    
    if masses is not None:
        sizes = np.cbrt(masses / np.mean(masses)) * 3
    else:
        sizes = 3
    
    fig = go.Figure(data=[go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            size=sizes,
            color='#3B82F6',
            opacity=0.6,
            line=dict(width=0)
        )
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        paper_bgcolor='#0A0E14',
        plot_bgcolor='#161B22',
        font=dict(color='#E6EDF3'),
        xaxis=dict(gridcolor='#30363D', zerolinecolor='#30363D'),
        yaxis=dict(gridcolor='#30363D', zerolinecolor='#30363D'),
        width=700,
        height=700
    )
    
    return fig



def calculate_density_profile(
    positions: np.ndarray,
    masses: np.ndarray,
    bins: int = 30,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate radial density profile.
    
    ρ(r) = dM / dV = (M in shell) / (4π r² dr)
    
    Args:
        positions: (N, 3) positions
        masses: (N,) masses
        bins: Number of radial bins
        r_min: Minimum radius (auto if None)
        r_max: Maximum radius (auto if None)
        
    Returns:
        (radii, densities) bin centers and surface densities
    """
    radii = np.linalg.norm(positions, axis=1)
    
    if r_min is None:
        r_min = max(radii.min(), 0.01)  # Avoid r=0
    if r_max is None:
        r_max = radii.max()
    
    bin_edges = np.logspace(np.log10(r_min), np.log10(r_max), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mass_in_bin = np.zeros(bins)
    for i in range(bins):
        mask = (radii >= bin_edges[i]) & (radii < bin_edges[i+1])
        mass_in_bin[i] = masses[mask].sum()
    
    volumes = (4.0/3.0) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    
    densities = mass_in_bin / volumes
    
    return bin_centers, densities


def plot_density_profile(
    positions: np.ndarray,
    masses: np.ndarray,
    title: str = "Radial Density Profile",
    compare_data: Optional[List[Tuple[np.ndarray, np.ndarray, str]]] = None
):
    """
    Plot radial density profile.
    
    Args:
        positions: (N, 3) positions
        masses: (N,) masses
        title: Plot title
        compare_data: List of (radii, densities, label) for comparison
        
    Returns:
        Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly required")
    
    # Calculate density
    radii, densities = calculate_density_profile(positions, masses)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=radii,
        y=densities,
        mode='lines+markers',
        name='Simulation',
        line=dict(color='#3B82F6', width=2),
        marker=dict(size=6)
    ))
    
    if compare_data is not None:
        colors = ['#10B981', '#F59E0B', '#EF4444']
        for i, (r, rho, label) in enumerate(compare_data):
            fig.add_trace(go.Scatter(
                x=r, y=rho,
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Radius (pc)',
        yaxis_title='Density (M☉/pc³)',
        xaxis_type='log',
        yaxis_type='log',
        paper_bgcolor='#0A0E14',
        plot_bgcolor='#161B22',
        font=dict(color='#E6EDF3'),
        xaxis=dict(gridcolor='#30363D'),
        yaxis=dict(gridcolor='#30363D'),
        legend=dict(bgcolor='#161B22', bordercolor='#30363D'),
        width=800,
        height=500
    )
    
    return fig



def plot_energy_evolution(
    times: np.ndarray,
    kinetic: np.ndarray,
    potential: np.ndarray,
    total: np.ndarray,
    title: str = "Energy Evolution"
):
    """
    Plot energy components over time.
    
    Args:
        times: Time array (Gyr)
        kinetic: Kinetic energy array
        potential: Potential energy array
        total: Total energy array
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly required")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Energy Components', 'Relative Energy Drift'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(go.Scatter(
        x=times, y=kinetic,
        name='Kinetic',
        line=dict(color='#10B981', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=times, y=potential,
        name='Potential',
        line=dict(color='#EF4444', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=times, y=total,
        name='Total',
        line=dict(color='#3B82F6', width=3)
    ), row=1, col=1)
    
    E_initial = total[0]
    drift = np.abs(total - E_initial) / np.abs(E_initial) * 100
    
    fig.add_trace(go.Scatter(
        x=times, y=drift,
        name='Drift',
        line=dict(color='#F59E0B', width=2),
        showlegend=False
    ), row=2, col=1)
    
    fig.update_xaxes(title_text="Time (Gyr)", row=2, col=1, gridcolor='#30363D')
    fig.update_yaxes(title_text="Energy", row=1, col=1, gridcolor='#30363D')
    fig.update_yaxes(title_text="Drift (%)", row=2, col=1, gridcolor='#30363D')
    
    fig.update_layout(
        title=title,
        paper_bgcolor='#0A0E14',
        plot_bgcolor='#161B22',
        font=dict(color='#E6EDF3'),
        height=700,
        width=800,
        legend=dict(bgcolor='#161B22', bordercolor='#30363D')
    )
    
    return fig



def calculate_velocity_dispersion_profile(
    positions: np.ndarray,
    velocities: np.ndarray,
    bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate velocity dispersion as function of radius.
    
    σ_v(r) = sqrt(<v²>) in radial bins
    
    Args:
        positions: (N, 3) positions
        velocities: (N, 3) velocities
        bins: Number of radial bins
        
    Returns:
        (radii, dispersions) bin centers and 1D velocity dispersions
    """
    radii = np.linalg.norm(positions, axis=1)
    vel_mag = np.linalg.norm(velocities, axis=1)
    
    bin_edges = np.logspace(np.log10(max(radii.min(), 0.01)), 
                            np.log10(radii.max()), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    dispersions = np.zeros(bins)
    for i in range(bins):
        mask = (radii >= bin_edges[i]) & (radii < bin_edges[i+1])
        if mask.sum() > 0:
            dispersions[i] = np.std(vel_mag[mask])
    
    return bin_centers, dispersions


def plot_velocity_dispersion(
    positions: np.ndarray,
    velocities: np.ndarray,
    title: str = "Velocity Dispersion Profile"
):
    """
    Plot velocity dispersion vs radius.
    
    Args:
        positions: (N, 3) positions
        velocities: (N, 3) velocities
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly required")
    
    radii, dispersions = calculate_velocity_dispersion_profile(
        positions, velocities
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=radii,
        y=dispersions,
        mode='lines+markers',
        line=dict(color='#3B82F6', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Radius (pc)',
        yaxis_title='Velocity Dispersion σ (km/s)',
        xaxis_type='log',
        paper_bgcolor='#0A0E14',
        plot_bgcolor='#161B22',
        font=dict(color='#E6EDF3'),
        xaxis=dict(gridcolor='#30363D'),
        yaxis=dict(gridcolor='#30363D'),
        width=800,
        height=500
    )
    
    return fig



__all__ = [
    'plot_particles_3d',
    'plot_particles_2d',
    'calculate_density_profile',
    'plot_density_profile',
    'plot_energy_evolution',
    'calculate_velocity_dispersion_profile',
    'plot_velocity_dispersion',
    'PLOTLY_AVAILABLE',
    'MATPLOTLIB_AVAILABLE'
]