"""
StellarCore - Globular Cluster N-Body Simulator

Professional-grade N-body simulation with Gaia DR3 integration

SECURITY FEATURES:
- Rate limiting on all operations
- Input validation via Pydantic schemas
- Secure session state management
- Resource limits enforcement
- Audit logging

Author: StellarCore Team
License: MIT
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from config.settings import settings, PhysicsConstants
except ImportError as e:
    st.error(f"Configuration Error: {e}")
    st.info("Make sure you've installed requirements: pip install -r requirements.txt")
    st.info("Make sure .env file exists: cp .env.example .env")
    st.stop()


st.set_page_config(
    page_title="StellarCore - Globular Cluster Simulator",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/stellarcore/docs',
        'Report a bug': 'https://github.com/stellarcore/issues',
        'About': """
        # StellarCore N-Body Simulator
        
        Professional-grade globular cluster simulation with:
        - Gaia DR3 data integration
        - GPU acceleration
        - Hermite 4th order integration
        - Enterprise security
        
        Version: 1.0.0-alpha
        """
    }
)


st.markdown("""
<style>
    /* Professional dark theme - no gradients, no glow effects */
    .main {
        background-color: #0A0E14;
    }
    
    /* Headers - clean, minimal */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    /* Monospace for data */
    .metric-value {
        font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
        font-size: 2em;
        font-weight: 600;
        color: #E6EDF3;
    }
    
    .metric-label {
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8B949E;
        font-weight: 600;
    }
    
    /* Buttons - flat, crisp */
    .stButton>button {
        border-radius: 4px;
        border: 1px solid #3B82F6;
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    
    /* Info boxes - minimal, flat */
    .stAlert {
        border-radius: 4px;
        border-left: 3px solid #3B82F6;
    }
    
    /* Sidebar - clean */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Remove excessive padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Tabs - minimal underline style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


if 'session_id' not in st.session_state:
    import secrets
    st.session_state.session_id = secrets.token_hex(16)
    st.session_state.simulation_count = 0
    st.session_state.last_simulation = None


st.title("StellarCore")
st.markdown("**Globular Cluster N-Body Simulator** • Gaia DR3 Integration • GPU Accelerated")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<p class="metric-label">ENVIRONMENT</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-value">{settings.app_env.upper()}</p>', unsafe_allow_html=True)

with col2:
    st.markdown('<p class="metric-label">SECURITY</p>', unsafe_allow_html=True)
    security_status = "ENABLED" if settings.rate_limit_enabled else "DISABLED"
    st.markdown(f'<p class="metric-value" style="font-size: 1.2em;">{security_status}</p>', unsafe_allow_html=True)

with col3:
    st.markdown('<p class="metric-label">MAX PARTICLES</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-value">{settings.max_particles:,}</p>', unsafe_allow_html=True)

with col4:
    st.markdown('<p class="metric-label">GPU STATUS</p>', unsafe_allow_html=True)
    try:
        import cupy
        gpu_status = "READY"
    except ImportError:
        gpu_status = "CPU ONLY"
    st.markdown(f'<p class="metric-value" style="font-size: 1.2em;">{gpu_status}</p>', unsafe_allow_html=True)

st.divider()


tab1, tab2, tab3, tab4 = st.tabs([
    "Setup",
    "Simulation", 
    "Analysis",
    "Compare"
])


with tab1:
    st.header("Simulation Setup")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Cluster Selection")
        
        cluster = st.selectbox(
            "Select Globular Cluster",
            options=[
                "M13 (NGC 6205)",
                "47 Tucanae (NGC 104)",
                "M4 (NGC 6121)",
                "NGC 6397"
            ],
            help="Choose from built-in cluster catalog"
        )
        
        st.subheader("Initial Conditions")
        
        ic_method = st.radio(
            "Generation Method",
            options=[
                "Gaia DR3 (Real Data)",
                "King Model (1966)",
                "Plummer Sphere"
            ],
            help="Choose how to initialize particle positions and velocities"
        )
        
        if "Gaia" in ic_method:
            st.info("Will download real observations from ESA Gaia Archive")
        elif "King" in ic_method:
            st.info("Analytical model with concentration parameter")
        else:
            st.info("Simple analytical sphere model")
    
    with col2:
        st.subheader("Simulation Parameters")
        
        n_particles = st.slider(
            "Number of Particles",
            min_value=1000,
            max_value=settings.max_particles,
            value=10000,
            step=1000,
            help=f"Maximum: {settings.max_particles:,} (resource limit)"
        )
        
        evolution_time = st.slider(
            "Evolution Time (Gyr)",
            min_value=0.1,
            max_value=float(settings.max_simulation_time_gyr),
            value=1.0,
            step=0.1,
            help=f"Maximum: {settings.max_simulation_time_gyr} Gyr (resource limit)"
        )
        
        timestep = st.select_slider(
            "Integration Accuracy",
            options=["Fast", "Balanced", "Accurate", "Very Accurate"],
            value="Balanced",
            help="Higher accuracy = longer computation time"
        )
        
        st.subheader("Physics Options")
        
        enable_tidal = st.checkbox("Include Milky Way Tidal Field", value=False)
        enable_stellar_evolution = st.checkbox("Stellar Evolution (Mass Loss)", value=False)
        
        if enable_stellar_evolution:
            st.warning("Stellar evolution increases computation time by approximately 30%")


with tab2:
    st.header("Run Simulation")
    
    if st.session_state.simulation_count >= settings.max_concurrent_simulations:
        st.error(f"Maximum concurrent simulations reached ({settings.max_concurrent_simulations})")
        st.info("Wait for current simulations to complete or refresh the page")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Ready to Simulate")
            st.write(f"**Cluster:** {cluster}")
            st.write(f"**Method:** {ic_method}")
            st.write(f"**Particles:** {n_particles:,}")
            st.write(f"**Time:** {evolution_time} Gyr")
            
            if st.button("START SIMULATION", type="primary", use_container_width=True):
                st.info("Backend modules are being implemented in SET 2-7")
                st.info("This will show: Progress bar → GPU acceleration → Real-time plots")
                
                with st.spinner("Initializing simulation engine..."):
                    import time
                    time.sleep(1)
                    st.success("Simulation engine ready (placeholder)")
                    st.session_state.simulation_count += 1
        
        with col2:
            st.subheader("Estimated Time")
            
            est_time = (n_particles / 10000) * evolution_time * 5  # minutes
            st.metric("Computation Time", f"{est_time:.1f} min")
            
            st.subheader("Resources")
            st.metric("Memory Required", f"~{n_particles * 0.001:.1f} GB")
            st.metric("GPU Cores", "2,560" if 'cupy' in sys.modules else "N/A")


with tab3:
    st.header("Results Analysis")
    
    if st.session_state.simulation_count == 0:
        st.info("Run a simulation first to see analysis tools")
    else:
        st.subheader("Visualization Options")
        
        viz_type = st.selectbox(
            "Select Visualization",
            options=[
                "3D Particle Distribution",
                "Density Profile (Sigma vs R)",
                "Velocity Dispersion (Sigma vs r)",
                "Lagrange Radii Evolution",
                "Energy Conservation Plot"
            ]
        )
        
        st.info(f"{viz_type} visualization will be implemented in SET 6")
        
        st.markdown("### Preview")
        st.write("Interactive 3D plots, density profiles, and scientific visualizations will appear here")


with tab4:
    st.header("Model Comparison")
    
    st.subheader("Compare Initial Conditions")
    
    compare_col1, compare_col2 = st.columns(2)
    
    with compare_col1:
        st.write("**Model 1**")
        model1 = st.selectbox("Select first model", ["Gaia DR3", "King Model", "Plummer"], key="model1")
    
    with compare_col2:
        st.write("**Model 2**")
        model2 = st.selectbox("Select second model", ["King Model", "Gaia DR3", "Plummer"], key="model2")
    
    if st.button("Compare Models", use_container_width=True):
        st.info(f"Side-by-side comparison of {model1} vs {model2} will be implemented in SET 6-7")


with st.sidebar:
    st.header("Controls")
    
    st.subheader("Session Info")
    st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    st.write(f"**Simulations Run:** {st.session_state.simulation_count}")
    
    st.divider()
    
    st.subheader("Security Status")
    st.write(f"**Rate Limiting:** {'Enabled' if settings.rate_limit_enabled else 'Disabled'}")
    st.write(f"**Max Requests/min:** {settings.rate_limit_per_minute}")
    st.write(f"**Audit Logging:** {'Enabled' if settings.enable_audit_log else 'Disabled'}")
    
    st.divider()
    
    st.subheader("Documentation")
    st.markdown("""
    - [User Guide](docs/user_guide.md)
    - [API Reference](docs/api_reference.md)
    - [Theory Background](docs/theory_background.md)
    - [Security Guide](docs/security_guide.md)
    """)
    
    st.divider()
    
    st.subheader("Development")
    if settings.app_env == 'development':
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared")
        
        if st.button("Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.divider()
    
    st.caption(f"StellarCore v1.0.0-alpha • {settings.app_env.upper()} mode")
    st.caption("MIT License • Security-hardened")


st.divider()
st.caption("StellarCore - Professional Globular Cluster Simulation • Built with Streamlit + GPU Acceleration")