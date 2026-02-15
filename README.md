# StellarCore - Globular Cluster N-Body Simulator

Professional-grade N-body simulation tool for globular clusters with Gaia DR3 integration.

## Security Features

- **Rate Limiting**: Protection against DoS attacks
- **Input Validation**: Pydantic schemas for all user inputs
- **Secure Configuration**: Environment-based secrets management
- **OWASP Compliance**: Following industry best practices
- **Audit Logging**: Security event tracking
- **Resource Limits**: Prevention of resource exhaustion

## Quick Start

### 1. Initial Setup
```bash
# Navigate to project directory
cd ~/Projects/StellarCore

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU support (if you have NVIDIA GPU + CUDA)
pip install -r requirements-gpu.txt
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your preferred settings
# CRITICAL: Change SECRET_KEY
nano .env
```

### 3. Test Configuration
```bash
# Verify installation
python scripts/test_setup.py

# Expected output: "ALL CHECKS PASSED"
```

### 4. Launch Application
```bash
# Start the web interface
streamlit run app.py

# Opens browser to http://localhost:8501
```

## Project Structure
```
StellarCore/
├── config/              # Secure configuration management
├── simulator/           # N-body simulation engine
├── data/               # Data loaders and cluster catalog
├── utils/              # Analysis and visualization
├── tests/              # Test suite
├── .streamlit/         # Streamlit configuration
└── app.py              # Main application
```

## Security Checklist

Before deploying to production:

- [ ] Change `SECRET_KEY` in `.env` to cryptographically random value
- [ ] Set `DEBUG_MODE=false`
- [ ] Set `APP_ENV=production`
- [ ] Review and adjust rate limits
- [ ] Enable HTTPS (use reverse proxy like nginx)
- [ ] Set up monitoring and logging
- [ ] Regular security updates
- [ ] Never commit `.env` or `.streamlit/secrets.toml`

## Development Roadmap

After SET 1 is working:
- SET 2: GPU backend and physics engine
- SET 3: Initial conditions generators
- SET 4: N-body integrators
- SET 5: Gaia data loader
- SET 6: Analysis utilities
- SET 7: Main Streamlit application integration
- SET 8: Testing suite

## Troubleshooting

**Configuration Error: SECRET_KEY must be changed**
```bash
# Edit .env and set a secure SECRET_KEY (min 32 characters)
nano .env
```

**ModuleNotFoundError: No module named 'pydantic'**
```bash
# Install dependencies
pip install -r requirements.txt
```

**Port 8501 already in use**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

## License

MIT License - See LICENSE file

## Scientific Background

This tool implements state-of-the-art N-body methods for simulating globular cluster evolution:
- Hermite 4th order integration
- GPU acceleration via CuPy
- Gaia DR3 data integration
- King (1966) and Plummer sphere models
- Professional astronomical visualization

---