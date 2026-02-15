"""
StellarCore Setup Verification Script

Run this to verify your installation is working correctly.
"""

import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_python_version():
    """Verify Python version is 3.8+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   [OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   [FAIL] Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_package(package_name, display_name=None):
    """Check if a package is installed"""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   [OK] {display_name} ({version})")
        return True
    except ImportError:
        print(f"   [FAIL] {display_name} (not installed)")
        return False

def check_packages():
    """Verify all required packages are installed"""
    print("\nChecking required packages...")
    
    packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('plotly', 'Plotly'),
        ('streamlit', 'Streamlit'),
        ('astropy', 'Astropy'),
        ('astroquery', 'Astroquery'),
        ('pydantic', 'Pydantic'),
        ('dotenv', 'python-dotenv'),
    ]
    
    results = [check_package(pkg, name) for pkg, name in packages]
    return all(results)

def check_gpu_support():
    """Check if GPU support is available"""
    print("\nChecking GPU support...")
    try:
        import cupy as cp
        print(f"   [OK] CuPy installed (version {cp.__version__})")
        
        # Try to access GPU
        try:
            device = cp.cuda.Device()
            print(f"   [OK] GPU detected: {device.attributes['Name']}")
            print(f"   [OK] CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
            return True
        except Exception as e:
            print(f"   [WARN] CuPy installed but no GPU accessible: {e}")
            return False
    except ImportError:
        print("   [WARN] CuPy not installed (GPU acceleration disabled)")
        print("   [INFO] Install with: pip install -r requirements-gpu.txt")
        return False

def check_config():
    """Verify configuration loads correctly"""
    print("\nChecking configuration...")
    try:
        from config.settings import settings
        print(f"   [OK] Configuration loaded")
        print(f"   [OK] Environment: {settings.app_env}")
        print(f"   [OK] Rate limiting: {'enabled' if settings.rate_limit_enabled else 'disabled'}")
        print(f"   [OK] Max particles: {settings.max_particles:,}")
        print(f"   [OK] Debug mode: {settings.debug_mode}")
        return True
    except Exception as e:
        print(f"   [FAIL] Configuration error: {e}")
        print(f"   [INFO] Make sure .env file exists (copy from .env.example)")
        return False

def check_directories():
    """Verify required directories exist"""
    print("\nChecking directories...")
    
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        'config',
        'simulator',
        'data',
        'utils',
        '.streamlit',
        'data/precomputed',
        'logs'
    ]
    
    results = []
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"   [OK] {dir_path}/")
            results.append(True)
        else:
            print(f"   [FAIL] {dir_path}/ (missing)")
            results.append(False)
    
    return all(results)

def check_env_file():
    """Check if .env file exists"""
    print("\nChecking environment configuration...")
    
    env_path = Path(__file__).parent.parent / '.env'
    env_example = Path(__file__).parent.parent / '.env.example'
    
    if env_path.exists():
        print(f"   [OK] .env file exists")
        return True
    else:
        print(f"   [WARN] .env file not found")
        if env_example.exists():
            print(f"   [INFO] Copy .env.example to .env and configure")
            print(f"   [INFO] Command: cp .env.example .env")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("StellarCore Setup Verification")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_packages(),
        check_env_file(),
        check_config(),
        check_directories(),
    ]
    
    # GPU check is optional
    check_gpu_support()
    
    print("\n" + "=" * 60)
    if all(checks):
        print("ALL CHECKS PASSED")
        print("=" * 60)
        print("\nReady to run! Try:")
        print("   streamlit run app.py")
        print("\nOr read the docs:")
        print("   cat README.md")
        return 0
    else:
        print("SOME CHECKS FAILED")
        print("=" * 60)
        print("\nFix the issues above and run this script again.")
        print("\nQuick fixes:")
        print("   - Install packages: pip install -r requirements.txt")
        print("   - Create .env: cp .env.example .env")
        print("   - Edit .env: Change SECRET_KEY to something unique")
        return 1

if __name__ == "__main__":
    sys.exit(main())