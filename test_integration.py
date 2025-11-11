"""
System Integration Test
Verifies all components are working together
"""

import sys
from pathlib import Path
import pandas as pd

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_test(name: str, passed: bool, message: str = ""):
    """Print test result"""
    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"  {status} {name}")
    if message:
        print(f"      {message}")

def print_section(title: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

def test_environment():
    """Test basic Python environment"""
    print_section("1. Testing Python Environment")
    
    # Python version
    version_ok = sys.version_info >= (3, 8)
    print_test(
        "Python version >= 3.8",
        version_ok,
        f"Found: Python {sys.version_info.major}.{sys.version_info.minor}"
    )
    
    # Required packages
    packages = {
        'pandas': 'Data manipulation',
        'pathlib': 'File path handling (built-in)',
        'json': 'JSON handling (built-in)',
        'datetime': 'Date/time handling (built-in)'
    }
    
    for package, description in packages.items():
        try:
            __import__(package)
            print_test(f"{package:15} - {description}", True)
        except ImportError:
            print_test(f"{package:15} - {description}", False, "Not installed")
    
    # Optional packages
    optional = ['streamlit', 'openai']
    for package in optional:
        try:
            __import__(package)
            print_test(f"{package:15} (optional)", True)
        except ImportError:
            print_test(
                f"{package:15} (optional)", 
                False, 
                f"Install with: pip install {package}"
            )

def test_file_structure():
    """Test directory structure"""
    print_section("2. Testing File Structure")
    
    base_dir = Path.cwd()
    
    # Check current directory
    print(f"  📁 Working directory: {base_dir}")
    print()
    
    # Required directories
    required_dirs = {
        'data': 'Dataset storage',
        'outputs': 'Generated reports and results (will be created if missing)'
    }
    
    for dir_name, description in required_dirs.items():
        dir_path = base_dir / dir_name
        exists = dir_path.exists()
        
        if not exists and dir_name == 'outputs':
            # Create outputs directory if missing
            try:
                dir_path.mkdir(exist_ok=True)
                exists = True
                print_test(f"{dir_name:15} - {description}", True, "Created")
            except Exception as e:
                print_test(f"{dir_name:15} - {description}", False, str(e))
        else:
            print_test(f"{dir_name:15} - {description}", exists)
    
    # Optional directories
    optional_dirs = {
        'agents': 'Original Kosmos framework agents',
        'knowledge': 'Literature database',
        'prompts': 'Prompt templates'
    }
    
    for dir_name, description in optional_dirs.items():
        dir_path = base_dir / dir_name
        exists = dir_path.exists()
        status = "Found" if exists else "Not found (optional)"
        print_test(f"{dir_name:15} - {description}", True, status)

def test_dataset():
    """Test dataset availability"""
    print_section("3. Testing Dataset")
    
    base_dir = Path.cwd()
    data_dir = base_dir / "data"
    
    if not data_dir.exists():
        print_test("Data directory", False, "data/ directory not found")
        return False
    
    # Look for datasets
    dataset_names = [
        "customers.csv",
        "customer_data.csv", 
        "transactions.csv",
        "transaction_data.csv"
    ]
    
    found_dataset = None
    for name in dataset_names:
        path = data_dir / name
        if path.exists():
            found_dataset = path
            break
    
    if found_dataset:
        print_test("Dataset found", True, f"{found_dataset.name}")
        
        # Try to load it
        try:
            df = pd.read_csv(found_dataset)
            print_test("Dataset loads", True, f"{df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Check for reasonable size
            size_ok = df.shape[0] > 0 and df.shape[1] > 0
            print_test("Dataset has data", size_ok)
            
            # Show columns
            print(f"\n      Columns: {', '.join(df.columns[:5])}")
            if len(df.columns) > 5:
                print(f"      ... and {len(df.columns) - 5} more")
            
            return True
        except Exception as e:
            print_test("Dataset loads", False, str(e))
            return False
    else:
        print_test("Dataset found", False, f"Looking for: {', '.join(dataset_names)}")
        return False

def test_components():
    """Test component files"""
    print_section("4. Testing Component Files")
    
    base_dir = Path.cwd()
    
    # Core new components
    core_files = {
        'auto_enhanced_report.py': 'Enhanced report generator',
        'world_model_builder.py': 'World model management',
        'schema_aware_questions.py': 'Schema-based question generation',
        'safe_analysis_agent.py': 'Safe data analysis'
    }
    
    for filename, description in core_files.items():
        path = base_dir / filename
        exists = path.exists()
        print_test(f"{filename:30} - {description}", exists)
        
        if exists:
            # Try to import as module
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("test_module", path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    print_test(f"  └─ {filename:28} imports", True)
            except Exception as e:
                print_test(f"  └─ {filename:28} imports", False, str(e))
    
    # Original framework files (optional)
    print()
    original_files = {
        'main.py': 'Kosmos orchestrator',
        'config.py': 'Configuration'
    }
    
    for filename, description in original_files.items():
        path = base_dir / filename
        exists = path.exists()
        status = "Found" if exists else "Not found (optional)"
        print_test(f"{filename:30} - {description}", True, status)

def test_integration():
    """Test component integration"""
    print_section("5. Testing Component Integration")
    
    # Test enhanced report generator
    try:
        from auto_enhanced_report import AutoEnhancedReportGenerator
        print_test("Import AutoEnhancedReportGenerator", True)
        
        # Try to instantiate
        generator = AutoEnhancedReportGenerator()
        print_test("Instantiate report generator", True)
        
    except Exception as e:
        print_test("Import AutoEnhancedReportGenerator", False, str(e))
    
    # Test world model
    try:
        from world_model_builder import WorldModel
        print_test("Import WorldModel", True)
        
        # Try to instantiate
        wm = WorldModel()
        wm.set_objective("Test objective")
        print_test("Instantiate world model", True)
        
    except Exception as e:
        print_test("Import WorldModel", False, str(e))
    
    # Test if agents available (optional)
    try:
        from agents.world_model import WorldModel as OriginalWM
        print_test("Original Kosmos agents", True, "Available")
    except ImportError:
        print_test("Original Kosmos agents", True, "Not found (optional)")

def test_streamlit():
    """Test Streamlit app"""
    print_section("6. Testing Streamlit App")
    
    base_dir = Path.cwd()
    
    # Check for streamlit apps
    streamlit_files = [
        'streamlit_integrated.py',
        'streamlit_app_enhanced.py',
        'streamlit_app_FIXED.py',
        'streamlit_app.py'
    ]
    
    found_apps = []
    for filename in streamlit_files:
        path = base_dir / filename
        if path.exists():
            found_apps.append(filename)
            print_test(f"Streamlit app: {filename}", True)
    
    if not found_apps:
        print_test("Streamlit app", False, "No streamlit app found")
    else:
        print(f"\n  💡 To run: streamlit run {found_apps[0]}")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print_test("Streamlit package", True, f"Version {streamlit.__version__}")
    except ImportError:
        print_test("Streamlit package", False, "Install with: pip install streamlit")

def print_summary():
    """Print test summary and recommendations"""
    print_section("Summary & Recommendations")
    
    print(f"{Colors.BOLD}✅ Quick Start:{Colors.END}")
    print()
    print("  1. Verify dataset is in place:")
    print(f"     {Colors.GREEN}python check_dataset_schema_FIXED.py{Colors.END}")
    print()
    print("  2. Run integrated Streamlit app:")
    print(f"     {Colors.GREEN}streamlit run streamlit_integrated.py{Colors.END}")
    print()
    print("  3. Or test components individually:")
    print(f"     {Colors.GREEN}python schema_aware_questions.py data/customers.csv{Colors.END}")
    print(f"     {Colors.GREEN}python safe_analysis_agent.py data/customers.csv{Colors.END}")
    print()
    print(f"{Colors.BOLD}📖 For detailed integration guide:{Colors.END}")
    print(f"     {Colors.BLUE}See INTEGRATION_GUIDE.md{Colors.END}")
    print()

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("=" * 80)
    print("AUTONOMOUS DISCOVERY SYSTEM - INTEGRATION TEST")
    print("=" * 80)
    print(Colors.END)
    
    try:
        test_environment()
        test_file_structure()
        dataset_ok = test_dataset()
        test_components()
        test_integration()
        test_streamlit()
        print_summary()
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Integration test complete!{Colors.END}\n")
        
        if not dataset_ok:
            print(f"{Colors.YELLOW}⚠️  Note: Dataset not found. Some functionality will be limited.{Colors.END}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user{Colors.END}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.RED}❌ Test failed with error: {e}{Colors.END}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
