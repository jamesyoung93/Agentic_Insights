"""
Dataset Schema Checker - Fixed Version
Checks for common dataset files and displays schema information
"""

import pandas as pd
import sys
from pathlib import Path

def find_dataset():
    """Find dataset in common locations"""
    
    # Common dataset names
    dataset_names = [
        "customers.csv",
        "customer_data.csv",
        "transactions.csv",
        "transaction_data.csv",
        "data.csv"
    ]
    
    # Common locations
    search_paths = [
        Path.cwd() / "data",  # data/ subdirectory
        Path.cwd(),            # current directory
        Path.cwd().parent / "data",  # parent's data/ directory
    ]
    
    print("🔍 Searching for dataset...")
    print(f"   Current directory: {Path.cwd()}")
    print()
    
    # Search all combinations
    for search_dir in search_paths:
        if not search_dir.exists():
            continue
        
        for dataset_name in dataset_names:
            dataset_path = search_dir / dataset_name
            if dataset_path.exists():
                return dataset_path
    
    return None

def analyze_dataset(path: Path):
    """Analyze and display dataset schema"""
    
    print(f"✅ Found dataset at: {path}")
    print()
    
    try:
        # Load dataset
        df = pd.read_csv(path)
        
        print("=" * 80)
        print("DATASET SCHEMA ANALYSIS")
        print("=" * 80)
        print()
        
        # Basic info
        print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print()
        
        # Column information
        print("📋 COLUMNS:")
        print("-" * 80)
        
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            
            # Get unique value info
            n_unique = df[col].nunique()
            
            # Sample values
            if pd.api.types.is_numeric_dtype(df[col]):
                sample = f"[{df[col].min():.2f} to {df[col].max():.2f}]"
            else:
                samples = df[col].dropna().unique()[:3]
                sample = f"[{', '.join(str(s) for s in samples)}...]" if len(samples) > 0 else "[empty]"
            
            print(f"  • {col:25} | {str(dtype):15} | {n_unique:6,} unique | "
                  f"{null_count:5,} nulls ({null_pct:5.1f}%) | {sample}")
        
        print()
        print("-" * 80)
        print()
        
        # Data types summary
        print("📊 DATA TYPES SUMMARY:")
        print(f"   Numeric:     {df.select_dtypes(include=['int64', 'float64']).shape[1]} columns")
        print(f"   Categorical: {df.select_dtypes(include=['object']).shape[1]} columns")
        print(f"   Boolean:     {df.select_dtypes(include=['bool']).shape[1]} columns")
        print(f"   Datetime:    {df.select_dtypes(include=['datetime64']).shape[1]} columns")
        print()
        
        # Sample rows
        print("📄 SAMPLE ROWS (first 5):")
        print("-" * 80)
        print(df.head().to_string())
        print()
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    # Check if path provided as argument
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
        
        if not dataset_path.exists():
            print(f"❌ File not found: {dataset_path}")
            print(f"   Looking in: {Path.cwd()}")
            sys.exit(1)
    else:
        # Auto-search for dataset
        dataset_path = find_dataset()
        
        if dataset_path is None:
            print("❌ Could not find dataset in standard locations")
            print()
            print("Searched in:")
            print(f"  • {Path.cwd() / 'data'}")
            print(f"  • {Path.cwd()}")
            print()
            print("💡 Usage:")
            print(f"   python {Path(__file__).name} <path_to_dataset.csv>")
            print()
            print("   Example:")
            print(f"   python {Path(__file__).name} data/customers.csv")
            sys.exit(1)
    
    # Analyze the dataset
    success = analyze_dataset(dataset_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
