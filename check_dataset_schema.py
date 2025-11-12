import pandas as pd
from pathlib import Path

# Try to find and load the dataset
data_paths = [
    Path.cwd() / "data" / "customer_data.csv",
    Path.cwd() / "customer_data.csv",
    Path.cwd() / "transactions_data.csv",
]

print("Searching for dataset...")
for path in data_paths:
    if path.exists():
        print(f"\n✅ Found dataset at: {path}")
        df = pd.read_csv(path)
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumn names:")
        for col in df.columns:
            print(f"  - {col}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nData types:")
        print(df.dtypes)
        break
else:
    print("\n❌ Could not find dataset in standard locations")
    print("Please specify the path to your customer transaction data")
