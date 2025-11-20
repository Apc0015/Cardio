#!/usr/bin/env python3
"""
CardioFusion Data Setup Script
Checks for existing data in data/raw/ directory

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path
import argparse


def check_data_exists(data_dir="."):
    """
    Check if cardiovascular disease dataset exists locally

    Args:
        data_dir: Directory to check for data (default: current directory)

    Returns:
        bool: True if data exists, False otherwise
    """
    print("="*70)
    print("🩺 CardioFusion Data Setup")
    print("="*70)

    data_path = Path(data_dir)

    print(f"\n📂 Data Configuration:")
    print(f"   Directory: {data_path.absolute()}")

    # Check if data already exists
    csv_files = list(data_path.glob("*.csv"))

    if csv_files:
        print("\n✅ Dataset found!")
        print(f"\n📁 Found {len(csv_files)} CSV file(s) in {data_path.absolute()}:")
        total_size = 0
        for csv_file in csv_files:
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"   - {csv_file.name} ({size_mb:.2f} MB)")
        print(f"\n📊 Total size: {total_size:.2f} MB")
        print("\n💡 You can now run the preprocessing notebook!")
        return True
    else:
        print("\n❌ No dataset found!")
        print(f"\n📋 Please place your cardiovascular disease dataset in:")
        print(f"   {data_path.absolute()}/")
        print("\n📥 Expected file:")
        print("   - cardio_train.csv (or similar cardiovascular disease dataset)")
        print("\n🔗 Dataset Source:")
        print("   Kaggle: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
        print("   Or any compatible cardiovascular disease CSV dataset")
        print("\n📝 Instructions:")
        print("   1. Download the dataset from Kaggle or your source")
        print("   2. Extract the CSV file")
        print(f"   3. Place it in: {data_path.absolute()}/")
        print("   4. Run this script again to verify")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Check for CardioFusion dataset in local directory"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing the dataset (default: data/raw)"
    )

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check for dataset
    success = check_data_exists(args.data_dir)

    if success:
        print("\n✅ Setup complete! You're ready to preprocess the data.")
        sys.exit(0)
    else:
        print("\n⚠️  Setup incomplete. Please add the dataset and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
