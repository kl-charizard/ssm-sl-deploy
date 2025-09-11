#!/usr/bin/env python3
"""
Dataset Download Script for Sign Language Detection Framework

This script helps download and organize various sign language datasets.
Currently supports:
- ASL Alphabet Dataset from Kaggle
- WLASL Dataset (future)
- MS-ASL Dataset (future)

Usage:
    python scripts/download_datasets.py --dataset asl_alphabet --output datasets/
    python scripts/download_datasets.py --dataset asl_alphabet,wlasl --output datasets/
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List
import requests
from tqdm import tqdm


class DatasetDownloader:
    """Handles downloading and organizing sign language datasets."""
    
    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def download_asl_alphabet(self) -> bool:
        """Download ASL Alphabet dataset from Kaggle."""
        print("📥 Downloading ASL Alphabet Dataset...")
        
        # Check if kaggle is installed
        try:
            import kaggle
        except ImportError:
            print("❌ Kaggle API not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            import kaggle
        
        try:
            # Download dataset
            dataset_path = self.output_dir / "asl_alphabet"
            if dataset_path.exists():
                response = input(f"Dataset already exists at {dataset_path}. Overwrite? [y/N]: ")
                if response.lower() != 'y':
                    print("⏭️ Skipping download.")
                    return True
                shutil.rmtree(dataset_path)
            
            print("🔄 Downloading from Kaggle (this may take a few minutes)...")
            
            # Use kaggle API to download
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            # Download to temporary directory
            temp_dir = self.output_dir / "temp_asl"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                api.dataset_download_files(
                    'grassknoted/asl-alphabet',
                    path=str(temp_dir),
                    unzip=True
                )
                
                # Move and organize files
                extracted_path = temp_dir / "asl_alphabet_train"
                if extracted_path.exists():
                    shutil.move(str(extracted_path), str(dataset_path))
                else:
                    # Handle different extraction structures
                    for item in temp_dir.glob("*"):
                        if item.is_dir() and "train" in item.name.lower():
                            shutil.move(str(item), str(dataset_path))
                            break
                
                # Clean up
                shutil.rmtree(temp_dir)
                
                print(f"✅ ASL Alphabet dataset downloaded successfully to {dataset_path}")
                self._verify_asl_alphabet(dataset_path)
                return True
                
            except Exception as e:
                print(f"❌ Error downloading with Kaggle API: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
                
        except Exception as e:
            print(f"❌ Failed to download ASL Alphabet dataset: {e}")
            print("💡 Try downloading manually from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
            return False
    
    def download_wlasl(self) -> bool:
        """Download WLASL dataset (placeholder for future implementation)."""
        print("📥 WLASL Dataset download...")
        print("⚠️ WLASL dataset download is not fully implemented yet.")
        print("📝 Manual download instructions:")
        print("   1. Visit: https://github.com/dxli94/WLASL")
        print("   2. Clone the repository")
        print("   3. Follow their download instructions")
        print("   4. Extract frames using their provided scripts")
        
        # Create placeholder directory
        wlasl_path = self.output_dir / "wlasl"
        wlasl_path.mkdir(exist_ok=True)
        (wlasl_path / "README.md").write_text(
            "# WLASL Dataset\n\n"
            "Please download manually from: https://github.com/dxli94/WLASL\n"
            "This dataset requires ~2TB of storage space.\n"
        )
        return True
    
    def download_ms_asl(self) -> bool:
        """Download MS-ASL dataset (placeholder for future implementation)."""
        print("📥 MS-ASL Dataset download...")
        print("⚠️ MS-ASL dataset download is not fully implemented yet.")
        print("📝 Manual download instructions:")
        print("   1. Visit: https://www.microsoft.com/en-us/research/project/ms-asl/")
        print("   2. Follow their download procedure")
        print("   3. Complete required forms and agreements")
        
        # Create placeholder directory
        ms_asl_path = self.output_dir / "ms_asl"
        ms_asl_path.mkdir(exist_ok=True)
        (ms_asl_path / "README.md").write_text(
            "# MS-ASL Dataset\n\n"
            "Please download manually from: https://www.microsoft.com/en-us/research/project/ms-asl/\n"
            "Requires registration and approval.\n"
        )
        return True
    
    def _verify_asl_alphabet(self, dataset_path: Path):
        """Verify ASL Alphabet dataset structure."""
        expected_classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]
        
        print("🔍 Verifying dataset structure...")
        found_classes = []
        total_images = 0
        
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                found_classes.append(class_dir.name)
                class_images = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png")))
                total_images += class_images
                print(f"   📁 {class_dir.name}: {class_images} images")
        
        print(f"📊 Dataset summary:")
        print(f"   🏷️ Classes found: {len(found_classes)}/{len(expected_classes)}")
        print(f"   🖼️ Total images: {total_images}")
        
        missing = set(expected_classes) - set(found_classes)
        if missing:
            print(f"   ⚠️ Missing classes: {missing}")
        else:
            print(f"   ✅ All expected classes found!")
    
    def download_datasets(self, datasets: List[str]) -> bool:
        """Download specified datasets."""
        success = True
        
        for dataset in datasets:
            dataset = dataset.strip().lower()
            
            if dataset == "asl_alphabet":
                success &= self.download_asl_alphabet()
            elif dataset == "wlasl":
                success &= self.download_wlasl()
            elif dataset == "ms_asl":
                success &= self.download_ms_asl()
            else:
                print(f"❌ Unknown dataset: {dataset}")
                print(f"   Supported datasets: asl_alphabet, wlasl, ms_asl")
                success = False
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Download sign language datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download ASL Alphabet dataset
    python scripts/download_datasets.py --dataset asl_alphabet
    
    # Download multiple datasets
    python scripts/download_datasets.py --dataset asl_alphabet,wlasl --output datasets/
    
    # Download to custom directory
    python scripts/download_datasets.py --dataset asl_alphabet --output /path/to/datasets
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset(s) to download (comma-separated). Options: asl_alphabet, wlasl, ms_asl"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="datasets",
        help="Output directory for datasets (default: datasets/)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if dataset already exists"
    )
    
    args = parser.parse_args()
    
    # Parse dataset list
    datasets = [d.strip() for d in args.dataset.split(",")]
    
    print("🤟 Sign Language Dataset Downloader")
    print("=" * 50)
    print(f"📂 Output directory: {args.output}")
    print(f"📦 Datasets to download: {', '.join(datasets)}")
    print()
    
    # Create downloader
    downloader = DatasetDownloader(args.output)
    
    # Download datasets
    success = downloader.download_datasets(datasets)
    
    if success:
        print("\n🎉 All datasets downloaded successfully!")
        print(f"📁 Datasets are available in: {args.output}")
        print("\n🚀 Next steps:")
        print("   1. Verify dataset structure")
        print("   2. Update config.yaml if needed")
        print("   3. Start training: python train.py")
    else:
        print("\n❌ Some datasets failed to download.")
        print("   Check the error messages above and try manual download if needed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
