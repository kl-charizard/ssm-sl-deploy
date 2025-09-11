# Setup Instructions

This guide provides detailed setup instructions for the Sign Language Detection Framework across different operating systems and environments.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA support)

### Recommended Requirements
- **Python**: 3.9+
- **RAM**: 16GB or higher
- **Storage**: 50GB+ for datasets and models
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **CPU**: Multi-core processor (8+ cores recommended)

## 💻 Platform-Specific Installation

### macOS

#### Prerequisites
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+
brew install python@3.9

# Install system dependencies
brew install cmake libjpeg libpng
```

#### Python Environment Setup
```bash
# Create virtual environment
python3.9 -m venv sign-lang-env
source sign-lang-env/bin/activate

# Clone repository
git clone https://github.com/kl-charizard/ssm-sl-deploy.git
cd ssm-sl-deploy

# Install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Apple Silicon (M1/M2) Specific
```bash
# For Apple Silicon Macs, use MPS acceleration
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Install PyTorch with MPS support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**⚠️ Important for Apple Silicon Users:**
- **Training**: Works well with MPS acceleration
- **Evaluation/Demos**: Use `--device cpu` to avoid compatibility issues
- **Performance**: Set `num_workers: 0` in `config.yaml` for optimal performance

### Windows

#### Prerequisites
1. **Install Python 3.9+** from [python.org](https://www.python.org/downloads/)
2. **Install Visual Studio Build Tools** (for compilation):
   - Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
   - Install "C++ build tools" workload

#### Setup
```cmd
# Create virtual environment
python -m venv sign-lang-env
sign-lang-env\Scripts\activate

# Clone repository
git clone https://github.com/kl-charizard/ssm-sl-deploy.git
cd ssm-sl-deploy

# Install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### CUDA Setup (Optional)
```cmd
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install python3.9 python3.9-venv python3.9-dev
sudo apt install build-essential cmake git

# Install system libraries
sudo apt install libjpeg-dev libpng-dev libopencv-dev
```

#### Python Environment Setup
```bash
# Create virtual environment
python3.9 -m venv sign-lang-env
source sign-lang-env/bin/activate

# Clone repository
git clone https://github.com/kl-charizard/ssm-sl-deploy.git
cd ssm-sl-deploy

# Install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### CUDA Setup (NVIDIA GPUs)
```bash
# Check if CUDA is available
nvidia-smi

# Install CUDA toolkit (if not installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🐳 Docker Setup

### Using Docker
```bash
# Build Docker image
docker build -t sign-language-detection .

# Run container with GPU support
docker run --gpus all -it -v $(pwd):/workspace sign-language-detection
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# Run training
docker-compose exec app python train.py --dataset asl_alphabet
```

## ☁️ Cloud Environment Setup

### Google Colab

**Quick Setup (Recommended):**
```python
# Clone repository and install dependencies
!git clone https://github.com/kl-charizard/ssm-sl-deploy.git
%cd ssm-sl-deploy
!pip install -r requirements.txt

# Download ASL Alphabet dataset
!python scripts/download_datasets.py --dataset asl_alphabet --output datasets/

# Verify installation
!python -c "import torch; print(f'PyTorch: {torch.__version__}')"
!python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**With Google Drive (for persistent storage):**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone to Drive for persistence
!git clone https://github.com/kl-charizard/ssm-sl-deploy.git /content/drive/MyDrive/ssm-sl-deploy
%cd /content/drive/MyDrive/ssm-sl-deploy
!pip install -r requirements.txt

# Download dataset to Drive
!python scripts/download_datasets.py --dataset asl_alphabet --output datasets/
```

**Start Training in Colab:**
```python
# Quick training example
!python train.py --dataset asl_alphabet --model efficientnet_b0 --epochs 10

# Evaluate the trained model
!python evaluate.py --model-path checkpoints/best_model.pth --dataset asl_alphabet

# Run webcam demo (if camera available)
!python demo.py webcam --model checkpoints/best_model.pth
```

**Note**: Colab automatically uses GPU when available, so no device specification needed.

### AWS EC2
```bash
# Launch GPU instance (p3.2xlarge recommended)
# Install NVIDIA drivers and CUDA

# Setup environment
sudo apt update
sudo apt install python3.9 python3.9-venv git

# Clone and setup
git clone https://github.com/kl-charizard/ssm-sl-deploy.git
cd ssm-sl-deploy
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Azure ML
```python
# Create compute instance
# Use provided ML environment or create custom

# Install dependencies
%pip install -r requirements.txt

# Configure paths for Azure ML datasets
```

## 📊 Dataset Setup

### ASL Alphabet Dataset

#### Option 1: Download from Kaggle
```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials
# Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

# Extract to datasets folder
mkdir -p datasets/asl_alphabet
unzip asl_alphabet.zip -d datasets/asl_alphabet/
```

#### Option 2: Manual Setup
1. Create directory structure:
   ```
   datasets/asl_alphabet/
   ├── A/
   ├── B/
   ├── C/
   └── ...
   ```
2. Place images in corresponding class folders

### Custom Dataset Setup
```bash
# Create custom dataset structure
mkdir -p datasets/custom/{class1,class2,class3}

# Update config.yaml
# Add custom dataset configuration
```

## ⚙️ Configuration

### Environment Variables
```bash
# Create .env file
cat << EOF > .env
PYTHONPATH=${PWD}/src
CUDA_VISIBLE_DEVICES=0
WANDB_API_KEY=your_wandb_key_here
TENSORBOARD_LOG_DIR=logs
EOF
```

### Configuration File
```bash
# Copy and customize config
cp config.yaml my_config.yaml

# Edit configuration as needed
nano my_config.yaml
```

## 🧪 Verification

### Test Installation
```bash
# Activate environment
source sign-lang-env/bin/activate  # Linux/macOS
# or
sign-lang-env\Scripts\activate     # Windows

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Test CUDA availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test MPS availability (macOS Apple Silicon)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Run Quick Test
```bash
# Test with dummy data
python -c "
from src.models.model_factory import create_model
model = create_model('efficientnet_b0', num_classes=29)
print(f'Model created successfully: {model.__class__.__name__}')
"

# Test data loading (if dataset available)
python -c "
from src.data.data_loader import create_data_loaders
try:
    loaders = create_data_loaders('asl_alphabet', 'datasets/asl_alphabet')
    print('Data loaders created successfully')
except Exception as e:
    print(f'Data loader test failed: {e}')
"
```

## 🚨 Troubleshooting

### Common Issues

#### Low Accuracy on Apple Silicon (M1/M2 Macs)
**Problem**: Model shows very low accuracy (3-5%) during evaluation
**Solution**: Use CPU device explicitly
```bash
# For evaluation
python evaluate.py --model-path checkpoints/best_model.pth --dataset asl_alphabet --device cpu

# For webcam demo
python demo.py webcam --model checkpoints/best_model.pth --device cpu
```

#### PyTorch 2.6 Compatibility Issues
**Problem**: `UnpicklingError: Weights only load failed` when loading models
**Solution**: Fixed in latest version - update your code:
```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

#### ImportError: No module named 'cv2'
```bash
# Solution
pip install opencv-python
```

#### CUDA out of memory
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # Reduce from 32

# Or use gradient accumulation
```

#### Permission denied (macOS/Linux)
```bash
# Make scripts executable
chmod +x train.py evaluate.py demo.py
```

#### ImportError: DLL load failed (Windows)
```cmd
# Install Visual C++ Redistributable
# Download from Microsoft website
```

#### Slow training on macOS
```bash
# Optimize data loading in config.yaml
system:
  num_workers: 0  # Use single-threaded loading
  pin_memory: false  # Disable for MPS
  mixed_precision: false  # Disable for MPS
```

### Performance Optimization

#### For CPU-only systems:
```yaml
# In config.yaml
system:
  device: "cpu"
  num_workers: 4  # Adjust based on CPU cores
  mixed_precision: false
```

#### For GPU systems:
```yaml
# In config.yaml
system:
  device: "auto"
  num_workers: 8
  mixed_precision: true
  pin_memory: true
```

## 📞 Getting Help

If you encounter issues:

1. **Check system requirements** - Ensure your system meets minimum requirements
2. **Update dependencies** - Run `pip install --upgrade -r requirements.txt`
3. **Check logs** - Look at error messages and logs for clues
4. **Search issues** - Check GitHub issues for similar problems
5. **Create issue** - If problem persists, create a detailed issue report

### Reporting Issues

When reporting issues, please include:
- Operating system and version
- Python version
- PyTorch version
- CUDA version (if applicable)
- Complete error message
- Steps to reproduce

## 🔄 Keeping Up to Date

### Update Framework
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Update configuration (check for new options)
cp config.yaml config_backup.yaml
# Merge any new configuration options
```

### Version Management
```bash
# Check current version
python -c "from src import __version__; print(__version__)"

# View changelog
cat CHANGELOG.md
```

---

**Setup complete!** 🎉 You're ready to start training sign language detection models and deploy them to production!
