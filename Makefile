# Makefile for Sign Language Detection Framework

.PHONY: help install train evaluate demo clean test lint format setup-env

# Default target
help:
	@echo "Sign Language Detection Framework"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install      - Install all dependencies"
	@echo "  setup-env    - Set up development environment"
	@echo "  train        - Run basic training example"
	@echo "  evaluate     - Run model evaluation"
	@echo "  demo         - Run webcam demo (requires trained model)"
	@echo "  test         - Run tests"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean up generated files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run in Docker container"
	@echo ""
	@echo "Quick start:"
	@echo "  make install"
	@echo "  make train"
	@echo "  make demo"

# Installation
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

setup-env:
	@echo "Setting up development environment..."
	python -m venv sign-lang-env
	@echo "Virtual environment created. Activate with:"
	@echo "  source sign-lang-env/bin/activate  (Linux/macOS)"
	@echo "  sign-lang-env\\Scripts\\activate     (Windows)"

# Training
train:
	@echo "Running basic training example..."
	python examples/basic_training.py

train-custom:
	@echo "Setting up custom dataset training..."
	python examples/custom_dataset_example.py

train-mobile:
	@echo "Training mobile-optimized model..."
	python examples/mobile_deployment_example.py

# Evaluation
evaluate:
	@if [ ! -f checkpoints/basic_training_example/best_model.pth ]; then \
		echo "No trained model found. Run 'make train' first."; \
		exit 1; \
	fi
	@echo "Evaluating trained model..."
	python evaluate.py --model-path checkpoints/basic_training_example/best_model.pth --dataset asl_alphabet

# Demo applications
demo:
	@if [ ! -f checkpoints/basic_training_example/best_model.pth ]; then \
		echo "No trained model found. Run 'make train' first."; \
		exit 1; \
	fi
	@echo "Starting webcam demo..."
	python demo.py webcam --model checkpoints/basic_training_example/best_model.pth

demo-streamlit:
	@echo "Starting Streamlit web app..."
	python demo.py streamlit

demo-image:
	@if [ ! -f checkpoints/basic_training_example/best_model.pth ]; then \
		echo "No trained model found. Run 'make train' first."; \
		exit 1; \
	fi
	@echo "Usage: make demo-image IMAGE_PATH=/path/to/image.jpg"
	@if [ "$(IMAGE_PATH)" ]; then \
		python demo.py image --model checkpoints/basic_training_example/best_model.pth --image $(IMAGE_PATH); \
	else \
		echo "Please specify IMAGE_PATH. Example:"; \
		echo "  make demo-image IMAGE_PATH=test_image.jpg"; \
	fi

benchmark:
	@if [ ! -f checkpoints/basic_training_example/best_model.pth ]; then \
		echo "No trained model found. Run 'make train' first."; \
		exit 1; \
	fi
	@echo "Benchmarking model performance..."
	python demo.py benchmark --model checkpoints/basic_training_example/best_model.pth

# Development tools
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

lint:
	@echo "Running code linting..."
	flake8 src/ --max-line-length=88 --extend-ignore=E203,W503
	pylint src/ --disable=C0114,C0115,C0116

format:
	@echo "Formatting code..."
	black src/ examples/ *.py
	isort src/ examples/ *.py
	@echo "Code formatted successfully!"

type-check:
	@echo "Running type checking..."
	mypy src/ --ignore-missing-imports

# Cleanup
clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/
	@echo "Cleanup completed!"

clean-all: clean
	@echo "Removing logs, checkpoints, and results..."
	rm -rf logs/
	rm -rf checkpoints/
	rm -rf exports/
	rm -rf examples/results/
	rm -rf examples/mobile_exports/
	rm -rf examples/mobile_deployment/
	@echo "All generated files removed!"

# Docker
docker-build:
	@echo "Building Docker image..."
	docker build -t sign-language-detection .

docker-run:
	@echo "Running Docker container..."
	docker run -it --gpus all -v $(PWD):/workspace sign-language-detection

# Data management
download-dataset:
	@echo "Downloading ASL alphabet dataset..."
	@echo "Please follow the instructions in SETUP.md to download the dataset from Kaggle"
	@echo "Or use the following commands if you have Kaggle API setup:"
	@echo "  kaggle datasets download -d grassknoted/asl-alphabet"
	@echo "  unzip asl-alphabet.zip -d datasets/"

verify-dataset:
	@echo "Verifying dataset structure..."
	@if [ -d "datasets/asl_alphabet" ]; then \
		echo "✅ ASL alphabet dataset found"; \
		echo "Classes: $$(ls datasets/asl_alphabet | wc -l)"; \
		echo "Total images: $$(find datasets/asl_alphabet -name "*.jpg" | wc -l)"; \
	else \
		echo "❌ ASL alphabet dataset not found"; \
		echo "Please download the dataset first (see SETUP.md)"; \
	fi

# Configuration
config-check:
	@echo "Checking configuration..."
	python -c "from src.utils.config import config; print('✅ Configuration loaded successfully')"
	python -c "from src.utils.config import config; print(f'Device: {config.device}')"

# Quick setup for new users
quick-start: install download-dataset train demo
	@echo "🎉 Quick start completed!"
	@echo "Your sign language detection model is ready to use!"

# Development setup
dev-setup: setup-env install
	@echo "Development environment setup completed!"
	@echo "Don't forget to activate your virtual environment:"
	@echo "  source sign-lang-env/bin/activate"

# Export model for deployment
export-model:
	@if [ ! -f checkpoints/basic_training_example/best_model.pth ]; then \
		echo "No trained model found. Run 'make train' first."; \
		exit 1; \
	fi
	@echo "Exporting model for deployment..."
	python -c "
import sys
sys.path.append('src')
from src.deployment.export import export_model
from src.models.model_factory import create_model
from src.utils.config import config
import torch

# Load model
checkpoint = torch.load('checkpoints/basic_training_example/best_model.pth', map_location='cpu')
model = create_model('efficientnet_b0', num_classes=29, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Export
class_names = config.get('dataset.asl_alphabet.classes')
exported = export_model(
    model=model,
    class_names=class_names,
    export_formats=['torchscript', 'onnx'],
    output_dir='exports/production'
)
print('✅ Model exported to exports/production/')
for fmt, path in exported.items():
    print(f'  {fmt}: {path}')
"

# Performance profiling
profile:
	@echo "Profiling model performance..."
	python -m cProfile -o profile_results.prof demo.py benchmark --model checkpoints/basic_training_example/best_model.pth --iterations 10
	@echo "Profile results saved to profile_results.prof"
	@echo "View with: python -m snakeviz profile_results.prof"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "📚 Available documentation:"
	@echo "  README.md - Main project documentation"
	@echo "  SETUP.md - Setup and installation guide"
	@echo "  examples/README.md - Examples documentation"
	@echo "  config.yaml - Configuration reference"

# Version info
version:
	@echo "Sign Language Detection Framework"
	@echo "================================"
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
	@python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
	@python -c "from src.utils.config import config; print(f'Device: {config.device}')"

# CI/CD helpers
ci-install:
	pip install -r requirements.txt
	pip install pytest flake8 black isort mypy

ci-test: ci-install test lint type-check
	@echo "All CI checks passed!"

# Advanced features
quantize-model:
	@if [ ! -f checkpoints/basic_training_example/best_model.pth ]; then \
		echo "No trained model found. Run 'make train' first."; \
		exit 1; \
	fi
	@echo "Applying model quantization..."
	python -c "
import sys
sys.path.append('src')
from src.deployment.quantization import quantize_model
from src.models.model_factory import create_model
import torch

# Load and quantize model
checkpoint = torch.load('checkpoints/basic_training_example/best_model.pth', map_location='cpu')
model = create_model('efficientnet_b0', num_classes=29, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])

quantized_model = quantize_model(model, method='dynamic')
torch.save(quantized_model.state_dict(), 'exports/quantized_model.pth')
print('✅ Quantized model saved to exports/quantized_model.pth')
"
