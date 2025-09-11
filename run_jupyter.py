#!/usr/bin/env python3
"""
Launch Jupyter notebook with proper environment setup for sign language detection framework.

This script:
1. Sets up the Python path for importing framework modules
2. Configures Jupyter with useful extensions
3. Provides notebook templates for common tasks
"""

import sys
import os
import subprocess
from pathlib import Path


def setup_jupyter_environment():
    """Setup Jupyter environment with proper configuration."""
    
    # Add src to Python path
    project_root = Path(__file__).parent
    src_path = str(project_root / 'src')
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Set environment variables
    os.environ['PYTHONPATH'] = src_path
    os.environ['PROJECT_ROOT'] = str(project_root)
    
    print("🔧 Environment configured:")
    print(f"   Project root: {project_root}")
    print(f"   Python path: {src_path}")


def create_notebook_templates():
    """Create notebook templates for common tasks."""
    
    notebooks_dir = Path('notebooks')
    notebooks_dir.mkdir(exist_ok=True)
    
    templates = {
        'training_template.ipynb': create_training_template(),
        'evaluation_template.ipynb': create_evaluation_template(),
        'data_exploration.ipynb': create_data_exploration_template(),
        'model_comparison.ipynb': create_model_comparison_template()
    }
    
    created_templates = []
    for filename, content in templates.items():
        template_path = notebooks_dir / filename
        if not template_path.exists():
            with open(template_path, 'w') as f:
                f.write(content)
            created_templates.append(filename)
    
    if created_templates:
        print(f"📓 Created notebook templates:")
        for template in created_templates:
            print(f"   {template}")
    
    return notebooks_dir


def create_training_template():
    """Create training notebook template."""
    return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sign Language Detection - Training Notebook\\n",
    "\\n",
    "This notebook provides an interactive environment for training sign language detection models.\\n",
    "\\n",
    "## Setup\\n",
    "Run the cells below to set up the environment and load the framework."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup imports and environment\\n",
    "import sys\\n",
    "import os\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Add framework to path\\n",
    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\\n",
    "sys.path.append(str(project_root / 'src'))\\n",
    "\\n",
    "# Framework imports\\n",
    "import torch\\n",
    "import matplotlib.pyplot as plt\\n",
    "import numpy as np\\n",
    "\\n",
    "from src.data.data_loader import create_data_loaders\\n",
    "from src.models.model_factory import create_model\\n",
    "from src.training.trainer import Trainer\\n",
    "from src.utils.config import config\\n",
    "\\n",
    "print(f'PyTorch version: {torch.__version__}')\\n",
    "print(f'Device: {config.device}')\\n",
    "print('✅ Framework loaded successfully!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Configuration\\n",
    "Set up your training configuration here."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training configuration\\n",
    "CONFIG = {\\n",
    "    'dataset_name': 'asl_alphabet',\\n",
    "    'data_dir': 'datasets/asl_alphabet',\\n",
    "    'model_architecture': 'efficientnet_b0',\\n",
    "    'batch_size': 32,\\n",
    "    'epochs': 50,\\n",
    "    'learning_rate': 0.001,\\n",
    "    'experiment_name': 'notebook_training'\\n",
    "}\\n",
    "\\n",
    "print('Configuration:')\\n",
    "for key, value in CONFIG.items():\\n",
    "    print(f'  {key}: {value}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Loading\\n",
    "Load and explore the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\\n",
    "data_loaders = create_data_loaders(\\n",
    "    dataset_name=CONFIG['dataset_name'],\\n",
    "    data_dir=CONFIG['data_dir'],\\n",
    "    batch_size=CONFIG['batch_size']\\n",
    ")\\n",
    "\\n",
    "print(f'Train samples: {len(data_loaders[\"train\"].dataset)}')\\n",
    "print(f'Val samples: {len(data_loaders[\"val\"].dataset)}')\\n",
    "print(f'Test samples: {len(data_loaders[\"test\"].dataset) if \"test\" in data_loaders else 0}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize sample data\\n",
    "def show_batch(data_loader, num_samples=8):\\n",
    "    images, labels = next(iter(data_loader))\\n",
    "    \\n",
    "    fig, axes = plt.subplots(2, 4, figsize=(15, 8))\\n",
    "    axes = axes.ravel()\\n",
    "    \\n",
    "    for i in range(min(num_samples, len(images))):\\n",
    "        img = images[i]\\n",
    "        # Denormalize if needed\\n",
    "        if img.min() < 0:\\n",
    "            mean = torch.tensor([0.485, 0.456, 0.406])\\n",
    "            std = torch.tensor([0.229, 0.224, 0.225])\\n",
    "            img = img * std[:, None, None] + mean[:, None, None]\\n",
    "        \\n",
    "        img = torch.clamp(img, 0, 1)\\n",
    "        img_np = img.permute(1, 2, 0).numpy()\\n",
    "        \\n",
    "        axes[i].imshow(img_np)\\n",
    "        axes[i].set_title(f'Class: {labels[i].item()}')\\n",
    "        axes[i].axis('off')\\n",
    "    \\n",
    "    plt.tight_layout()\\n",
    "    plt.show()\\n",
    "\\n",
    "show_batch(data_loaders['train'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Creation\\n",
    "Create and configure the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get class names\\n",
    "class_names = config.get('dataset.asl_alphabet.classes')\\n",
    "print(f'Classes ({len(class_names)}): {class_names}')\\n",
    "\\n",
    "# Create model\\n",
    "model = create_model(\\n",
    "    architecture=CONFIG['model_architecture'],\\n",
    "    num_classes=len(class_names),\\n",
    "    pretrained=True\\n",
    ")\\n",
    "\\n",
    "print(f'Model: {model.__class__.__name__}')\\n",
    "print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training\\n",
    "Set up and run training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create trainer\\n",
    "trainer = Trainer(\\n",
    "    model=model,\\n",
    "    train_loader=data_loaders['train'],\\n",
    "    val_loader=data_loaders['val'],\\n",
    "    test_loader=data_loaders.get('test'),\\n",
    "    class_names=class_names,\\n",
    "    experiment_name=CONFIG['experiment_name']\\n",
    ")\\n",
    "\\n",
    "print('Trainer initialized!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training configuration\\n",
    "optimizer_config = {\\n",
    "    'name': 'adam',\\n",
    "    'lr': CONFIG['learning_rate'],\\n",
    "    'weight_decay': 1e-4\\n",
    "}\\n",
    "\\n",
    "scheduler_config = {\\n",
    "    'name': 'cosine',\\n",
    "    'warmup_epochs': 5\\n",
    "}\\n",
    "\\n",
    "# Start training (run this cell to train)\\n",
    "# trainer.train(\\n",
    "#     epochs=CONFIG['epochs'],\\n",
    "#     optimizer_config=optimizer_config,\\n",
    "#     scheduler_config=scheduler_config\\n",
    "# )\\n",
    "\\n",
    "print('Ready to train! Uncomment the trainer.train() call above to start training.')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''


def create_evaluation_template():
    """Create evaluation notebook template."""
    return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sign Language Detection - Model Evaluation\\n",
    "\\n",
    "This notebook provides tools for comprehensive model evaluation and analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup\\n",
    "import sys\\n",
    "from pathlib import Path\\n",
    "\\n",
    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\\n",
    "sys.path.append(str(project_root / 'src'))\\n",
    "\\n",
    "import torch\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import numpy as np\\n",
    "\\n",
    "from src.evaluation.evaluator import ModelEvaluator\\n",
    "from src.evaluation.analysis import ModelAnalysis\\n",
    "from src.models.model_factory import create_model\\n",
    "from src.data.data_loader import create_data_loaders\\n",
    "from src.utils.config import config\\n",
    "\\n",
    "print('✅ Evaluation environment loaded!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Model and Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configuration\\n",
    "MODEL_PATH = 'checkpoints/basic_training_example/best_model.pth'\\n",
    "DATASET_NAME = 'asl_alphabet'\\n",
    "DATA_DIR = 'datasets/asl_alphabet'\\n",
    "\\n",
    "# Load model\\n",
    "checkpoint = torch.load(MODEL_PATH, map_location='cpu')\\n",
    "class_names = config.get('dataset.asl_alphabet.classes')\\n",
    "\\n",
    "model = create_model('efficientnet_b0', num_classes=len(class_names), pretrained=False)\\n",
    "model.load_state_dict(checkpoint['model_state_dict'])\\n",
    "\\n",
    "# Load data\\n",
    "data_loaders = create_data_loaders(DATASET_NAME, DATA_DIR)\\n",
    "\\n",
    "print(f'Model loaded: {model.__class__.__name__}')\\n",
    "print(f'Test samples: {len(data_loaders[\"test\"].dataset)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Comprehensive Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run evaluation\\n",
    "evaluator = ModelEvaluator(model, class_names)\\n",
    "results = evaluator.evaluate_dataset(\\n",
    "    data_loaders['test'],\\n",
    "    save_predictions=True,\\n",
    "    save_dir='notebook_results'\\n",
    ")\\n",
    "\\n",
    "print(f'Accuracy: {results[\"accuracy\"]:.4f}')\\n",
    "print(f'F1-Score: {results[\"f1_macro\"]:.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot confusion matrix\\n",
    "evaluator.plot_confusion_matrix()\\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''


def create_data_exploration_template():
    """Create data exploration notebook template."""
    return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Exploration - Sign Language Dataset\\n",
    "\\n",
    "Explore and analyze the sign language dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup\\n",
    "import sys\\n",
    "from pathlib import Path\\n",
    "\\n",
    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\\n",
    "sys.path.append(str(project_root / 'src'))\\n",
    "\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import numpy as np\\n",
    "from collections import Counter\\n",
    "\\n",
    "from src.data.data_loader import create_data_loaders\\n",
    "from src.utils.config import config\\n",
    "\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "print('✅ Data exploration environment ready!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dataset Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\\n",
    "data_loaders = create_data_loaders('asl_alphabet', 'datasets/asl_alphabet')\\n",
    "class_names = config.get('dataset.asl_alphabet.classes')\\n",
    "\\n",
    "print(f'Dataset: ASL Alphabet')\\n",
    "print(f'Classes: {len(class_names)}')\\n",
    "print(f'Class names: {class_names}')\\n",
    "print(f'Training samples: {len(data_loaders[\"train\"].dataset)}')\\n",
    "print(f'Validation samples: {len(data_loaders[\"val\"].dataset)}')\\n",
    "print(f'Test samples: {len(data_loaders[\"test\"].dataset)}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''


def create_model_comparison_template():
    """Create model comparison notebook template."""
    return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Comparison - Sign Language Detection\\n",
    "\\n",
    "Compare different model architectures and their performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup for model comparison\\n",
    "import sys\\n",
    "from pathlib import Path\\n",
    "\\n",
    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\\n",
    "sys.path.append(str(project_root / 'src'))\\n",
    "\\n",
    "from src.models.model_factory import create_model, get_model_recommendations\\n",
    "from src.evaluation.analysis import compare_models\\n",
    "from src.utils.config import config\\n",
    "\\n",
    "print('✅ Model comparison environment ready!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Architecture Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get model recommendations\\n",
    "desktop_recs = get_model_recommendations('desktop', 'accuracy')\\n",
    "mobile_recs = get_model_recommendations('mobile', 'speed')\\n",
    "\\n",
    "print('Desktop recommendations:', desktop_recs['primary'])\\n",
    "print('Mobile recommendations:', mobile_recs['primary'])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''


def launch_jupyter():
    """Launch Jupyter notebook server."""
    try:
        print("🚀 Launching Jupyter notebook server...")
        print("   Navigate to http://localhost:8888 in your browser")
        print("   Press Ctrl+C to stop the server")
        print()
        
        # Launch Jupyter
        subprocess.run([
            sys.executable, '-m', 'jupyter', 'notebook',
            '--notebook-dir=.',
            '--ip=localhost',
            '--port=8888',
            '--no-browser',
            '--allow-root'
        ])
        
    except KeyboardInterrupt:
        print("\n📝 Jupyter server stopped.")
    except FileNotFoundError:
        print("❌ Jupyter not found. Install with: pip install jupyter")
    except Exception as e:
        print(f"❌ Failed to launch Jupyter: {e}")


def main():
    """Main function."""
    print("📓 Sign Language Detection - Jupyter Launcher")
    print("=" * 50)
    
    # Setup environment
    setup_jupyter_environment()
    
    # Create notebook templates
    notebooks_dir = create_notebook_templates()
    
    print(f"📁 Notebooks directory: {notebooks_dir}")
    print()
    print("Available notebooks:")
    print("  📓 training_template.ipynb - Interactive training")
    print("  📊 evaluation_template.ipynb - Model evaluation")
    print("  🔍 data_exploration.ipynb - Dataset analysis")
    print("  ⚖️  model_comparison.ipynb - Compare models")
    print()
    
    # Launch Jupyter
    launch_jupyter()


if __name__ == '__main__':
    main()
