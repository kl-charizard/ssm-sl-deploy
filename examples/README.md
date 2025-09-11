# Examples

This directory contains practical examples showing how to use the Sign Language Detection Framework for various tasks.

## 📚 Available Examples

### 1. Basic Training (`basic_training.py`)
**What it shows:**
- Setting up a basic training pipeline
- Training a model on ASL alphabet dataset
- Saving and evaluating the trained model

**Prerequisites:**
- ASL alphabet dataset downloaded to `datasets/asl_alphabet/`

**Usage:**
```bash
cd examples
python basic_training.py
```

**Expected output:**
- Trained model saved to `checkpoints/basic_training_example/`
- Training logs in `logs/basic_training_example/`
- Evaluation results in `examples/results/`

---

### 2. Custom Dataset (`custom_dataset_example.py`)
**What it shows:**
- Creating a custom dataset structure
- Configuring the framework for custom classes
- Training a model on custom sign language data

**What you'll learn:**
- How to organize custom datasets
- Configuring custom class names
- Training with limited data

**Usage:**
```bash
cd examples
python custom_dataset_example.py
```

**The script will:**
1. Create a custom dataset structure in `examples/custom_dataset/`
2. Guide you through adding your own images
3. Train a model on your custom data (if images are provided)

---

### 3. Mobile Deployment (`mobile_deployment_example.py`)
**What it shows:**
- Training a mobile-optimized model
- Applying quantization for efficiency
- Exporting to mobile formats (PyTorch Mobile, ONNX, Core ML, TensorFlow Lite)
- Creating deployment packages for Android and iOS

**What you'll learn:**
- Mobile-specific model optimization
- Quantization techniques
- Cross-platform model export
- Integration code generation

**Usage:**
```bash
cd examples
python mobile_deployment_example.py
```

**Output:**
- Mobile-optimized models in `examples/mobile_exports/`
- Complete deployment packages in `examples/mobile_deployment/`
- Performance benchmarks and integration code

---

## 🚀 Running Examples

### Prerequisites
1. **Framework installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset available:**
   - For basic training: ASL alphabet dataset in `datasets/asl_alphabet/`
   - For custom dataset: Your own sign language images

### Quick Start
```bash
# Clone repository (if not already done)
git clone https://github.com/yourorg/sign-language-detection.git
cd sign-language-detection

# Run basic example
python examples/basic_training.py

# Or run specific example
cd examples
python basic_training.py
```

## 📊 Example Outputs

### Basic Training Example
```
🚀 Basic Training Example
==================================================
📊 Dataset: asl_alphabet
🏗️  Model: efficientnet_b0
📈 Epochs: 10

Loading data...
Found 87,000 total samples across 29 classes
Train samples: 69,600
Val samples: 13,050
Test samples: 4,350

Creating model...
Model parameters: 4,013,953

Starting training...
==================================================
Epoch 1/10
  Batch [ 100/2175] ( 4.6%) | Loss: 2.1234 | Time: 0.045s
  ...

✅ Training completed successfully!

Test Results:
  Accuracy: 0.9234
  F1-Score: 0.9187
  Results saved to: examples/results
```

### Custom Dataset Example
```
🎯 Custom Dataset Example
==================================================
Creating example custom dataset structure at examples/custom_dataset
✅ Custom dataset structure created at examples/custom_dataset
📝 Add your images to the class directories
⚙️  Configuration saved to examples/custom_dataset/config.txt

Classes created:
  hello/     - Place 'hello' sign images here
  thank_you/ - Place 'thank_you' sign images here
  please/    - Place 'please' sign images here
  yes/       - Place 'yes' sign images here
  no/        - Place 'no' sign images here
```

### Mobile Deployment Example
```
📱 Mobile Deployment Example
==================================================
🏗️  Training mobile-optimized model...
📊 Model: MobileNetV3
📊 Parameters: 1,529,968

🔧 Applying quantization...
📊 Original model size: 5.8 MB
📊 Quantized model size: 1.5 MB
📊 Size reduction: 74.1%

📤 Exporting to mobile formats...
✅ PyTorch Mobile: examples/mobile_exports/model_mobile.pt
✅ ONNX: examples/mobile_exports/model.onnx
✅ Core ML: examples/mobile_exports/model.mlmodel

📦 Creating deployment packages...
✅ Deployment package created at: examples/mobile_deployment

⚡ Benchmarking mobile performance...
📊 Benchmarking 224x224 input:
  Average time: 45.2 ms
  Average FPS: 22.1
  Mobile performance: Excellent
```

## 🛠️ Customizing Examples

### Modifying Training Parameters
Edit the configuration variables at the top of each example:

```python
# In basic_training.py
num_epochs = 20        # Increase for better accuracy
model_architecture = 'resnet50'  # Try different architectures
batch_size = 64        # Adjust based on your GPU memory
```

### Adding Your Own Classes
For custom dataset example:

```python
# In custom_dataset_example.py
custom_classes = ['hello', 'goodbye', 'please', 'thank_you', 'love', 'family']
```

### Changing Export Formats
For mobile deployment:

```python
# In mobile_deployment_example.py
export_formats = ['torchscript', 'onnx', 'coreml']  # Choose formats you need
target_platforms = ['android']  # Or ['ios'] or ['android', 'ios']
```

## 📝 Common Issues and Solutions

### "Dataset not found"
**Problem:** `❌ Dataset not found at datasets/asl_alphabet`

**Solution:** 
1. Download the ASL alphabet dataset from Kaggle
2. Extract to `datasets/asl_alphabet/` directory
3. See `SETUP.md` for detailed instructions

### "CUDA out of memory"
**Problem:** GPU memory error during training

**Solution:** Reduce batch size in the example:
```python
batch_size = 16  # Reduce from 32
```

### "Module not found"
**Problem:** Import errors when running examples

**Solution:** Run from the project root directory:
```bash
# From project root
python examples/basic_training.py

# Not from examples directory
```

## 🎯 What to Try Next

After running the examples:

1. **Experiment with different models:**
   - Try `mobilenet_v3_small` for speed
   - Try `efficientnet_b1` for accuracy
   - Try custom architectures

2. **Test different datasets:**
   - Create your own sign language dataset
   - Try different augmentation techniques
   - Experiment with data balancing

3. **Deploy to real applications:**
   - Integrate exported models into mobile apps
   - Create web applications using the Streamlit demo
   - Build real-time detection systems

4. **Advanced techniques:**
   - Implement ensemble methods
   - Try different quantization techniques
   - Experiment with knowledge distillation

## 📞 Getting Help

If you run into issues with the examples:

1. **Check the error message** - Most issues are related to missing datasets or dependencies
2. **Review prerequisites** - Make sure you have everything installed
3. **Check the main README** - General setup and troubleshooting information
4. **Open an issue** - If you find a bug or need help with customization

---

**Happy coding!** 🚀 These examples should give you a solid foundation for building your own sign language detection applications.
