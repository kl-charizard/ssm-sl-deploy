#!/usr/bin/env python3
"""
Mobile deployment example for sign language detection.

This example shows how to:
1. Train a mobile-optimized model
2. Apply quantization for efficiency
3. Export to mobile formats
4. Create deployment packages
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.data_loader import create_data_loaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.deployment.quantization import ModelQuantizer
from src.deployment.export import ModelExporter
from src.deployment.mobile_deployment import MobileDeploymentPrep
from src.utils.config import config


def train_mobile_optimized_model():
    """Train a model optimized for mobile deployment."""
    print("🏗️  Training mobile-optimized model...")
    
    # Use ASL alphabet dataset
    dataset_name = 'asl_alphabet'
    data_dir = 'datasets/asl_alphabet'
    
    if not Path(data_dir).exists():
        print(f"❌ Dataset not found at {data_dir}")
        return None, None
    
    # Create data loaders with smaller batch size for mobile training
    data_loaders = create_data_loaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=16,  # Smaller batch size
        num_workers=4
    )
    
    class_names = config.get('dataset.asl_alphabet.classes')
    
    # Create mobile-optimized model
    model = create_model(
        architecture='mobilenet_v3_small',  # Mobile-optimized architecture
        num_classes=len(class_names),
        pretrained=True,
        dropout_rate=0.1  # Lower dropout for mobile
    )
    
    print(f"📊 Model: {model.__class__.__name__}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Quick training (fewer epochs for example)
    trainer = Trainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        class_names=class_names,
        experiment_name='mobile_deployment_example'
    )
    
    # Mobile-optimized training config
    optimizer_config = {
        'name': 'adam',
        'lr': 0.001,
        'weight_decay': 1e-5  # Lower weight decay
    }
    
    scheduler_config = {
        'name': 'cosine',
        'warmup_epochs': 2
    }
    
    # Train for fewer epochs (this is just an example)
    print("🚀 Starting mobile-optimized training...")
    trainer.train(
        epochs=5,  # Short training for example
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config
    )
    
    print("✅ Mobile model training completed!")
    return model, class_names


def apply_quantization(model, class_names):
    """Apply quantization to reduce model size."""
    print("\n🔧 Applying quantization...")
    
    # Create quantizer
    quantizer = ModelQuantizer(model)
    
    # Apply dynamic quantization (fastest method)
    quantized_model = quantizer.dynamic_quantization()
    
    # Get model sizes
    original_size = quantizer._get_model_size(model) / (1024 * 1024)  # MB
    quantized_size = quantizer._get_model_size(quantized_model) / (1024 * 1024)  # MB
    
    print(f"📊 Original model size: {original_size:.1f} MB")
    print(f"📊 Quantized model size: {quantized_size:.1f} MB")
    print(f"📊 Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
    
    return quantized_model


def export_to_mobile_formats(model, class_names):
    """Export model to various mobile formats."""
    print("\n📤 Exporting to mobile formats...")
    
    exporter = ModelExporter(model, class_names)
    export_dir = Path('examples/mobile_exports')
    export_dir.mkdir(parents=True, exist_ok=True)
    
    exported_models = {}
    
    # Export to PyTorch Mobile
    try:
        pt_path = exporter.export_torchscript(
            str(export_dir / 'model_mobile.pt'),
            method='script',
            optimize=True
        )
        exported_models['pytorch_mobile'] = pt_path
        print(f"✅ PyTorch Mobile: {pt_path}")
    except Exception as e:
        print(f"❌ PyTorch Mobile export failed: {e}")
    
    # Export to ONNX
    try:
        onnx_path = exporter.export_onnx(
            str(export_dir / 'model.onnx'),
            opset_version=11
        )
        exported_models['onnx'] = onnx_path
        print(f"✅ ONNX: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
    
    # Try Core ML export (macOS only)
    try:
        coreml_path = exporter.export_coreml(
            str(export_dir / 'model.mlmodel')
        )
        exported_models['coreml'] = coreml_path
        print(f"✅ Core ML: {coreml_path}")
    except Exception as e:
        print(f"⚠️  Core ML export skipped: {e}")
    
    # Try TensorFlow Lite export
    try:
        tflite_path = exporter.export_tensorflow_lite(
            str(export_dir / 'model.tflite'),
            quantize=True
        )
        exported_models['tflite'] = tflite_path
        print(f"✅ TensorFlow Lite: {tflite_path}")
    except Exception as e:
        print(f"⚠️  TensorFlow Lite export skipped: {e}")
    
    return exported_models


def create_deployment_packages(model, class_names):
    """Create complete mobile deployment packages."""
    print("\n📦 Creating deployment packages...")
    
    mobile_prep = MobileDeploymentPrep(
        model=model,
        class_names=class_names,
        target_platforms=['android', 'ios']
    )
    
    deployment_dir = Path('examples/mobile_deployment')
    
    # Create complete deployment package
    package_path = mobile_prep.create_deployment_package(
        output_dir=str(deployment_dir),
        include_examples=True,
        include_benchmarks=True
    )
    
    print(f"✅ Deployment package created at: {package_path}")
    
    # Show what was created
    print("\n📁 Package contents:")
    for item in sorted(Path(package_path).rglob('*')):
        if item.is_file():
            rel_path = item.relative_to(package_path)
            print(f"  📄 {rel_path}")


def benchmark_mobile_performance(model, class_names):
    """Benchmark model performance for mobile deployment."""
    print("\n⚡ Benchmarking mobile performance...")
    
    from src.demo.inference_engine import InferenceEngine
    
    # Create inference engine
    engine = InferenceEngine(model, class_names)
    
    # Benchmark different input sizes
    input_sizes = [(128, 128), (160, 160), (224, 224)]
    
    for size in input_sizes:
        print(f"\n📊 Benchmarking {size[0]}x{size[1]} input:")
        
        results = engine.benchmark(
            num_iterations=50,
            image_size=size
        )
        
        print(f"  Average time: {results['avg_time_per_inference']*1000:.1f} ms")
        print(f"  Average FPS: {results['avg_fps']:.1f}")
        print(f"  95th percentile: {results['p95_time']*1000:.1f} ms")
        
        # Mobile performance assessment
        if results['avg_time_per_inference'] < 0.1:  # < 100ms
            mobile_rating = "Excellent"
        elif results['avg_time_per_inference'] < 0.2:  # < 200ms
            mobile_rating = "Good"
        elif results['avg_time_per_inference'] < 0.5:  # < 500ms
            mobile_rating = "Acceptable"
        else:
            mobile_rating = "Poor"
        
        print(f"  Mobile performance: {mobile_rating}")


def main():
    """Run mobile deployment example."""
    print("📱 Mobile Deployment Example")
    print("=" * 50)
    
    try:
        # Step 1: Train mobile-optimized model
        model, class_names = train_mobile_optimized_model()
        
        if model is None:
            print("❌ Could not train model. Please check dataset setup.")
            return
        
        # Step 2: Apply quantization
        quantized_model = apply_quantization(model, class_names)
        
        # Step 3: Export to mobile formats
        exported_models = export_to_mobile_formats(quantized_model, class_names)
        
        # Step 4: Create deployment packages
        create_deployment_packages(quantized_model, class_names)
        
        # Step 5: Benchmark performance
        benchmark_mobile_performance(quantized_model, class_names)
        
        print("\n🎉 Mobile deployment example completed!")
        print("\n📋 Summary:")
        print("✅ Trained mobile-optimized model")
        print("✅ Applied quantization for size reduction")
        print(f"✅ Exported to {len(exported_models)} mobile formats")
        print("✅ Created deployment packages with integration code")
        print("✅ Benchmarked mobile performance")
        
        print("\n🚀 Next steps:")
        print("- Check examples/mobile_deployment/ for deployment packages")
        print("- Review Android/iOS integration code")
        print("- Test on actual mobile devices")
        print("- Optimize further based on target device performance")
        
    except Exception as e:
        print(f"❌ Mobile deployment example failed: {e}")
        raise


if __name__ == '__main__':
    main()
