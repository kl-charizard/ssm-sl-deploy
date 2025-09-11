"""Mobile deployment preparation utilities."""

import torch
import torch.nn as nn
import numpy as np
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import subprocess
import platform

from ..models.base_model import BaseSignLanguageModel
from .quantization import ModelQuantizer
from .export import ModelExporter
from ..utils.config import config


class MobileDeploymentPrep:
    """Prepare models for mobile deployment across platforms."""
    
    def __init__(
        self,
        model: BaseSignLanguageModel,
        class_names: List[str],
        target_platforms: List[str] = ['android', 'ios']
    ):
        """Initialize mobile deployment preparation.
        
        Args:
            model: Model to prepare for deployment
            class_names: List of class names
            target_platforms: Target mobile platforms
        """
        self.model = model
        self.class_names = class_names
        self.target_platforms = target_platforms
        self.quantizer = ModelQuantizer(model)
        self.exporter = ModelExporter(model, class_names)
        
        # Model optimization results
        self.optimized_models = {}
        self.deployment_artifacts = {}
        
        print(f"Mobile deployment prep initialized for: {', '.join(target_platforms)}")
    
    def prepare_for_android(
        self,
        output_dir: str,
        quantization_method: str = 'dynamic',
        use_nnapi: bool = True
    ) -> Dict[str, Any]:
        """Prepare model for Android deployment.
        
        Args:
            output_dir: Output directory for Android artifacts
            quantization_method: Quantization method to use
            use_nnapi: Whether to optimize for Android NNAPI
            
        Returns:
            Dictionary with Android deployment artifacts
        """
        print("Preparing model for Android deployment...")
        
        output_dir = Path(output_dir)
        android_dir = output_dir / 'android'
        android_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        # 1. Quantize model for mobile efficiency
        print("  Quantizing model for mobile...")
        if quantization_method == 'dynamic':
            quantized_model = self.quantizer.dynamic_quantization()
        else:
            raise NotImplementedError(f"Quantization method {quantization_method} not implemented")
        
        # 2. Export to PyTorch Mobile
        print("  Exporting to PyTorch Mobile format...")
        mobile_model_path = self._export_pytorch_mobile(
            quantized_model, 
            str(android_dir / 'model_mobile.ptl')
        )
        artifacts['pytorch_mobile'] = mobile_model_path
        
        # 3. Export to TensorFlow Lite for Android
        try:
            print("  Exporting to TensorFlow Lite...")
            tflite_path = self.exporter.export_tensorflow_lite(
                str(android_dir / 'model.tflite'),
                quantize=True
            )
            artifacts['tflite'] = tflite_path
        except Exception as e:
            print(f"  Warning: TFLite export failed: {e}")
        
        # 4. Create Android integration files
        print("  Creating Android integration files...")
        self._create_android_integration(android_dir, artifacts)
        
        # 5. Generate performance report
        print("  Generating performance report...")
        perf_report = self._generate_mobile_performance_report(quantized_model, 'android')
        artifacts['performance_report'] = str(android_dir / 'performance_report.json')
        
        with open(artifacts['performance_report'], 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        print(f"Android deployment artifacts saved to {android_dir}")
        return artifacts
    
    def prepare_for_ios(
        self,
        output_dir: str,
        quantization_method: str = 'dynamic',
        use_coreml: bool = True
    ) -> Dict[str, Any]:
        """Prepare model for iOS deployment.
        
        Args:
            output_dir: Output directory for iOS artifacts
            quantization_method: Quantization method to use
            use_coreml: Whether to export to Core ML
            
        Returns:
            Dictionary with iOS deployment artifacts
        """
        print("Preparing model for iOS deployment...")
        
        output_dir = Path(output_dir)
        ios_dir = output_dir / 'ios'
        ios_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        # 1. Quantize model
        print("  Quantizing model for mobile...")
        if quantization_method == 'dynamic':
            quantized_model = self.quantizer.dynamic_quantization()
        else:
            raise NotImplementedError(f"Quantization method {quantization_method} not implemented")
        
        # 2. Export to PyTorch Mobile
        print("  Exporting to PyTorch Mobile format...")
        mobile_model_path = self._export_pytorch_mobile(
            quantized_model,
            str(ios_dir / 'model_mobile.ptl')
        )
        artifacts['pytorch_mobile'] = mobile_model_path
        
        # 3. Export to Core ML if requested
        if use_coreml:
            try:
                print("  Exporting to Core ML...")
                coreml_path = self.exporter.export_coreml(
                    str(ios_dir / 'SignLanguageModel.mlmodel')
                )
                artifacts['coreml'] = coreml_path
            except Exception as e:
                print(f"  Warning: Core ML export failed: {e}")
        
        # 4. Create iOS integration files
        print("  Creating iOS integration files...")
        self._create_ios_integration(ios_dir, artifacts)
        
        # 5. Generate performance report
        print("  Generating performance report...")
        perf_report = self._generate_mobile_performance_report(quantized_model, 'ios')
        artifacts['performance_report'] = str(ios_dir / 'performance_report.json')
        
        with open(artifacts['performance_report'], 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        print(f"iOS deployment artifacts saved to {ios_dir}")
        return artifacts
    
    def _export_pytorch_mobile(self, model: nn.Module, save_path: str) -> str:
        """Export model to PyTorch Mobile format.
        
        Args:
            model: Model to export
            save_path: Path to save mobile model
            
        Returns:
            Path to saved model
        """
        # Create example input
        example_input = torch.randn(1, 3, 224, 224)
        
        try:
            # Script the model
            scripted_model = torch.jit.script(model)
            
            # Optimize for mobile
            mobile_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
            
            # Save
            mobile_model._save_for_lite_interpreter(save_path)
            
            print(f"    PyTorch Mobile model saved to {save_path}")
            return save_path
            
        except Exception as e:
            # Fallback to regular torchscript
            print(f"    Mobile optimization failed, using regular TorchScript: {e}")
            scripted_model = torch.jit.trace(model, example_input)
            scripted_model.save(save_path)
            return save_path
    
    def _create_android_integration(self, output_dir: Path, artifacts: Dict[str, str]):
        """Create Android integration files."""
        # Create Java model wrapper
        java_wrapper = self._generate_android_java_wrapper()
        with open(output_dir / 'SignLanguageClassifier.java', 'w') as f:
            f.write(java_wrapper)
        
        # Create Kotlin model wrapper
        kotlin_wrapper = self._generate_android_kotlin_wrapper()
        with open(output_dir / 'SignLanguageClassifier.kt', 'w') as f:
            f.write(kotlin_wrapper)
        
        # Create build.gradle dependencies
        gradle_deps = self._generate_android_gradle_dependencies()
        with open(output_dir / 'dependencies.gradle', 'w') as f:
            f.write(gradle_deps)
        
        # Create README
        readme = self._generate_android_readme(artifacts)
        with open(output_dir / 'README.md', 'w') as f:
            f.write(readme)
    
    def _create_ios_integration(self, output_dir: Path, artifacts: Dict[str, str]):
        """Create iOS integration files."""
        # Create Swift model wrapper
        swift_wrapper = self._generate_ios_swift_wrapper()
        with open(output_dir / 'SignLanguageClassifier.swift', 'w') as f:
            f.write(swift_wrapper)
        
        # Create Objective-C wrapper
        objc_header = self._generate_ios_objc_header()
        with open(output_dir / 'SignLanguageClassifier.h', 'w') as f:
            f.write(objc_header)
        
        objc_impl = self._generate_ios_objc_implementation()
        with open(output_dir / 'SignLanguageClassifier.m', 'w') as f:
            f.write(objc_impl)
        
        # Create Podspec file
        podspec = self._generate_ios_podspec()
        with open(output_dir / 'SignLanguageDetection.podspec', 'w') as f:
            f.write(podspec)
        
        # Create README
        readme = self._generate_ios_readme(artifacts)
        with open(output_dir / 'README.md', 'w') as f:
            f.write(readme)
    
    def _generate_mobile_performance_report(
        self,
        model: nn.Module,
        platform: str
    ) -> Dict[str, Any]:
        """Generate performance report for mobile deployment."""
        # Model size analysis
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Architecture analysis
        layer_count = len(list(model.modules()))
        conv_layers = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
        linear_layers = len([m for m in model.modules() if isinstance(m, nn.Linear)])
        
        # Estimated performance (platform-specific)
        if platform == 'android':
            estimated_inference_ms = self._estimate_android_inference_time(model)
            recommended_devices = [
                "Flagship devices (Snapdragon 8 Gen 1+)",
                "Mid-range devices (Snapdragon 7 series)",
                "Devices with Android 8.0+ for NNAPI support"
            ]
        else:  # iOS
            estimated_inference_ms = self._estimate_ios_inference_time(model)
            recommended_devices = [
                "iPhone 12 and newer",
                "iPhone X/XS series",
                "iPad Pro (2018 and newer)"
            ]
        
        report = {
            'model_analysis': {
                'total_parameters': total_params,
                'model_size_mb': model_size_mb,
                'layer_count': layer_count,
                'conv_layers': conv_layers,
                'linear_layers': linear_layers
            },
            'performance_estimates': {
                'estimated_inference_time_ms': estimated_inference_ms,
                'estimated_fps': 1000 / estimated_inference_ms if estimated_inference_ms > 0 else 0,
                'memory_usage_mb': model_size_mb * 1.5,  # Rough estimate
                'battery_impact': 'Medium' if estimated_inference_ms < 100 else 'High'
            },
            'deployment_recommendations': {
                'target_devices': recommended_devices,
                'optimization_suggestions': self._get_optimization_suggestions(model),
                'integration_notes': self._get_integration_notes(platform)
            },
            'class_information': {
                'num_classes': len(self.class_names),
                'class_names': self.class_names
            }
        }
        
        return report
    
    def _estimate_android_inference_time(self, model: nn.Module) -> float:
        """Estimate inference time on Android devices."""
        # Rough estimation based on model complexity
        param_count = sum(p.numel() for p in model.parameters())
        
        # Base time estimation (ms) - very rough
        if param_count < 1e6:  # < 1M params
            base_time = 20
        elif param_count < 5e6:  # < 5M params
            base_time = 50
        elif param_count < 20e6:  # < 20M params
            base_time = 100
        else:
            base_time = 200
        
        # Adjust for quantization (assume 30% speedup)
        return base_time * 0.7
    
    def _estimate_ios_inference_time(self, model: nn.Module) -> float:
        """Estimate inference time on iOS devices."""
        # iOS devices generally perform better than Android
        android_time = self._estimate_android_inference_time(model)
        return android_time * 0.8  # Assume 20% faster on iOS
    
    def _get_optimization_suggestions(self, model: nn.Module) -> List[str]:
        """Get optimization suggestions for the model."""
        suggestions = []
        
        param_count = sum(p.numel() for p in model.parameters())
        
        if param_count > 10e6:
            suggestions.append("Consider using a smaller model architecture")
            suggestions.append("Apply more aggressive quantization")
        
        if param_count > 5e6:
            suggestions.append("Use INT8 quantization for better performance")
            suggestions.append("Consider model pruning techniques")
        
        # Architecture-specific suggestions
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if len(conv_layers) > 20:
            suggestions.append("Consider using depthwise separable convolutions")
        
        suggestions.extend([
            "Test on target devices before deployment",
            "Monitor battery usage during inference",
            "Implement proper model warming for consistent performance"
        ])
        
        return suggestions
    
    def _get_integration_notes(self, platform: str) -> List[str]:
        """Get platform-specific integration notes."""
        if platform == 'android':
            return [
                "Add PyTorch Android library to your app dependencies",
                "Place model file in assets folder",
                "Use AsyncTask or coroutines for inference to avoid blocking UI",
                "Consider using Android NNAPI for hardware acceleration",
                "Test on different Android versions and devices"
            ]
        else:  # iOS
            return [
                "Add model file to Xcode project bundle",
                "Use Core ML for best iOS integration",
                "Implement proper error handling for model loading",
                "Use background queues for inference",
                "Test on different iOS devices and versions"
            ]
    
    def create_deployment_package(
        self,
        output_dir: str,
        include_examples: bool = True,
        include_benchmarks: bool = True
    ) -> str:
        """Create complete deployment package.
        
        Args:
            output_dir: Output directory for deployment package
            include_examples: Whether to include example code
            include_benchmarks: Whether to include benchmark results
            
        Returns:
            Path to deployment package
        """
        print("Creating complete deployment package...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare models for all target platforms
        all_artifacts = {}
        
        for platform in self.target_platforms:
            if platform.lower() == 'android':
                artifacts = self.prepare_for_android(str(output_dir))
                all_artifacts['android'] = artifacts
            elif platform.lower() == 'ios':
                artifacts = self.prepare_for_ios(str(output_dir))
                all_artifacts['ios'] = artifacts
        
        # Create package metadata
        package_metadata = {
            'version': '1.0.0',
            'model_info': self.model.get_model_info(),
            'class_names': self.class_names,
            'target_platforms': self.target_platforms,
            'artifacts': all_artifacts,
            'created_at': np.datetime64('now').isoformat()
        }
        
        with open(output_dir / 'package_info.json', 'w') as f:
            json.dump(package_metadata, f, indent=2, default=str)
        
        # Create main README
        main_readme = self._generate_main_readme(all_artifacts)
        with open(output_dir / 'README.md', 'w') as f:
            f.write(main_readme)
        
        # Include examples if requested
        if include_examples:
            self._create_example_projects(output_dir)
        
        print(f"Deployment package created at {output_dir}")
        return str(output_dir)
    
    def _generate_android_java_wrapper(self) -> str:
        """Generate Android Java wrapper code."""
        return f'''
package com.yourapp.signlanguage;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class SignLanguageClassifier {{
    private static final String TAG = "SignLanguageClassifier";
    private Module module;
    private String[] classNames = {{{", ".join([f'"{name}"' for name in self.class_names])}}};
    
    public SignLanguageClassifier(Context context, String modelPath) {{
        try {{
            module = LiteModuleLoader.load(assetFilePath(context, modelPath));
        }} catch (IOException e) {{
            Log.e(TAG, "Error loading model", e);
        }}
    }}
    
    public PredictionResult predict(Bitmap bitmap) {{
        if (module == null) {{
            return null;
        }}
        
        // Preprocess image
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );
        
        // Run inference
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] outputs = outputTensor.getDataAsFloatArray();
        
        // Apply softmax
        float[] probabilities = softmax(outputs);
        
        // Find best prediction
        int maxIndex = 0;
        float maxProb = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {{
            if (probabilities[i] > maxProb) {{
                maxProb = probabilities[i];
                maxIndex = i;
            }}
        }}
        
        return new PredictionResult(
            classNames[maxIndex],
            maxIndex,
            maxProb,
            probabilities
        );
    }}
    
    private float[] softmax(float[] logits) {{
        float[] result = new float[logits.length];
        float sum = 0.0f;
        
        // Find max for numerical stability
        float max = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {{
            if (logit > max) max = logit;
        }}
        
        // Compute exp and sum
        for (int i = 0; i < logits.length; i++) {{
            result[i] = (float) Math.exp(logits[i] - max);
            sum += result[i];
        }}
        
        // Normalize
        for (int i = 0; i < result.length; i++) {{
            result[i] /= sum;
        }}
        
        return result;
    }}
    
    private String assetFilePath(Context context, String assetName) throws IOException {{
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {{
            return file.getAbsolutePath();
        }}
        
        try (InputStream is = context.getAssets().open(assetName)) {{
            try (FileOutputStream os = new FileOutputStream(file)) {{
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {{
                    os.write(buffer, 0, read);
                }}
                os.flush();
            }}
            return file.getAbsolutePath();
        }}
    }}
    
    public static class PredictionResult {{
        public final String className;
        public final int classIndex;
        public final float confidence;
        public final float[] probabilities;
        
        public PredictionResult(String className, int classIndex, float confidence, float[] probabilities) {{
            this.className = className;
            this.classIndex = classIndex;
            this.confidence = confidence;
            this.probabilities = probabilities;
        }}
        
        @Override
        public String toString() {{
            return String.format("PredictionResult{{className='%s', confidence=%.3f}}", 
                className, confidence);
        }}
    }}
}}
'''
    
    def _generate_android_kotlin_wrapper(self) -> str:
        """Generate Android Kotlin wrapper code."""
        class_names_str = self.class_names.__repr__()
        
        return f'''
package com.yourapp.signlanguage

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.*
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.exp

class SignLanguageClassifier(context: Context, modelPath: String) {{
    
    companion object {{
        private const val TAG = "SignLanguageClassifier"
    }}
    
    private var module: Module? = null
    private val classNames = arrayOf({", ".join([f'"{name}"' for name in self.class_names])})
    
    init {{
        try {{
            module = LiteModuleLoader.load(assetFilePath(context, modelPath))
        }} catch (e: IOException) {{
            Log.e(TAG, "Error loading model", e)
        }}
    }}
    
    fun predict(bitmap: Bitmap): PredictionResult? {{
        val model = module ?: return null
        
        // Preprocess image
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        
        // Run inference
        val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
        val outputs = outputTensor.dataAsFloatArray
        
        // Apply softmax
        val probabilities = softmax(outputs)
        
        // Find best prediction
        val maxIndex = probabilities.indices.maxByOrNull {{ probabilities[it] }} ?: 0
        val maxProb = probabilities[maxIndex]
        
        return PredictionResult(
            className = classNames[maxIndex],
            classIndex = maxIndex,
            confidence = maxProb,
            probabilities = probabilities
        )
    }}
    
    private fun softmax(logits: FloatArray): FloatArray {{
        val result = FloatArray(logits.size)
        
        // Find max for numerical stability
        val max = logits.maxOrNull() ?: 0f
        
        // Compute exp and sum
        var sum = 0f
        for (i in logits.indices) {{
            result[i] = exp(logits[i] - max)
            sum += result[i]
        }}
        
        // Normalize
        for (i in result.indices) {{
            result[i] /= sum
        }}
        
        return result
    }}
    
    private fun assetFilePath(context: Context, assetName: String): String {{
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {{
            return file.absolutePath
        }}
        
        context.assets.open(assetName).use {{ inputStream ->
            FileOutputStream(file).use {{ outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also {{ read = it }} != -1) {{
                    outputStream.write(buffer, 0, read)
                }}
                outputStream.flush()
            }}
        }}
        return file.absolutePath
    }}
    
    data class PredictionResult(
        val className: String,
        val classIndex: Int,
        val confidence: Float,
        val probabilities: FloatArray
    ) {{
        override fun toString(): String {{
            return "PredictionResult(className='$className', confidence=${{String.format("%.3f", confidence)}})"
        }}
        
        override fun equals(other: Any?): Boolean {{
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            
            other as PredictionResult
            
            if (className != other.className) return false
            if (classIndex != other.classIndex) return false
            if (confidence != other.confidence) return false
            if (!probabilities.contentEquals(other.probabilities)) return false
            
            return true
        }}
        
        override fun hashCode(): Int {{
            var result = className.hashCode()
            result = 31 * result + classIndex
            result = 31 * result + confidence.hashCode()
            result = 31 * result + probabilities.contentHashCode()
            return result
        }}
    }}
}}
'''
    
    def _generate_android_gradle_dependencies(self) -> str:
        """Generate Android Gradle dependencies."""
        return '''
// Add to your app's build.gradle dependencies block

dependencies {
    // PyTorch Android
    implementation 'org.pytorch:pytorch_android_lite:1.12.2'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.12.2'
    
    // Optional: For TensorFlow Lite support
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.3'
}

// Add to your app's build.gradle android block
android {
    aaptOptions {
        noCompress "ptl", "tflite"
    }
}
'''
    
    def _generate_android_readme(self, artifacts: Dict[str, str]) -> str:
        """Generate Android README."""
        return f'''# Sign Language Detection - Android Integration

## Overview
This package contains everything needed to integrate sign language detection into your Android application.

## Files Included
- `model_mobile.ptl` - Quantized PyTorch Mobile model
- `SignLanguageClassifier.java` - Java wrapper class
- `SignLanguageClassifier.kt` - Kotlin wrapper class  
- `dependencies.gradle` - Required dependencies
- `performance_report.json` - Performance analysis

## Quick Setup

### 1. Add Dependencies
Add the dependencies from `dependencies.gradle` to your app's `build.gradle` file.

### 2. Add Model to Assets
Copy `model_mobile.ptl` to your app's `src/main/assets/` folder.

### 3. Initialize Classifier
```kotlin
val classifier = SignLanguageClassifier(context, "model_mobile.ptl")
```

### 4. Make Predictions
```kotlin
val bitmap: Bitmap = // your image
val result = classifier.predict(bitmap)
result?.let {{
    Log.d("Prediction", "Class: ${{result.className}}, Confidence: ${{result.confidence}}")
}}
```

## Model Information
- **Classes**: {len(self.class_names)}
- **Input Size**: 224x224 pixels
- **Model Format**: PyTorch Mobile (.ptl)
- **Quantization**: Dynamic INT8

## Performance Notes
- See `performance_report.json` for detailed performance analysis
- Test on your target devices for accurate performance metrics
- Consider using background threads for inference

## Troubleshooting
1. **Model Loading Error**: Ensure the model file is in the assets folder
2. **Out of Memory**: Try reducing image resolution before inference
3. **Slow Performance**: Test on different devices and consider further optimization

## Support
For issues or questions, please refer to the main project documentation.
'''
    
    def _generate_ios_swift_wrapper(self) -> str:
        """Generate iOS Swift wrapper code."""
        return f'''
import Foundation
import CoreML
import Vision
import UIKit

@available(iOS 11.0, *)
public class SignLanguageClassifier {{
    
    private var model: MLModel?
    private let classNames = [{", ".join([f'"{name}"' for name in self.class_names])}]
    
    public struct PredictionResult {{
        public let className: String
        public let classIndex: Int
        public let confidence: Float
        public let probabilities: [Float]
        
        public init(className: String, classIndex: Int, confidence: Float, probabilities: [Float]) {{
            self.className = className
            self.classIndex = classIndex
            self.confidence = confidence
            self.probabilities = probabilities
        }}
    }}
    
    public enum ClassifierError: Error {{
        case modelLoadingFailed
        case predictionFailed
        case invalidInput
        case modelNotLoaded
    }}
    
    public init(modelName: String = "SignLanguageModel") throws {{
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodel") else {{
            throw ClassifierError.modelLoadingFailed
        }}
        
        do {{
            self.model = try MLModel(contentsOf: modelURL)
        }} catch {{
            throw ClassifierError.modelLoadingFailed
        }}
    }}
    
    public func predict(image: UIImage, completion: @escaping (Result<PredictionResult, ClassifierError>) -> Void) {{
        guard let model = model else {{
            completion(.failure(.modelNotLoaded))
            return
        }}
        
        guard let cgImage = image.cgImage else {{
            completion(.failure(.invalidInput))
            return
        }}
        
        // Create Vision request
        let request = VNCoreMLRequest(model: try! VNCoreMLModel(for: model)) {{ request, error in
            if let error = error {{
                completion(.failure(.predictionFailed))
                return
            }}
            
            guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                  let multiArray = results.first?.featureValue.multiArrayValue else {{
                completion(.failure(.predictionFailed))
                return
            }}
            
            // Convert to probabilities
            let probabilities = self.softmax(logits: self.multiArrayToFloatArray(multiArray))
            
            // Find best prediction
            guard let maxIndex = probabilities.enumerated().max(by: {{ $0.element < $1.element }})?.offset else {{
                completion(.failure(.predictionFailed))
                return
            }}
            
            let result = PredictionResult(
                className: self.classNames[maxIndex],
                classIndex: maxIndex,
                confidence: probabilities[maxIndex],
                probabilities: probabilities
            )
            
            completion(.success(result))
        }}
        
        // Perform request
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {{
            try handler.perform([request])
        }} catch {{
            completion(.failure(.predictionFailed))
        }}
    }}
    
    // Synchronous prediction method
    public func predict(image: UIImage) throws -> PredictionResult {{
        var result: Result<PredictionResult, ClassifierError>?
        let semaphore = DispatchSemaphore(value: 0)
        
        predict(image: image) {{ res in
            result = res
            semaphore.signal()
        }}
        
        semaphore.wait()
        
        switch result! {{
        case .success(let prediction):
            return prediction
        case .failure(let error):
            throw error
        }}
    }}
    
    private func multiArrayToFloatArray(_ multiArray: MLMultiArray) -> [Float] {{
        let count = multiArray.count
        let doublePtr = multiArray.dataPointer.bindMemory(to: Double.self, capacity: count)
        let doubleBuffer = UnsafeBufferPointer(start: doublePtr, count: count)
        return Array(doubleBuffer).map {{ Float($0) }}
    }}
    
    private func softmax(logits: [Float]) -> [Float] {{
        let maxLogit = logits.max() ?? 0
        let expValues = logits.map {{ exp($0 - maxLogit) }}
        let sumExp = expValues.reduce(0, +)
        return expValues.map {{ $0 / sumExp }}
    }}
}}

// MARK: - Convenience Extensions
@available(iOS 11.0, *)
public extension SignLanguageClassifier {{
    
    /// Predict from camera capture
    func predict(from sampleBuffer: CMSampleBuffer, completion: @escaping (Result<PredictionResult, ClassifierError>) -> Void) {{
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {{
            completion(.failure(.invalidInput))
            return
        }}
        
        let ciImage = CIImage(cvImageBuffer: imageBuffer)
        let context = CIContext(options: nil)
        
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {{
            completion(.failure(.invalidInput))
            return
        }}
        
        let image = UIImage(cgImage: cgImage)
        predict(image: image, completion: completion)
    }}
    
    /// Get top-k predictions
    func topKPredictions(image: UIImage, k: Int = 5, completion: @escaping (Result<[PredictionResult], ClassifierError>) -> Void) {{
        predict(image: image) {{ result in
            switch result {{
            case .success(let prediction):
                let sortedIndices = prediction.probabilities.enumerated()
                    .sorted {{ $0.element > $1.element }}
                    .prefix(k)
                    .map {{ $0.offset }}
                
                let topKResults = sortedIndices.map {{ index in
                    PredictionResult(
                        className: self.classNames[index],
                        classIndex: index,
                        confidence: prediction.probabilities[index],
                        probabilities: prediction.probabilities
                    )
                }}
                
                completion(.success(topKResults))
            case .failure(let error):
                completion(.failure(error))
            }}
        }}
    }}
}}
'''
    
    def _generate_ios_objc_header(self) -> str:
        """Generate iOS Objective-C header."""
        return '''
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface SLPredictionResult : NSObject

@property (nonatomic, readonly) NSString *className;
@property (nonatomic, readonly) NSInteger classIndex;
@property (nonatomic, readonly) float confidence;
@property (nonatomic, readonly) NSArray<NSNumber *> *probabilities;

- (instancetype)initWithClassName:(NSString *)className
                       classIndex:(NSInteger)classIndex  
                       confidence:(float)confidence
                    probabilities:(NSArray<NSNumber *> *)probabilities;

@end

@interface SignLanguageClassifier : NSObject

- (nullable instancetype)initWithModelName:(NSString *)modelName error:(NSError **)error;

- (void)predictWithImage:(UIImage *)image
              completion:(void (^)(SLPredictionResult * _Nullable result, NSError * _Nullable error))completion;

- (nullable SLPredictionResult *)predictWithImage:(UIImage *)image error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
'''
    
    def _generate_ios_objc_implementation(self) -> str:
        """Generate iOS Objective-C implementation."""
        class_names_array = ", ".join([f'@"{name}"' for name in self.class_names])
        
        return f'''
#import "SignLanguageClassifier.h"
#import <Vision/Vision.h>

static NSString * const SLClassifierErrorDomain = @"SignLanguageClassifierErrorDomain";

typedef NS_ENUM(NSInteger, SLClassifierError) {{
    SLClassifierErrorModelLoadingFailed = 1000,
    SLClassifierErrorPredictionFailed,
    SLClassifierErrorInvalidInput,
    SLClassifierErrorModelNotLoaded
}};

@implementation SLPredictionResult

- (instancetype)initWithClassName:(NSString *)className
                       classIndex:(NSInteger)classIndex  
                       confidence:(float)confidence
                    probabilities:(NSArray<NSNumber *> *)probabilities {{
    self = [super init];
    if (self) {{
        _className = [className copy];
        _classIndex = classIndex;
        _confidence = confidence;
        _probabilities = [probabilities copy];
    }}
    return self;
}}

- (NSString *)description {{
    return [NSString stringWithFormat:@"PredictionResult(className='%@', confidence=%.3f)", 
            self.className, self.confidence];
}}

@end

@interface SignLanguageClassifier ()

@property (nonatomic, strong) MLModel *model;
@property (nonatomic, strong) NSArray<NSString *> *classNames;

@end

@implementation SignLanguageClassifier

- (nullable instancetype)initWithModelName:(NSString *)modelName error:(NSError **)error {{
    self = [super init];
    if (self) {{
        self.classNames = @[{class_names_array}];
        
        NSURL *modelURL = [[NSBundle mainBundle] URLForResource:modelName withExtension:@"mlmodel"];
        if (!modelURL) {{
            if (error) {{
                *error = [NSError errorWithDomain:SLClassifierErrorDomain
                                             code:SLClassifierErrorModelLoadingFailed
                                         userInfo:@{{NSLocalizedDescriptionKey: @"Model file not found"}}];
            }}
            return nil;
        }}
        
        NSError *modelError;
        self.model = [[MLModel alloc] initWithContentsOfURL:modelURL error:&modelError];
        if (!self.model) {{
            if (error) {{
                *error = modelError;
            }}
            return nil;
        }}
    }}
    return self;
}}

- (void)predictWithImage:(UIImage *)image
              completion:(void (^)(SLPredictionResult * _Nullable result, NSError * _Nullable error))completion {{
    
    if (!self.model) {{
        NSError *error = [NSError errorWithDomain:SLClassifierErrorDomain
                                             code:SLClassifierErrorModelNotLoaded
                                         userInfo:@{{NSLocalizedDescriptionKey: @"Model not loaded"}}];
        completion(nil, error);
        return;
    }}
    
    CGImageRef cgImage = image.CGImage;
    if (!cgImage) {{
        NSError *error = [NSError errorWithDomain:SLClassifierErrorDomain
                                             code:SLClassifierErrorInvalidInput
                                         userInfo:@{{NSLocalizedDescriptionKey: @"Invalid image"}}];
        completion(nil, error);
        return;
    }}
    
    NSError *modelError;
    VNCoreMLModel *visionModel = [VNCoreMLModel modelForMLModel:self.model error:&modelError];
    if (!visionModel) {{
        completion(nil, modelError);
        return;
    }}
    
    VNCoreMLRequest *request = [[VNCoreMLRequest alloc] initWithModel:visionModel
                                                     completionHandler:^(VNRequest *request, NSError *error) {{
        if (error) {{
            completion(nil, error);
            return;
        }}
        
        VNCoreMLFeatureValueObservation *observation = request.results.firstObject;
        if (!observation || !observation.featureValue.multiArrayValue) {{
            NSError *predictionError = [NSError errorWithDomain:SLClassifierErrorDomain
                                                           code:SLClassifierErrorPredictionFailed
                                                       userInfo:@{{NSLocalizedDescriptionKey: @"Prediction failed"}}];
            completion(nil, predictionError);
            return;
        }}
        
        MLMultiArray *output = observation.featureValue.multiArrayValue;
        NSArray<NSNumber *> *probabilities = [self softmaxFromMultiArray:output];
        
        // Find best prediction
        NSInteger maxIndex = 0;
        float maxProb = probabilities[0].floatValue;
        for (NSInteger i = 1; i < probabilities.count; i++) {{
            float prob = probabilities[i].floatValue;
            if (prob > maxProb) {{
                maxProb = prob;
                maxIndex = i;
            }}
        }}
        
        SLPredictionResult *result = [[SLPredictionResult alloc] initWithClassName:self.classNames[maxIndex]
                                                                        classIndex:maxIndex
                                                                        confidence:maxProb
                                                                     probabilities:probabilities];
        completion(result, nil);
    }}];
    
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{{}}];
    NSError *handlerError;
    if (![handler performRequests:@[request] error:&handlerError]) {{
        completion(nil, handlerError);
    }}
}}

- (nullable SLPredictionResult *)predictWithImage:(UIImage *)image error:(NSError **)error {{
    __block SLPredictionResult *result = nil;
    __block NSError *blockError = nil;
    
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    
    [self predictWithImage:image completion:^(SLPredictionResult * _Nullable res, NSError * _Nullable err) {{
        result = res;
        blockError = err;
        dispatch_semaphore_signal(semaphore);
    }}];
    
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
    
    if (error) {{
        *error = blockError;
    }}
    
    return result;
}}

- (NSArray<NSNumber *> *)softmaxFromMultiArray:(MLMultiArray *)multiArray {{
    NSMutableArray<NSNumber *> *logits = [NSMutableArray array];
    
    // Convert to float array
    for (NSInteger i = 0; i < multiArray.count; i++) {{
        NSNumber *value = [multiArray objectAtIndexedSubscript:i];
        [logits addObject:value];
    }}
    
    // Find max for numerical stability
    float maxLogit = -INFINITY;
    for (NSNumber *logit in logits) {{
        if (logit.floatValue > maxLogit) {{
            maxLogit = logit.floatValue;
        }}
    }}
    
    // Compute exp and sum
    NSMutableArray<NSNumber *> *expValues = [NSMutableArray array];
    float sumExp = 0.0f;
    for (NSNumber *logit in logits) {{
        float expValue = expf(logit.floatValue - maxLogit);
        [expValues addObject:@(expValue)];
        sumExp += expValue;
    }}
    
    // Normalize
    NSMutableArray<NSNumber *> *probabilities = [NSMutableArray array];
    for (NSNumber *expValue in expValues) {{
        [probabilities addObject:@(expValue.floatValue / sumExp)];
    }}
    
    return [probabilities copy];
}}

@end
'''
    
    def _generate_ios_podspec(self) -> str:
        """Generate iOS Podspec file."""
        return '''
Pod::Spec.new do |spec|
  spec.name          = "SignLanguageDetection"
  spec.version       = "1.0.0"
  spec.summary       = "Sign Language Detection SDK for iOS"
  spec.description   = <<-DESC
                       A comprehensive SDK for detecting sign language gestures in iOS applications.
                       Includes Core ML model and easy-to-use Swift/Objective-C interfaces.
                       DESC

  spec.homepage      = "https://github.com/yourorg/sign-language-detection"
  spec.license       = { :type => "MIT", :file => "LICENSE" }
  spec.author        = { "Your Name" => "your.email@example.com" }
  
  spec.ios.deployment_target = "11.0"
  spec.swift_version = "5.0"
  
  spec.source        = { :git => "https://github.com/yourorg/sign-language-detection.git", :tag => "#{spec.version}" }
  
  spec.source_files  = "Sources/**/*.{swift,h,m}"
  spec.resources     = "Resources/*.mlmodel"
  
  spec.frameworks    = "CoreML", "Vision", "UIKit", "Foundation"
  
  spec.dependency "TensorFlowLiteSwift", "~> 2.13.0"
end
'''
    
    def _generate_ios_readme(self, artifacts: Dict[str, str]) -> str:
        """Generate iOS README."""
        return f'''# Sign Language Detection - iOS Integration

## Overview
This package contains everything needed to integrate sign language detection into your iOS application.

## Files Included
- `SignLanguageModel.mlmodel` - Core ML model
- `model_mobile.ptl` - PyTorch Mobile model (alternative)
- `SignLanguageClassifier.swift` - Swift wrapper class
- `SignLanguageClassifier.h/m` - Objective-C wrapper classes
- `SignLanguageDetection.podspec` - CocoaPods specification
- `performance_report.json` - Performance analysis

## Requirements
- iOS 11.0+
- Xcode 12.0+
- Swift 5.0+

## Installation

### CocoaPods
Add to your `Podfile`:
```ruby
pod 'SignLanguageDetection', '~> 1.0'
```

### Manual Installation
1. Drag and drop the Core ML model file into your Xcode project
2. Add the Swift/Objective-C source files to your project
3. Ensure the model is included in your app bundle

## Quick Setup

### Swift
```swift
import SignLanguageDetection

// Initialize classifier
let classifier = try SignLanguageClassifier(modelName: "SignLanguageModel")

// Make prediction
classifier.predict(image: yourUIImage) {{ result in
    switch result {{
    case .success(let prediction):
        print("Predicted: \\(prediction.className) with \\(prediction.confidence) confidence")
    case .failure(let error):
        print("Prediction failed: \\(error)")
    }}
}}
```

### Objective-C
```objc
#import "SignLanguageClassifier.h"

// Initialize classifier
NSError *error;
SignLanguageClassifier *classifier = [[SignLanguageClassifier alloc] initWithModelName:@"SignLanguageModel" error:&error];

// Make prediction
[classifier predictWithImage:yourUIImage completion:^(SLPredictionResult *result, NSError *error) {{
    if (result) {{
        NSLog(@"Predicted: %@ with %.3f confidence", result.className, result.confidence);
    }}
}}];
```

## Model Information
- **Classes**: {len(self.class_names)}
- **Input Size**: 224x224 pixels
- **Model Format**: Core ML (.mlmodel)
- **iOS Version**: 11.0+

## Performance Notes
- See `performance_report.json` for detailed performance analysis
- Core ML provides automatic optimization for different devices
- Consider using background queues for inference

## Best Practices
1. **Memory Management**: Release classifier when not needed
2. **Threading**: Use background queues for inference to avoid blocking UI
3. **Error Handling**: Always handle potential errors in prediction
4. **Performance**: Test on your target devices

## Camera Integration
The classifier works seamlessly with camera input:

```swift
// In your camera delegate
func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {{
    classifier.predict(from: sampleBuffer) {{ result in
        // Handle prediction result
    }}
}}
```

## Troubleshooting
1. **Model Loading Error**: Ensure the .mlmodel file is in your app bundle
2. **Memory Issues**: Process images on background queue
3. **Performance Issues**: Check device compatibility in performance report

## Support
For issues or questions, please refer to the main project documentation.
'''
    
    def _generate_main_readme(self, all_artifacts: Dict[str, Any]) -> str:
        """Generate main deployment package README."""
        return f'''# Sign Language Detection - Mobile Deployment Package

## Overview
This package contains optimized models and integration code for deploying sign language detection on mobile devices.

## Package Contents

### Models
- Quantized models optimized for mobile deployment
- Multiple format support (PyTorch Mobile, Core ML, TensorFlow Lite)
- Performance reports and benchmarks

### Integration Code
- Ready-to-use wrapper classes for Android and iOS
- Example projects and usage documentation
- Build configuration files

## Supported Platforms
{chr(10).join([f"- {platform.title()}" for platform in self.target_platforms])}

## Model Information
- **Architecture**: {self.model.__class__.__name__}
- **Classes**: {len(self.class_names)} ({', '.join(self.class_names[:5])}...)
- **Input Size**: 224x224 pixels
- **Quantization**: Dynamic INT8
- **Mobile Optimized**: Yes

## Quick Start

### Android
1. Navigate to the `android/` directory
2. Follow the setup instructions in `android/README.md`
3. Add the model and wrapper classes to your Android project

### iOS
1. Navigate to the `ios/` directory  
2. Follow the setup instructions in `ios/README.md`
3. Add the Core ML model and wrapper classes to your iOS project

## Performance Summary
See platform-specific performance reports for detailed metrics:
- Android: `android/performance_report.json`
- iOS: `ios/performance_report.json`

## Model Classes
The model can detect the following sign language gestures:
{chr(10).join([f"- {name}" for name in self.class_names])}

## Technical Requirements

### Android
- Android API Level 21+ (Android 5.0)
- ARM64 or x86_64 architecture
- 50MB+ available storage
- 100MB+ RAM for inference

### iOS
- iOS 11.0+
- iPhone 6s/iPad (2017) or newer
- 50MB+ available storage
- Core ML support

## Integration Support
Each platform directory includes:
- Wrapper classes in native languages
- Example usage code
- Build configuration files
- Performance optimization guidelines

## Optimization Notes
- Models are pre-quantized for mobile efficiency
- Memory usage optimized for mobile constraints
- Battery usage considerations included in documentation

## Getting Help
- Check platform-specific README files for detailed instructions
- Review performance reports for optimization guidance
- Test on target devices before production deployment

## License
See LICENSE file for usage terms and conditions.

---
*Package generated on {np.datetime64('now').isoformat()}*
'''
    
    def _create_example_projects(self, output_dir: Path):
        """Create example projects for each platform."""
        # This would create minimal example projects
        # For brevity, just create placeholder directories
        
        if 'android' in self.target_platforms:
            android_example_dir = output_dir / 'examples' / 'android'
            android_example_dir.mkdir(parents=True, exist_ok=True)
            
            with open(android_example_dir / 'ExampleActivity.java', 'w') as f:
                f.write(self._generate_android_example())
        
        if 'ios' in self.target_platforms:
            ios_example_dir = output_dir / 'examples' / 'ios'
            ios_example_dir.mkdir(parents=True, exist_ok=True)
            
            with open(ios_example_dir / 'ExampleViewController.swift', 'w') as f:
                f.write(self._generate_ios_example())
    
    def _generate_android_example(self) -> str:
        """Generate Android example code."""
        return '''
// Example Android Activity using Sign Language Classifier

public class ExampleActivity extends AppCompatActivity {
    private SignLanguageClassifier classifier;
    private ImageView imageView;
    private TextView resultText;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_example);
        
        // Initialize views
        imageView = findViewById(R.id.imageView);
        resultText = findViewById(R.id.resultText);
        
        // Initialize classifier
        try {
            classifier = new SignLanguageClassifier(this, "model_mobile.ptl");
        } catch (Exception e) {
            Log.e("Example", "Failed to load classifier", e);
        }
        
        // Set up camera or image selection
        setupImageCapture();
    }
    
    private void classifyImage(Bitmap bitmap) {
        if (classifier != null) {
            new AsyncTask<Bitmap, Void, SignLanguageClassifier.PredictionResult>() {
                @Override
                protected PredictionResult doInBackground(Bitmap... bitmaps) {
                    return classifier.predict(bitmaps[0]);
                }
                
                @Override
                protected void onPostExecute(PredictionResult result) {
                    if (result != null) {
                        resultText.setText(String.format("Predicted: %s (%.2f)", 
                            result.className, result.confidence));
                    }
                }
            }.execute(bitmap);
        }
    }
}
'''
    
    def _generate_ios_example(self) -> str:
        """Generate iOS example code."""
        return '''
import UIKit
import AVFoundation

class ExampleViewController: UIViewController {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var captureButton: UIButton!
    
    private var classifier: SignLanguageClassifier?
    private var captureSession: AVCaptureSession?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Initialize classifier
        do {
            classifier = try SignLanguageClassifier(modelName: "SignLanguageModel")
        } catch {
            print("Failed to load classifier: \\(error)")
        }
        
        setupCamera()
    }
    
    private func setupCamera() {
        // Camera setup code here
    }
    
    @IBAction func captureButtonTapped(_ sender: UIButton) {
        // Capture image and classify
        guard let image = imageView.image else { return }
        
        classifier?.predict(image: image) { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let prediction):
                    self?.resultLabel.text = "Predicted: \\(prediction.className) (\\(String(format: "%.2f", prediction.confidence)))"
                case .failure(let error):
                    self?.resultLabel.text = "Error: \\(error.localizedDescription)"
                }
            }
        }
    }
}
'''


def optimize_for_mobile(
    model: BaseSignLanguageModel,
    class_names: List[str],
    output_dir: str,
    target_platforms: List[str] = ['android', 'ios'],
    quantization_method: str = 'dynamic',
    **kwargs
) -> Dict[str, Any]:
    """Optimize model for mobile deployment.
    
    Args:
        model: Model to optimize
        class_names: List of class names
        output_dir: Output directory for deployment artifacts
        target_platforms: Target mobile platforms
        quantization_method: Quantization method to use
        **kwargs: Additional optimization parameters
        
    Returns:
        Dictionary with optimization results
    """
    mobile_prep = MobileDeploymentPrep(model, class_names, target_platforms)
    
    results = {}
    
    for platform in target_platforms:
        if platform.lower() == 'android':
            results['android'] = mobile_prep.prepare_for_android(
                output_dir, 
                quantization_method=quantization_method,
                **kwargs.get('android', {})
            )
        elif platform.lower() == 'ios':
            results['ios'] = mobile_prep.prepare_for_ios(
                output_dir,
                quantization_method=quantization_method,
                **kwargs.get('ios', {})
            )
    
    # Create complete deployment package
    package_path = mobile_prep.create_deployment_package(
        output_dir,
        include_examples=kwargs.get('include_examples', True),
        include_benchmarks=kwargs.get('include_benchmarks', True)
    )
    
    results['package_path'] = package_path
    
    return results
