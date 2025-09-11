"""Model export utilities for different deployment formats."""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import json

from ..models.base_model import BaseSignLanguageModel
from ..utils.config import config


class ModelExporter:
    """Export models to various formats for deployment."""
    
    def __init__(
        self,
        model: BaseSignLanguageModel,
        class_names: List[str],
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ):
        """Initialize model exporter.
        
        Args:
            model: Model to export
            class_names: List of class names
            input_shape: Expected input shape (batch, channels, height, width)
        """
        self.model = model
        self.class_names = class_names
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Model exporter initialized")
        print(f"Input shape: {input_shape}")
        print(f"Output classes: {len(class_names)}")
    
    def export_torchscript(
        self,
        save_path: str,
        method: str = 'script',
        optimize: bool = True
    ) -> str:
        """Export model to TorchScript format.
        
        Args:
            save_path: Path to save the exported model
            method: Export method ('script' or 'trace')
            optimize: Whether to optimize the model
            
        Returns:
            Path to saved model
        """
        print(f"Exporting to TorchScript ({method})...")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if method == 'script':
                # Script the model
                scripted_model = torch.jit.script(self.model)
            elif method == 'trace':
                # Trace the model with example input
                example_input = torch.randn(self.input_shape).to(self.device)
                scripted_model = torch.jit.trace(self.model, example_input)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Optimize if requested
            if optimize:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            # Save model
            scripted_model.save(str(save_path))
            
            # Save metadata
            metadata = {
                'input_shape': list(self.input_shape),
                'class_names': self.class_names,
                'export_method': method,
                'optimized': optimize
            }
            
            metadata_path = save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"TorchScript model saved to {save_path}")
            return str(save_path)
            
        except Exception as e:
            print(f"Error exporting to TorchScript: {e}")
            raise
    
    def export_onnx(
        self,
        save_path: str,
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        verify: bool = True
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            save_path: Path to save the exported model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            verify: Whether to verify the exported model
            
        Returns:
            Path to saved model
        """
        print(f"Exporting to ONNX (opset {opset_version})...")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create example input
        example_input = torch.randn(self.input_shape).to(self.device)
        
        # Default dynamic axes for batch dimension
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                str(save_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Verify the exported model
            if verify:
                self._verify_onnx_model(str(save_path), example_input)
            
            # Save metadata
            metadata = {
                'input_shape': list(self.input_shape),
                'class_names': self.class_names,
                'opset_version': opset_version,
                'dynamic_axes': dynamic_axes
            }
            
            metadata_path = save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"ONNX model saved to {save_path}")
            return str(save_path)
            
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            raise
    
    def _verify_onnx_model(self, onnx_path: str, example_input: torch.Tensor):
        """Verify ONNX model by comparing outputs."""
        try:
            # Load ONNX model
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get PyTorch output
            with torch.no_grad():
                torch_output = self.model(example_input).cpu().numpy()
            
            # Get ONNX output
            ort_inputs = {ort_session.get_inputs()[0].name: example_input.cpu().numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            # Compare outputs
            max_diff = np.max(np.abs(torch_output - onnx_output))
            
            if max_diff < 1e-5:
                print(f"✓ ONNX model verification passed (max diff: {max_diff:.2e})")
            else:
                print(f"⚠ ONNX model verification warning (max diff: {max_diff:.2e})")
            
        except Exception as e:
            print(f"Warning: Could not verify ONNX model: {e}")
    
    def export_tensorflow_lite(
        self,
        save_path: str,
        quantize: bool = True,
        representative_dataset: Optional[callable] = None
    ) -> str:
        """Export model to TensorFlow Lite format.
        
        Args:
            save_path: Path to save the exported model
            quantize: Whether to apply quantization
            representative_dataset: Representative dataset for quantization
            
        Returns:
            Path to saved model
        """
        try:
            import tensorflow as tf
            print(f"Exporting to TensorFlow Lite...")
        except ImportError:
            raise ImportError("TensorFlow is required for TFLite export. Install with: pip install tensorflow")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # First export to ONNX, then convert to TFLite via TensorFlow
            onnx_path = save_path.with_suffix('.onnx')
            self.export_onnx(str(onnx_path), verify=False)
            
            # Convert ONNX to TensorFlow
            try:
                import onnx_tf
                from onnx_tf.backend import prepare
                
                onnx_model = onnx.load(str(onnx_path))
                tf_rep = prepare(onnx_model)
                
                # Create TensorFlow Lite converter
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    tf_rep.signatures[tf_rep.signature_keys[0]]
                )
                
                # Apply quantization if requested
                if quantize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    
                    if representative_dataset is not None:
                        converter.representative_dataset = representative_dataset
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                        converter.inference_input_type = tf.int8
                        converter.inference_output_type = tf.int8
                
                # Convert model
                tflite_model = converter.convert()
                
                # Save model
                with open(save_path, 'wb') as f:
                    f.write(tflite_model)
                
                # Clean up temporary ONNX file
                onnx_path.unlink()
                
                print(f"TensorFlow Lite model saved to {save_path}")
                return str(save_path)
                
            except ImportError:
                print("onnx-tensorflow is required for TFLite conversion")
                print("Install with: pip install onnx-tf")
                raise
            
        except Exception as e:
            print(f"Error exporting to TensorFlow Lite: {e}")
            raise
    
    def export_coreml(
        self,
        save_path: str,
        compute_units: str = 'ALL'
    ) -> str:
        """Export model to Core ML format (macOS/iOS).
        
        Args:
            save_path: Path to save the exported model
            compute_units: Compute units ('ALL', 'CPU_ONLY', 'CPU_AND_GPU')
            
        Returns:
            Path to saved model
        """
        try:
            import coremltools as ct
            print(f"Exporting to Core ML...")
        except ImportError:
            raise ImportError("coremltools is required for Core ML export. Install with: pip install coremltools")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create example input
            example_input = torch.randn(self.input_shape).to(self.device)
            
            # Trace the model
            traced_model = torch.jit.trace(self.model, example_input)
            
            # Convert to Core ML
            model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=self.input_shape)],
                compute_units=getattr(ct.ComputeUnit, compute_units),
                source='pytorch'
            )
            
            # Set metadata
            model.short_description = "Sign Language Detection Model"
            model.input_description['input'] = "Sign language gesture image"
            model.output_description['output'] = "Class probabilities"
            
            # Add class labels
            classifier_config = ct.ClassifierConfig(self.class_names)
            model = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=16
            )
            
            # Save model
            model.save(str(save_path))
            
            print(f"Core ML model saved to {save_path}")
            return str(save_path)
            
        except Exception as e:
            print(f"Error exporting to Core ML: {e}")
            raise
    
    def create_representative_dataset(
        self,
        data_loader,
        num_samples: int = 100
    ) -> callable:
        """Create representative dataset for TFLite quantization.
        
        Args:
            data_loader: DataLoader with calibration data
            num_samples: Number of samples to use
            
        Returns:
            Representative dataset function
        """
        representative_data = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader):
                if i >= num_samples:
                    break
                
                # Convert to numpy and normalize
                images_np = images.cpu().numpy().astype(np.float32)
                representative_data.extend(images_np)
        
        def representative_dataset_gen():
            for data in representative_data:
                yield [data.reshape(1, *data.shape)]
        
        return representative_dataset_gen
    
    def benchmark_exported_models(
        self,
        model_paths: Dict[str, str],
        num_iterations: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark exported models.
        
        Args:
            model_paths: Dictionary mapping format names to model paths
            num_iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results for each format
        """
        print("Benchmarking exported models...")
        
        results = {}
        example_input = torch.randn(self.input_shape).cpu().numpy()
        
        # Benchmark original PyTorch model
        if 'pytorch' not in model_paths:
            results['pytorch'] = self._benchmark_pytorch_model(num_iterations)
        
        # Benchmark ONNX model
        if 'onnx' in model_paths:
            results['onnx'] = self._benchmark_onnx_model(
                model_paths['onnx'], example_input, num_iterations
            )
        
        # Benchmark TensorFlow Lite model
        if 'tflite' in model_paths:
            results['tflite'] = self._benchmark_tflite_model(
                model_paths['tflite'], example_input, num_iterations
            )
        
        # Print comparison
        print("\nExported Model Benchmark Results:")
        for format_name, result in results.items():
            print(f"  {format_name:10}: {result['avg_time_ms']:.2f} ms "
                  f"(size: {result['model_size_mb']:.1f} MB)")
        
        return results
    
    def _benchmark_pytorch_model(self, num_iterations: int) -> Dict[str, Any]:
        """Benchmark original PyTorch model."""
        import time
        
        example_input = torch.randn(self.input_shape).to(self.device)
        times = []
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(example_input)
        
        # Benchmark
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(example_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate model size
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        return {
            'avg_time_ms': np.mean(times) * 1000,
            'model_size_mb': model_size / (1024 * 1024)
        }
    
    def _benchmark_onnx_model(
        self,
        model_path: str,
        example_input: np.ndarray,
        num_iterations: int
    ) -> Dict[str, Any]:
        """Benchmark ONNX model."""
        import time
        
        # Load ONNX model
        ort_session = ort.InferenceSession(model_path)
        input_name = ort_session.get_inputs()[0].name
        
        times = []
        
        # Warmup
        for _ in range(10):
            _ = ort_session.run(None, {input_name: example_input})
        
        # Benchmark
        for _ in range(num_iterations):
            start_time = time.time()
            _ = ort_session.run(None, {input_name: example_input})
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Get model size
        model_size = Path(model_path).stat().st_size
        
        return {
            'avg_time_ms': np.mean(times) * 1000,
            'model_size_mb': model_size / (1024 * 1024)
        }
    
    def _benchmark_tflite_model(
        self,
        model_path: str,
        example_input: np.ndarray,
        num_iterations: int
    ) -> Dict[str, Any]:
        """Benchmark TensorFlow Lite model."""
        try:
            import tensorflow as tf
            import time
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            times = []
            
            # Warmup
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], example_input)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
            
            # Benchmark
            for _ in range(num_iterations):
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], example_input)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Get model size
            model_size = Path(model_path).stat().st_size
            
            return {
                'avg_time_ms': np.mean(times) * 1000,
                'model_size_mb': model_size / (1024 * 1024)
            }
            
        except ImportError:
            return {'error': 'TensorFlow not available for benchmarking'}


def export_model(
    model: BaseSignLanguageModel,
    class_names: List[str],
    export_formats: List[str],
    output_dir: str,
    **kwargs
) -> Dict[str, str]:
    """Export model to multiple formats.
    
    Args:
        model: Model to export
        class_names: List of class names
        export_formats: List of formats to export to
        output_dir: Directory to save exported models
        **kwargs: Additional arguments for export functions
        
    Returns:
        Dictionary mapping formats to output paths
    """
    exporter = ModelExporter(model, class_names, **kwargs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_models = {}
    
    for fmt in export_formats:
        try:
            if fmt.lower() == 'torchscript':
                path = exporter.export_torchscript(
                    str(output_dir / 'model.pt'),
                    **kwargs.get('torchscript', {})
                )
                exported_models['torchscript'] = path
            
            elif fmt.lower() == 'onnx':
                path = exporter.export_onnx(
                    str(output_dir / 'model.onnx'),
                    **kwargs.get('onnx', {})
                )
                exported_models['onnx'] = path
            
            elif fmt.lower() == 'tflite':
                path = exporter.export_tensorflow_lite(
                    str(output_dir / 'model.tflite'),
                    **kwargs.get('tflite', {})
                )
                exported_models['tflite'] = path
            
            elif fmt.lower() == 'coreml':
                path = exporter.export_coreml(
                    str(output_dir / 'model.mlmodel'),
                    **kwargs.get('coreml', {})
                )
                exported_models['coreml'] = path
            
            else:
                print(f"Unknown export format: {fmt}")
        
        except Exception as e:
            print(f"Failed to export to {fmt}: {e}")
    
    return exported_models
