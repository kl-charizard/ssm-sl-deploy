"""Model quantization utilities for mobile deployment."""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..models.base_model import BaseSignLanguageModel
from ..utils.config import config


class QuantizableModel(nn.Module):
    """Wrapper to make any model quantizable."""
    
    def __init__(self, model: BaseSignLanguageModel):
        """Initialize quantizable wrapper.
        
        Args:
            model: Base model to make quantizable
        """
        super().__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization stubs."""
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class ModelQuantizer:
    """Model quantization toolkit."""
    
    def __init__(
        self,
        model: BaseSignLanguageModel,
        device: Optional[torch.device] = None
    ):
        """Initialize model quantizer.
        
        Args:
            model: Model to quantize
            device: Device to run quantization on
        """
        self.original_model = model
        self.device = device or config.device
        self.quantized_model = None
        self.calibration_loader = None
        
        # Supported quantization backends
        self.supported_backends = ['fbgemm', 'qnnpack']
        
        print(f"Model quantizer initialized")
        print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def prepare_for_quantization(
        self,
        quantization_method: str = 'dynamic',
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """Prepare model for quantization.
        
        Args:
            quantization_method: 'dynamic', 'static', or 'qat'
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            
        Returns:
            Prepared model
        """
        # Set quantization backend
        if backend in self.supported_backends:
            torch.backends.quantized.engine = backend
            print(f"Using quantization backend: {backend}")
        else:
            print(f"Unsupported backend {backend}, using default")
        
        # Wrap model to make it quantizable
        wrapped_model = QuantizableModel(self.original_model)
        wrapped_model.eval()
        
        if quantization_method == 'dynamic':
            # Dynamic quantization - no preparation needed
            return wrapped_model
        
        elif quantization_method == 'static':
            # Static quantization
            wrapped_model.qconfig = torch.quantization.get_default_qconfig(backend)
            prepared_model = torch.quantization.prepare(wrapped_model)
            return prepared_model
        
        elif quantization_method == 'qat':
            # Quantization-aware training
            wrapped_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
            prepared_model = torch.quantization.prepare_qat(wrapped_model)
            return prepared_model
        
        else:
            raise ValueError(f"Unsupported quantization method: {quantization_method}")
    
    def dynamic_quantization(
        self,
        dtype: torch.dtype = torch.qint8,
        qconfig_spec: Optional[Dict] = None
    ) -> nn.Module:
        """Apply dynamic quantization.
        
        Args:
            dtype: Quantization data type
            qconfig_spec: Custom quantization configuration
            
        Returns:
            Dynamically quantized model
        """
        print("Applying dynamic quantization...")
        
        # Default quantization spec for linear layers
        if qconfig_spec is None:
            qconfig_spec = {
                torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
                torch.nn.Conv2d: torch.quantization.default_dynamic_qconfig,
            }
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.original_model,
            qconfig_spec=qconfig_spec,
            dtype=dtype
        )
        
        self.quantized_model = quantized_model
        print("Dynamic quantization completed")
        
        return quantized_model
    
    def static_quantization(
        self,
        calibration_loader,
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """Apply static quantization with calibration.
        
        Args:
            calibration_loader: DataLoader for calibration
            backend: Quantization backend
            
        Returns:
            Statically quantized model
        """
        print("Applying static quantization...")
        
        # Prepare model
        prepared_model = self.prepare_for_quantization('static', backend)
        
        # Calibration phase
        print("Running calibration...")
        prepared_model.eval()
        
        with torch.no_grad():
            for i, (images, _) in enumerate(calibration_loader):
                if i >= 100:  # Limit calibration samples
                    break
                
                images = images.to(self.device)
                prepared_model(images)
                
                if (i + 1) % 20 == 0:
                    print(f"  Calibrated on {i + 1} batches")
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        self.quantized_model = quantized_model
        print("Static quantization completed")
        
        return quantized_model
    
    def quantization_aware_training(
        self,
        train_loader,
        num_epochs: int = 5,
        learning_rate: float = 0.0001,
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """Apply quantization-aware training.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of QAT epochs
            learning_rate: Learning rate for QAT
            backend: Quantization backend
            
        Returns:
            QAT quantized model
        """
        print("Applying quantization-aware training...")
        
        # Prepare model for QAT
        prepared_model = self.prepare_for_quantization('qat', backend)
        prepared_model.to(self.device)
        prepared_model.train()
        
        # Setup optimizer and criterion
        optimizer = torch.optim.Adam(prepared_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # QAT training loop
        for epoch in range(num_epochs):
            print(f"QAT Epoch {epoch + 1}/{num_epochs}")
            
            for i, (images, targets) in enumerate(train_loader):
                if i >= 100:  # Limit training batches for QAT
                    break
                
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = prepared_model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                if (i + 1) % 20 == 0:
                    print(f"  Batch {i + 1}, Loss: {loss.item():.4f}")
        
        # Convert to quantized model
        prepared_model.eval()
        quantized_model = torch.quantization.convert(prepared_model)
        
        self.quantized_model = quantized_model
        print("Quantization-aware training completed")
        
        return quantized_model
    
    def benchmark_quantized_model(
        self,
        test_loader,
        num_batches: int = 50
    ) -> Dict[str, Any]:
        """Benchmark quantized vs original model.
        
        Args:
            test_loader: Test data loader
            num_batches: Number of batches to benchmark
            
        Returns:
            Benchmark results
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantization first.")
        
        print("Benchmarking quantized model...")
        
        def benchmark_model(model, name):
            model.eval()
            times = []
            correct = 0
            total = 0
            
            with torch.no_grad():
                for i, (images, targets) in enumerate(test_loader):
                    if i >= num_batches:
                        break
                    
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    start_time = time.time()
                    outputs = model(images)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            accuracy = 100 * correct / total if total > 0 else 0
            avg_time = np.mean(times) * 1000  # Convert to ms
            
            return {
                'accuracy': accuracy,
                'avg_inference_time_ms': avg_time,
                'total_samples': total
            }
        
        # Benchmark original model
        original_results = benchmark_model(self.original_model, "Original")
        
        # Benchmark quantized model
        quantized_results = benchmark_model(self.quantized_model, "Quantized")
        
        # Calculate improvements
        speedup = original_results['avg_inference_time_ms'] / quantized_results['avg_inference_time_ms']
        accuracy_drop = original_results['accuracy'] - quantized_results['accuracy']
        
        # Get model sizes
        original_size = self._get_model_size(self.original_model)
        quantized_size = self._get_model_size(self.quantized_model)
        size_reduction = (original_size - quantized_size) / original_size * 100
        
        results = {
            'original': original_results,
            'quantized': quantized_results,
            'speedup': speedup,
            'accuracy_drop': accuracy_drop,
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'size_reduction_percent': size_reduction
        }
        
        print("\nQuantization Benchmark Results:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Accuracy drop: {accuracy_drop:.2f}%")
        print(f"  Size reduction: {size_reduction:.1f}%")
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def save_quantized_model(self, save_path: str):
        """Save quantized model.
        
        Args:
            save_path: Path to save the quantized model
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model to save")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with torch.jit.script for better mobile compatibility
        try:
            scripted_model = torch.jit.script(self.quantized_model)
            scripted_model.save(str(save_path))
            print(f"Quantized model saved to {save_path}")
        except Exception as e:
            # Fallback to regular torch.save
            torch.save(self.quantized_model.state_dict(), str(save_path))
            print(f"Quantized model state dict saved to {save_path}")
            print(f"Note: Could not script model ({e}), saved state dict instead")


def quantize_model(
    model: BaseSignLanguageModel,
    method: str = 'dynamic',
    calibration_loader = None,
    **kwargs
) -> nn.Module:
    """Quick quantization function.
    
    Args:
        model: Model to quantize
        method: Quantization method ('dynamic', 'static', 'qat')
        calibration_loader: Calibration data loader (for static/qat)
        **kwargs: Additional arguments
        
    Returns:
        Quantized model
    """
    quantizer = ModelQuantizer(model)
    
    if method == 'dynamic':
        return quantizer.dynamic_quantization(**kwargs)
    elif method == 'static':
        if calibration_loader is None:
            raise ValueError("Calibration loader required for static quantization")
        return quantizer.static_quantization(calibration_loader, **kwargs)
    elif method == 'qat':
        if calibration_loader is None:
            raise ValueError("Training loader required for QAT")
        return quantizer.quantization_aware_training(calibration_loader, **kwargs)
    else:
        raise ValueError(f"Unknown quantization method: {method}")


class TensorRTOptimizer:
    """TensorRT optimization for NVIDIA GPUs."""
    
    def __init__(self, model: BaseSignLanguageModel):
        """Initialize TensorRT optimizer.
        
        Args:
            model: Model to optimize
        """
        self.model = model
        self.optimized_model = None
        
        # Check if TensorRT is available
        try:
            import torch_tensorrt
            self.tensorrt_available = True
            print("TensorRT is available")
        except ImportError:
            self.tensorrt_available = False
            print("TensorRT not available - install torch-tensorrt for GPU optimization")
    
    def optimize(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        precision: str = 'fp16'
    ) -> Optional[torch.jit.ScriptModule]:
        """Optimize model with TensorRT.
        
        Args:
            input_shape: Expected input shape
            precision: Precision mode ('fp32', 'fp16', 'int8')
            
        Returns:
            TensorRT optimized model or None if not available
        """
        if not self.tensorrt_available:
            return None
        
        import torch_tensorrt
        
        print(f"Optimizing model with TensorRT ({precision})...")
        
        # Prepare model
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Create example input
        example_input = torch.randn(input_shape).to(device)
        
        # Script the model
        scripted_model = torch.jit.script(self.model)
        
        # TensorRT compilation
        optimized_model = torch_tensorrt.compile(
            scripted_model,
            inputs=[torch_tensorrt.Input(input_shape)],
            enabled_precisions={
                torch.float32 if precision == 'fp32' else
                torch.half if precision == 'fp16' else
                torch.int8
            }
        )
        
        self.optimized_model = optimized_model
        print("TensorRT optimization completed")
        
        return optimized_model
    
    def benchmark_tensorrt(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark TensorRT vs original model.
        
        Args:
            input_shape: Input shape for benchmarking
            num_iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        if not self.tensorrt_available or self.optimized_model is None:
            return {}
        
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        def benchmark(model, name):
            times = []
            model.eval()
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)
            
            return np.mean(times) * 1000  # Convert to ms
        
        original_time = benchmark(self.model, "Original")
        tensorrt_time = benchmark(self.optimized_model, "TensorRT")
        
        speedup = original_time / tensorrt_time
        
        results = {
            'original_time_ms': original_time,
            'tensorrt_time_ms': tensorrt_time,
            'speedup': speedup
        }
        
        print(f"TensorRT Benchmark Results:")
        print(f"  Original: {original_time:.2f} ms")
        print(f"  TensorRT: {tensorrt_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        return results
