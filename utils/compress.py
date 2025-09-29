"""
Model compression utilities for communication-efficient federated learning
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, OrderedDict
import copy


class ModelCompressor:
    """Base class for model compression techniques"""

    def __init__(self):
        self.compression_ratio = 0.0
        self.original_size = 0
        self.compressed_size = 0

    def compress(self, model_params: OrderedDict) -> Tuple[Any, Dict[str, Any]]:
        """Compress model parameters"""
        raise NotImplementedError

    def decompress(self, compressed_params: Any, metadata: Dict[str, Any]) -> OrderedDict:
        """Decompress model parameters"""
        raise NotImplementedError

    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics"""
        return {
            'compression_ratio': self.compression_ratio,
            'original_size': self.original_size,
            'compressed_size': self.compressed_size,
            'reduction_percentage': (1 - self.compression_ratio) * 100
        }


class TopKCompressor(ModelCompressor):
    """Top-K sparsification compression"""

    def __init__(self, sparsity_ratio: float = 0.1):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio  # Fraction of parameters to keep

    def compress(self, model_params: OrderedDict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compress by keeping only top-K parameters by magnitude

        Args:
            model_params: Model parameters as OrderedDict

        Returns:
            Tuple of (compressed_params, metadata)
        """

        compressed_params = {}
        metadata = {
            'shapes': {},
            'indices': {},
            'sparsity_ratio': self.sparsity_ratio
        }

        total_params = 0
        compressed_params_count = 0

        for name, param in model_params.items():
            param_tensor = param.detach().clone()
            original_shape = param_tensor.shape
            flat_param = param_tensor.flatten()

            total_params += flat_param.numel()

            # Calculate number of parameters to keep
            k = max(1, int(len(flat_param) * self.sparsity_ratio))

            # Get top-k indices by absolute value
            _, top_indices = torch.topk(torch.abs(flat_param), k)

            # Extract top-k values and indices
            top_values = flat_param[top_indices]

            compressed_params[name] = {
                'values': top_values,
                'indices': top_indices
            }

            metadata['shapes'][name] = original_shape
            compressed_params_count += k

        self.original_size = total_params
        self.compressed_size = compressed_params_count
        self.compression_ratio = compressed_params_count / total_params

        return compressed_params, metadata

    def decompress(self, compressed_params: Dict[str, Any], 
                  metadata: Dict[str, Any]) -> OrderedDict:
        """
        Decompress top-K compressed parameters

        Args:
            compressed_params: Compressed parameters
            metadata: Compression metadata

        Returns:
            Decompressed model parameters
        """

        decompressed_params = OrderedDict()

        for name, compressed_data in compressed_params.items():
            original_shape = metadata['shapes'][name]
            values = compressed_data['values']
            indices = compressed_data['indices']

            # Create sparse tensor
            flat_size = torch.prod(torch.tensor(original_shape))
            sparse_flat = torch.zeros(flat_size, dtype=values.dtype, device=values.device)
            sparse_flat[indices] = values

            # Reshape to original shape
            decompressed_param = sparse_flat.reshape(original_shape)
            decompressed_params[name] = decompressed_param

        return decompressed_params


class QuantizationCompressor(ModelCompressor):
    """Uniform quantization compression"""

    def __init__(self, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits

    def compress(self, model_params: OrderedDict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compress using uniform quantization

        Args:
            model_params: Model parameters as OrderedDict

        Returns:
            Tuple of (compressed_params, metadata)
        """

        compressed_params = {}
        metadata = {
            'shapes': {},
            'scales': {},
            'zero_points': {},
            'num_bits': self.num_bits
        }

        total_size = 0
        compressed_size = 0

        for name, param in model_params.items():
            param_tensor = param.detach().clone()
            original_shape = param_tensor.shape

            # Calculate quantization parameters
            min_val = param_tensor.min()
            max_val = param_tensor.max()

            scale = (max_val - min_val) / (self.num_levels - 1)
            zero_point = min_val

            # Quantize
            quantized = torch.round((param_tensor - zero_point) / scale)
            quantized = torch.clamp(quantized, 0, self.num_levels - 1)

            # Convert to appropriate integer type
            if self.num_bits <= 8:
                quantized = quantized.to(torch.uint8)
            elif self.num_bits <= 16:
                quantized = quantized.to(torch.int16)
            else:
                quantized = quantized.to(torch.int32)

            compressed_params[name] = quantized
            metadata['shapes'][name] = original_shape
            metadata['scales'][name] = scale
            metadata['zero_points'][name] = zero_point

            # Calculate sizes
            total_size += param_tensor.numel() * 4  # 32-bit float
            compressed_size += quantized.numel() * (self.num_bits / 8)

        self.original_size = total_size
        self.compressed_size = compressed_size
        self.compression_ratio = compressed_size / total_size

        return compressed_params, metadata

    def decompress(self, compressed_params: Dict[str, Any], 
                  metadata: Dict[str, Any]) -> OrderedDict:
        """
        Decompress quantized parameters

        Args:
            compressed_params: Compressed parameters
            metadata: Compression metadata

        Returns:
            Decompressed model parameters
        """

        decompressed_params = OrderedDict()

        for name, quantized_param in compressed_params.items():
            scale = metadata['scales'][name]
            zero_point = metadata['zero_points'][name]

            # Dequantize
            dequantized = quantized_param.to(torch.float32) * scale + zero_point

            decompressed_params[name] = dequantized

        return decompressed_params


class HybridCompressor(ModelCompressor):
    """Hybrid compression combining multiple techniques"""

    def __init__(self, topk_ratio: float = 0.1, quantization_bits: int = 8):
        super().__init__()
        self.topk_compressor = TopKCompressor(topk_ratio)
        self.quantization_compressor = QuantizationCompressor(quantization_bits)
        self.use_topk_for_large_layers = True

    def compress(self, model_params: OrderedDict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Apply hybrid compression strategy

        Args:
            model_params: Model parameters as OrderedDict

        Returns:
            Tuple of (compressed_params, metadata)
        """

        compressed_params = {}
        metadata = {'compression_methods': {}}

        total_original_size = 0
        total_compressed_size = 0

        for name, param in model_params.items():
            param_size = param.numel()
            total_original_size += param_size

            # Use Top-K for large layers (e.g., fully connected), quantization for others
            if param_size > 10000 and self.use_topk_for_large_layers:
                # Use Top-K compression
                single_param_dict = OrderedDict([(name, param)])
                comp_params, comp_metadata = self.topk_compressor.compress(single_param_dict)

                compressed_params[name] = comp_params[name]
                metadata[f'{name}_topk'] = comp_metadata
                metadata['compression_methods'][name] = 'topk'

                total_compressed_size += self.topk_compressor.compressed_size

            else:
                # Use quantization
                single_param_dict = OrderedDict([(name, param)])
                comp_params, comp_metadata = self.quantization_compressor.compress(single_param_dict)

                compressed_params[name] = comp_params[name]
                metadata[f'{name}_quant'] = comp_metadata
                metadata['compression_methods'][name] = 'quantization'

                total_compressed_size += self.quantization_compressor.compressed_size

        self.original_size = total_original_size
        self.compressed_size = total_compressed_size
        self.compression_ratio = total_compressed_size / total_original_size

        return compressed_params, metadata

    def decompress(self, compressed_params: Dict[str, Any], 
                  metadata: Dict[str, Any]) -> OrderedDict:
        """
        Decompress hybrid compressed parameters

        Args:
            compressed_params: Compressed parameters
            metadata: Compression metadata

        Returns:
            Decompressed model parameters
        """

        decompressed_params = OrderedDict()

        for name, comp_param in compressed_params.items():
            method = metadata['compression_methods'][name]

            if method == 'topk':
                single_comp_dict = {name: comp_param}
                single_metadata = metadata[f'{name}_topk']
                single_decomp = self.topk_compressor.decompress(single_comp_dict, single_metadata)
                decompressed_params[name] = single_decomp[name]

            elif method == 'quantization':
                single_comp_dict = {name: comp_param}
                single_metadata = metadata[f'{name}_quant']
                single_decomp = self.quantization_compressor.decompress(single_comp_dict, single_metadata)
                decompressed_params[name] = single_decomp[name]

        return decompressed_params


def get_compressor(compression_type: str, **kwargs) -> ModelCompressor:
    """Factory function to get compressor instance"""

    if compression_type.lower() == 'topk':
        return TopKCompressor(sparsity_ratio=kwargs.get('sparsity_ratio', 0.1))
    elif compression_type.lower() == 'quantization':
        return QuantizationCompressor(num_bits=kwargs.get('num_bits', 8))
    elif compression_type.lower() == 'hybrid':
        return HybridCompressor(
            topk_ratio=kwargs.get('topk_ratio', 0.1),
            quantization_bits=kwargs.get('quantization_bits', 8)
        )
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")


def calculate_model_size(model_params: OrderedDict) -> int:
    """Calculate model size in bytes"""
    total_size = 0
    for param in model_params.values():
        total_size += param.numel() * param.element_size()
    return total_size


def compression_analysis(original_params: OrderedDict, 
                        compressor: ModelCompressor) -> Dict[str, Any]:
    """
    Perform compression analysis

    Args:
        original_params: Original model parameters
        compressor: Compressor instance

    Returns:
        Analysis results
    """

    # Compress and decompress
    compressed_params, metadata = compressor.compress(original_params)
    decompressed_params = compressor.decompress(compressed_params, metadata)

    # Calculate reconstruction error
    total_error = 0.0
    total_norm = 0.0

    for name in original_params.keys():
        original = original_params[name]
        reconstructed = decompressed_params[name]

        error = torch.norm(original - reconstructed).item()
        norm = torch.norm(original).item()

        total_error += error ** 2
        total_norm += norm ** 2

    relative_error = np.sqrt(total_error) / np.sqrt(total_norm)

    # Get compression stats
    compression_stats = compressor.get_compression_stats()

    analysis_results = {
        **compression_stats,
        'relative_reconstruction_error': relative_error,
        'original_size_bytes': calculate_model_size(original_params),
        'compressed_size_estimate': compressor.compressed_size * 4,  # Rough estimate
        'metadata_size': len(str(metadata))
    }

    return analysis_results


if __name__ == "__main__":
    # Test compression utilities
    print("Testing compression utilities...")

    # Create a simple model for testing
    from collections import OrderedDict

    model_params = OrderedDict([
        ('layer1.weight', torch.randn(64, 32)),
        ('layer1.bias', torch.randn(64)),
        ('layer2.weight', torch.randn(10, 64)),
        ('layer2.bias', torch.randn(10))
    ])

    print(f"Original model size: {calculate_model_size(model_params)} bytes")

    # Test Top-K compression
    print("\n--- Top-K Compression ---")
    topk_compressor = TopKCompressor(sparsity_ratio=0.1)
    topk_analysis = compression_analysis(model_params, topk_compressor)
    for key, value in topk_analysis.items():
        print(f"{key}: {value}")

    # Test Quantization
    print("\n--- Quantization Compression ---")
    quant_compressor = QuantizationCompressor(num_bits=8)
    quant_analysis = compression_analysis(model_params, quant_compressor)
    for key, value in quant_analysis.items():
        print(f"{key}: {value}")

    # Test Hybrid compression
    print("\n--- Hybrid Compression ---")
    hybrid_compressor = HybridCompressor(topk_ratio=0.05, quantization_bits=8)
    hybrid_analysis = compression_analysis(model_params, hybrid_compressor)
    for key, value in hybrid_analysis.items():
        print(f"{key}: {value}")
