"""SirenCodec training modules.

Legacy MLX modules remain importable when MLX is installed. The active trainer uses
``train_cuda_main`` / ``torch_codec``.
"""

__all__ = ["train_cuda_main", "torch_codec"]
