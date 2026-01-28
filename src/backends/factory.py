"""
Backend Factory - Registry and factory for backend creation.

Provides a centralized registry for available backends and factory methods
for creating backend instances from configuration.
"""

from typing import Dict, Type, List
import logging

from .protocol import BackendProtocol, BackendConfig

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Registry for available ML framework backends.

    Backends register themselves with this class, and can be instantiated
    via the factory method based on configuration.
    """

    _backends: Dict[str, Type[BackendProtocol]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[BackendProtocol]) -> None:
        """
        Register a backend implementation.

        Args:
            name: Backend name (e.g., "pytorch", "mlx")
            backend_class: Backend class implementing BackendProtocol

        Raises:
            ValueError: If backend name already registered
        """
        if name in cls._backends:
            logger.warning(
                f"Backend '{name}' already registered. Overwriting with {backend_class}"
            )

        cls._backends[name] = backend_class
        logger.info(f"Registered backend: {name} -> {backend_class.__name__}")

    @classmethod
    def create(cls, config: BackendConfig) -> BackendProtocol:
        """
        Create a backend instance from configuration.

        Args:
            config: Backend configuration

        Returns:
            Initialized backend instance

        Raises:
            ValueError: If backend type not found in registry
        """
        backend_type = config.backend_type.lower()

        if backend_type not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Unknown backend: {backend_type}. "
                f"Available backends: {available}"
            )

        backend_class = cls._backends[backend_type]
        logger.info(f"Creating {backend_type} backend with config: {config}")

        return backend_class(config)

    @classmethod
    def list_backends(cls) -> List[str]:
        """
        List all registered backends.

        Returns:
            List of backend names
        """
        return list(cls._backends.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if backend is registered.

        Args:
            name: Backend name to check

        Returns:
            True if backend is registered
        """
        return name.lower() in cls._backends

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a backend (mainly for testing).

        Args:
            name: Backend name to unregister
        """
        if name in cls._backends:
            del cls._backends[name]
            logger.info(f"Unregistered backend: {name}")


def create_backend(
    backend_type: str,
    device: str = "auto",
    dtype: str = "float16",
    quantization: str = None,
    seed: int = 42,
    **kwargs,
) -> BackendProtocol:
    """
    Convenience function to create a backend with simplified interface.

    Args:
        backend_type: Backend name ("pytorch" or "mlx")
        device: Device to use ("auto", "mps", "cuda", "cpu", "gpu")
        dtype: Data type ("float16", "float32", "bfloat16")
        quantization: Quantization type ("int4", "int8", or None)
        seed: Random seed
        **kwargs: Additional config parameters

    Returns:
        Initialized backend instance

    Example:
        >>> backend = create_backend("mlx", device="mps")
        >>> model = backend.load_model("Qwen/Qwen2.5-7B-Instruct")
        >>> # With quantization
        >>> backend = create_backend("pytorch", quantization="int4")
        >>> model = backend.load_model("Qwen/Qwen2.5-72B-Instruct")
    """
    config = BackendConfig(
        backend_type=backend_type,
        device=device,
        dtype=dtype,
        quantization=quantization,
        seed=seed,
    )

    return BackendRegistry.create(config)


def register_backend(name: str):
    """
    Decorator for registering backend classes.

    Args:
        name: Backend name to register

    Example:
        @register_backend("pytorch")
        class PyTorchBackend:
            ...
    """

    def decorator(cls):
        BackendRegistry.register(name, cls)
        return cls

    return decorator
