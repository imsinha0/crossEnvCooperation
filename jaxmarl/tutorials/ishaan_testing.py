import numpy as np
import jax

print("NumPy version:", np.__version__)
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
print("CUDA available:", len(jax.devices('gpu')) > 0)

if len(jax.devices('gpu')) > 0:
    print("✓ SUCCESS: GPU backend is working!")
    # Test a simple GPU operation
    x = jax.numpy.array([1.0, 2.0, 3.0])
    print(f"Test array on device: {x.device()}")
else:
    print("⚠ Still using CPU backend - cuDNN may not be detected")
