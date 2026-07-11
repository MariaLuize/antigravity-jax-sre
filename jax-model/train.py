import jax
import jax.numpy as jnp

def main():
    print("================ JAX Device & Workload Verification ================")
    print(f"JAX Version: {jax.__version__}")
    
    # List and display all JAX visible devices (e.g., CPU, GPU, TPU)
    devices = jax.devices()
    print(f"Available visible devices: {devices}")
    
    # Check if a GPU is actively recognized
    gpu_devices = [d for d in devices if d.device_kind.lower() == "gpu"]
    if gpu_devices:
        print(f"Success! Detected {len(gpu_devices)} GPU device(s). Using GPU for computation.")
    else:
        print("Warning: No GPU devices found. JAX is running on CPU.")

    # Matrix multiplication configuration
    matrix_size = 2000
    print(f"\nInitializing random matrices of size {matrix_size}x{matrix_size}...")
    
    # Initialize random keys for JAX PRNG
    key = jax.random.PRNGKey(42)
    key_a, key_b = jax.random.split(key, 2)
    
    # Generate random matrices on the default device
    a = jax.random.normal(key_a, (matrix_size, matrix_size), dtype=jnp.float32)
    b = jax.random.normal(key_b, (matrix_size, matrix_size), dtype=jnp.float32)

    # Simple JIT-compiled matrix multiplication function
    @jax.jit
    def run_matmul(x, y):
        return jnp.matmul(x, y)

    print("Compiling matrix multiplication (JIT)...")
    # Warm-up (compiles the function)
    result = run_matmul(a, b)
    result.block_until_ready()

    print("Executing matrix multiplication...")
    # Execute compiled function
    result = run_matmul(a, b)
    result.block_until_ready()
    
    print(f"Result matrix shape: {result.shape}")
    print("Matrix multiplication executed successfully!")
    print("====================================================================")

if __name__ == "__main__":
    main()
