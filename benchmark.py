import torch
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 50)

# Warm up (compile kernels)
print("Warming up...")
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
for _ in range(10):
    torch.matmul(a, b)
torch.cuda.synchronize() if torch.cuda.is_available() else None

# Benchmark different matrix sizes
sizes = [512, 1024, 2048, 4096]
results = []

print(f"\n{'Size':<10} {'Time (ms)':<12} {'TFLOPS':<10}")
print("-" * 35)

for size in sizes:
    # Create random matrices (FP32)
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Sync before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Time it
    start = time.perf_counter()
    iterations = 100 if size < 2048 else 20
    for _ in range(iterations):
        c = torch.matmul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = (end - start) / iterations
    # FLOPS for matmul: 2 * N^3 (multiply-adds)
    flops = 2 * (size ** 3) / elapsed
    tflops = flops / 1e12
    
    results.append((size, elapsed * 1000, tflops))
    print(f"{size}x{size:<4} {elapsed*1000:>8.2f} ms {tflops:>8.2f} TFLOPS")

# Peak performance
peak = max(r[2] for r in results)
print(f"\n🚀 PEAK PERFORMANCE: {peak:.2f} TFLOPS (FP32)")

# Comparison
print("\n" + "=" * 50)
print("REFERENCE (FP32):")
print(f"  AMD 780M iGPU (yours):    ~{peak:.1f} TFLOPS")
print(f"  NVIDIA GTX 1650 Mobile:   ~3.0 TFLOPS")
print(f"  NVIDIA RTX 3060 Laptop:   ~13.0 TFLOPS")
print(f"  Apple M3 Pro:             ~4.0 TFLOPS")
print(f"  AMD RX 6800M (dedicated): ~11.0 TFLOPS")

# Memory bandwidth test (important for iGPUs)
print("\n" + "=" * 50)
print("MEMORY BANDWIDTH TEST:")
size = 1024 * 1024 * 256  # 1GB of floats
x = torch.randn(size, device=device)
y = torch.randn(size, device=device)

if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.perf_counter()
z = x + y  # Element-wise add (memory bound)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

bandwidth = (3 * size * 4) / (elapsed * 1e9)  # GB/s (read x, read y, write z)
print(f"  Bandwidth: {bandwidth:.1f} GB/s")
print(f"  (DDR5 shared memory - typical for iGPU: 50-100 GB/s)")

# Your training speed context
print("\n" + "=" * 50)
print("YOUR TRAINING CONTEXT:")
print(f"  Model params: ~18k")
print(f"  Batch size: 128")
print(f"  Steps/sec: ~2000")
print(f"  This is INSANE for an iGPU!")
