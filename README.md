## Experiment Setup
This repository contains CUDA kernels and experiments used for GPU performance and energy measurements.

Hardware & GPU architectures used
- Tesla T4 (Turing architecture, compute capability ~7.5)
- Tesla V100 (Volta architecture, compute capability ~7.0)

Default experiment parameters (common values used in this repo)
- Number of elements: 4,194,304 (4M)
- Threads per block: 256
- Max block size: 1024
- Default tile size: 16
- Power sampling interval: 10 ms
- Thermal stabilization time (before measurements): 30 s

Quick build/run example (requires CUDA toolkit):
- Compile: `nvcc Codes/Final.cu -o Final.exe`
- Run: `Final.exe` (Windows) or `./Final.exe` (Unix)

