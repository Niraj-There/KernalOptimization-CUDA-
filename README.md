# CUDA Kernel Optimization

> A comprehensive collection of CUDA optimization techniques focusing on Matrix Multiplication and Parallel Reduction algorithms.

## 📚 Repository Overview

This repository demonstrates various GPU optimization techniques through practical implementations and analysis.

- **Matrix Multiplication Optimizations**
  - Baseline implementation
  - Progressive optimization steps
  - Performance analysis on Tesla T4 and V100

- **Parallel Reduction Implementations**
  - 6 optimization versions
  - Interactive Jupyter notebooks
  - Performance comparisons

- **Detailed Performance Analysis**
  - Block configuration heatmaps
  - Energy efficiency metrics
  - Bandwidth utilization charts

## 🛠️ Project Structure

```
├── MatMul-Optimizations/
│   ├── Baseline.cu              # Basic implementation
│   ├── Final.cu                 # Optimized version
│   ├── PM_fixes.cu             # Performance monitoring
│   └── Block Config/            # Performance analysis
├── Parallel-Reduction/
│   ├── reduction[0-6].cu       # Progressive optimizations
│   └── Optimization/           # Jupyter implementations
└── Reduction-Profiling/
    └── reduction[0-6]_full.ipynb
```

## 📋 Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (latest version recommended)
- Python 3.x with packages:
  - numpy
  - matplotlib
  - jupyter

## 🚀 Quick Start

### Matrix Multiplication

```bash
cd MatMul-Optimizations
nvcc Final.cu -o matmul
./matmul
```

### Parallel Reduction

```bash
cd Parallel-Reduction
nvcc reduction6.cu -o reduce
./reduce
```

## 📊 Performance Results

- Detailed analysis for Tesla T4 and V100
- Block configuration impact studies
- Energy efficiency comparisons
- Bandwidth utilization metrics

## 📁 Data Organization

- `/MatMul-Optimizations` - Core matrix multiplication implementations
- `/Parallel-Reduction` - Reduction algorithm variations
- `/Reduction-Profiling` - Detailed performance analysis
- `/Optimization-Results` - Benchmark data and results


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
