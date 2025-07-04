# Energy-Aware CUDA Kernel Optimization Framework

## Overview

This repository hosts the code, data, and documentation for the research project **"A Systematic Performance Engineering Framework for Energy-Aware CUDA Kernel Optimization."**

Our research proposes and implements a comprehensive methodology for systematically evaluating and optimizing CUDA kernels, with a dual focus on quantifying and balancing computational performance and energy consumption in GPU-accelerated scientific applications. We demonstrate the framework's effectiveness using Dense Matrix-Matrix Multiplication (GEMM) as a representative workload.

## Features

-   **Hierarchical CUDA Kernel Implementations:** Six progressively optimized GEMM kernel variants, designed to evaluate the impact of specific optimization techniques (e.g., memory coalescing, shared memory utilization, register optimization, occupancy tuning, and comprehensive strategies).
-   **Real-time Energy & Performance Monitoring:** Integration with NVIDIA Management Library (NVML) for accurate power consumption measurements and CUDA Profiling Tools Interface (CUPTI) for detailed performance metrics.
-   **Reproducible Experimental Setup:** Scripts and configurations to replicate experiments across different NVIDIA GPU architectures (tested on Tesla T4 and Tesla V100).
-   **Systematic Evaluation:** Tools for analyzing energy–performance trade-offs and deriving actionable optimization guidelines.

## Getting Started

### Prerequisites

To run the experiments and analyze the results, you will need:

-   An NVIDIA GPU with CUDA support (e.g., Tesla T4, Tesla V100).
-   CUDA Toolkit (tested with 11.0).
-   NVIDIA Management Library (NVML), typically installed with NVIDIA drivers.
-   A Linux operating system (tested on Ubuntu 20.04).
-   Python 3.x (for analysis scripts and dependencies).

### Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/<your-username>/energy-aware-cuda-optimization.git
    cd energy-aware-cuda-optimization
    ```
2.  **Install Python dependencies:**
    ```
    pip install -r requirements.txt
    ```
3.  **Ensure CUDA and NVML setup:** Verify that your CUDA Toolkit and NVIDIA drivers (including NVML) are correctly installed and configured in your system's `PATH` and `LD_LIBRARY_PATH`.

### Usage

1.  **Compile the CUDA kernels:**
    Navigate to the `kernels/` directory and compile the desired kernel(s) using `nvcc`.
    Example:
    ```
    nvcc -o gemm_baseline kernels/gemm_baseline.cu
    ```
    (A `Makefile` or `build.sh` script might be provided in the future for easier compilation of all variants).

2.  **Run experiments and collect data:**
    Use the provided scripts in the `scripts/` directory to execute kernels and log performance/energy data.
    Example:
    ```
    ./scripts/run_all_experiments.sh
    ```
    (This script will execute all kernel variants and collect data into the `data/` directory).

3.  **Analyze the results:**
    Execute the analysis scripts to process the collected data and generate insights.
    Example:
    ```
    python scripts/analyze_energy_performance.py
    ```

For detailed instructions on running specific experiments and interpreting results, please refer to the documentation within the `docs/` folder.

## Project Structure

├── kernels/ # CUDA kernel source files (baseline, optimized variants)
├── scripts/ # Python scripts for running experiments, data collection, and analysis
├── data/ # Directory to store raw experimental data (performance logs, power logs)
├── docs/ # Detailed documentation, setup guides, and analysis explanations
├── figures/ # Generated plots and visualizations
├── src/ # Supporting source files (e.g., common utility functions)
├── requirements.txt # Python package dependencies
├── .gitignore # Files and directories to ignore in Git
├── LICENSE # Project license file
└── README.md # This README file


## Citation

If you find this framework or the associated data useful for your research, please cite our paper:

@inproceedings{shete2025energyaware,
author = {Aniruddha Shete and Niraj There and Amit Joshi},
title = {A Systematic Performance Engineering Framework for Energy-Aware CUDA Kernel Optimization},
booktitle = {Proceedings of the IEEE Conference on [Your Conference Name]},
year = {2025}
}

*(Please update the `booktitle` and `year` once the paper is published)*

## Contributing

We welcome contributions to this project! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Open an [issue](https://github.com/<your-username>/energy-aware-cuda-optimization/issues) to discuss your ideas.
2.  Fork the repository and submit a [pull request](https://github.com/<your-username>/energy-aware-cuda-optimization/pulls) with your changes.

## License

This project is open-source and licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions, collaborations, or further information, please feel free to reach out to the authors:

-   Aniruddha Shete: 20220802217@dypiu.ac.in
-   Niraj There: 20220802398@dypiu.ac.in
-   Amit Joshi: adj.comp@coeptech.ac.in
