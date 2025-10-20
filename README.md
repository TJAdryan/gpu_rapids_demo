# GPU Data Accelerator

A benchmarking project that compares CPU-based pandas operations with GPU-accelerated cuDF processing, revealing important insights about I/O vs computation bottlenecks in data processing workflows.

## Project Overview

This project benchmarks pandas vs cuDF performance on a large payment dataset (15+ million rows). The results revealed that for simple operations like filtering and basic aggregation, file I/O dominates processing time, making GPU acceleration benefits less apparent than expected.

## Key Findings

**Performance Breakdown:**
- **CPU Processing**: 45.79 seconds total (44s I/O + 0.1s computation) on 15.3M rows
- **GPU Processing**: 6.97 seconds total on 2M sample (mixed I/O + computation + transfer overhead)

**Critical Insights:**
- **I/O is the bottleneck**: 96% of CPU time spent reading files, not computing
- **Simple operations are fast**: Filtering 15M rows takes only 0.1 seconds on CPU
- **GPU overhead exists**: Memory transfers and setup costs impact simple operations
- **Benchmark design matters**: Fair comparisons require identical datasets and isolated operations

## Features

- **Realistic Benchmarking**: Tests common data operations (filter, groupby, sort) on real datasets
- **Performance Analysis**: Separates I/O time from computation time for accurate assessment
- **Memory Management**: Demonstrates GPU memory constraints and sampling strategies
- **Honest Evaluation**: Reports actual findings rather than marketing claims

## Requirements

### Hardware
- NVIDIA GPU with CUDA support
- Minimum 8GB GPU memory recommended
- CUDA 12.0 or higher

### Software
- Python 3.10+
- NVIDIA GPU drivers
- CUDA toolkit (for cuML functionality)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd gpu-data-accelerator
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Verify GPU setup:**
   ```bash
   nvidia-smi
   ```

## Usage

### Running the Benchmark

Execute the GPU vs CPU performance comparison:

```bash
uv run python gpu_benchmark.py
```

### Data Preparation

If you need to download the dataset:

```bash
uv run python get_kaggle_dataset.py
```

Note: Requires Kaggle API credentials configured.

### Interactive Analysis

Launch Jupyter for exploratory analysis:

```bash
uv run jupyter notebook
```

## Project Structure

```
gpu-data-accelerator/
├── gpu_benchmark.py          # Main benchmark script
├── get_kaggle_dataset.py     # Data download utility
├── test_cuda.ipynb          # GPU testing notebook
├── pyproject.toml           # Project dependencies
├── data/                    # Dataset directory
└── README.md               # This file
```

## Technical Details

### Benchmark Operations

The performance test includes common data science operations:

1. **Data Loading**: Reading large CSV files from disk
2. **Filtering**: Selecting records above payment threshold ($1,000)
3. **Aggregation**: Group by state and payment type with sum, mean, count
4. **Sorting**: Order results by total payment amount

**Important Note**: The current benchmark includes file I/O time, which dominates computation time for simple operations. Pure computational differences are much smaller than total execution time suggests.

### Memory Management

- **CPU**: Processes full dataset (15.3M rows) with standard pandas operations
- **GPU**: Uses sampling (2M rows) to avoid memory overflow on consumer GPUs
- **Limitation**: Different dataset sizes make direct performance comparison problematic
- **Learning**: Production GPU workflows require careful memory management strategies

### Dependencies

- **cudf-cu12**: GPU-accelerated DataFrame operations
- **cuml-cu12**: GPU-accelerated machine learning
- **dask-cudf-cu12**: Distributed GPU computing
- **pandas**: CPU baseline comparison
- **cupy**: GPU array operations

## Performance Notes

- **I/O Dominance**: File reading accounts for ~96% of total processing time
- **CPU Efficiency**: Simple operations (filtering, grouping) are already very fast on modern CPUs
- **GPU Overhead**: Memory transfers and setup costs can exceed benefits for simple operations
- **Dataset Size Impact**: GPU benefits emerge with larger datasets and more complex operations
- **Benchmark Limitations**: Current implementation compares different dataset sizes, limiting conclusions

## Lessons Learned

1. **I/O bottlenecks dominate** simple data processing workflows
2. **GPU acceleration benefits** depend heavily on operation complexity and dataset characteristics
3. **Fair benchmarking requires** identical datasets and isolated computational operations
4. **Memory management is critical** for GPU workflows with large datasets
5. **Honest evaluation** provides more value than inflated performance claims

## When GPU Acceleration Makes Sense

Based on this analysis, GPU acceleration is most beneficial for:
- **Compute-intensive operations** where processing time exceeds I/O time
- **Complex mathematical operations** (matrix operations, deep learning, scientific computing)
- **Iterative workflows** where data can remain in GPU memory between operations
- **Large datasets already in memory** where I/O is not the primary bottleneck

## Contributing

Contributions welcome! Particularly interested in:
- **Improved benchmark design** that isolates computational operations
- **Better memory management** strategies for large datasets
- **Additional operation types** that better showcase GPU advantages
- **Performance analysis tools** for more detailed profiling

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with your GPU setup
5. Submit a pull request

## License

This project is for educational and demonstration purposes.

## Author

Dominick Ryan - dominick@nextvaldata.com

## Acknowledgments

- NVIDIA RAPIDS team for cuDF and ecosystem
- Kaggle for providing the dataset
- Open source data science community