# GPU Data Accelerator

A benchmarking project that compares CPU-based pandas operations with GPU-accelerated cuDF processing, revealing important insights about I/O vs computation bottlenecks in data processing workflows. This is very narrow example to illustrate the type of performance gain I could expect on some of the data jobs that run in my pipelines.  The examples at the nvidia project https://rapids.ai/ will give a better sense of the possibilities.

## Project Overview

This project benchmarks pandas vs cuDF performance on a large payment dataset (15+ million rows). The results revealed that for simple operations like filtering and basic aggregation, file I/O dominates processing time, making GPU acceleration benefits less apparent than expected. Taking full advantage of the hardware requires altering the code to take full advantage of the GPU 

## Key Findings

**Updated Benchmark Results (gpu_benchmark.py):**

**CPU Processing (Pure Computation):**
- Dataset: 15.3 million rows (pre-loaded in memory)
- Processing time: 3.02 seconds (string operations + filtering + groupby + sorting)
- Operations: String contains, filtering, multi-key groupby aggregation, sorting
- Throughput: 5.09 million rows/second

**GPU Processing:**
-The initial numbers were terrible, with the GPU appearing slower than our optimized 3-second CPU baseline.
-The fix came down to proper architectural testing:
       Eliminating JIT Compilation Tax: Running a "warm-up" pass to prevent the first-run compilation time from skewing results.
-Addressing Memory Copy Latency: Pre-loading data directly into VRAM outside the timer.By accounting for these prerequisites, the GPU's true parallel power was unleashed:                     HardwareExecution TimeSpeed up .03 seconds.  71x improvemeent

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

Execute the improved GPU vs CPU performance comparison:

```bash
# Updated benchmark with fair comparison design
uv run python gpu_benchmark2.py

# Original benchmark (includes I/O timing)
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
├── gpu_benchmark.py        # Improved benchmark (pure computation timing)
├── get_kaggle_dataset.py    # Data download utility
├── test_cuda.ipynb         # GPU testing notebook
├── pyproject.toml          # Project dependencies
├── data/                   # Dataset directory
└── README.md              # This file
```

## Technical Details

### Benchmark Operations

The updated performance test (gpu_benchmark2.py) includes computationally intensive operations:

1. **String Processing**: Case-insensitive text search operations on payment descriptions
2. **Filtering**: Selecting records above payment threshold ($1,000)
3. **Multi-key Aggregation**: Complex groupby operations with multiple keys and aggregation functions
4. **Sorting**: Ordering results by aggregated values

**Benchmark Design Improvements:**
- **Pre-loads data**: Dataset loaded once into memory before timing begins
- **Pure computation timing**: Excludes file I/O from performance measurements
- **Identical operations**: Same dataset size and operations for fair comparison
- **Memory management**: Explicit cleanup and GPU synchronization

### Memory Management

- **CPU**: Processes full dataset (15.3M rows) with 3.02 seconds of pure computation
- **GPU**: Encounters memory limitations during string processing operations
- **Challenge**: Accounting for warm up operations and altering code for GPU optimized code
- **Solution**: Workloads on GPU can provide significant advantages with updated code.  

### Dependencies

- **cudf-cu12**: GPU-accelerated DataFrame operations
- **cuml-cu12**: GPU-accelerated machine learning
- **dask-cudf-cu12**: Distributed GPU computing
- **pandas**: CPU baseline comparison
- **cupy**: GPU array operations

## Performance Notes

- **Improved benchmark design**: gpu_benchmark2.py eliminates I/O timing for fair comparison
- **CPU performance**: 5.09 million rows/second for complex operations (string processing + aggregation)
- **GPU memory constraint**: Code needs to be updated specififcally to take advantage of hardware
- **Real-world insight**: 71x over CPU perfromance possible if code base updated when GPU hardware is available
- **Benchmark evolution**: Shows importance of iterative improvement in performance analysis

## Lessons Learned

1. **Benchmark design evolution**: Moving from I/O-inclusive to computation-only timing provides clearer insights
2. **CPU performance is impressive**: Modern CPUs handle 5+ million rows/second for complex operations
3. **GPU memory is the real constraint**: Not computation speed, but memory capacity for large datasets
4. **String operations are memory-intensive**: Text processing requires significant GPU memory allocation
5. **Fair comparison matters**: Identical datasets and isolated operations reveal true performance characteristics


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
