# GPU Data Accelerator

A demonstration project showcasing GPU-accelerated data processing using RAPIDS cuDF compared to traditional CPU-based pandas operations.

## Project Overview

This project benchmarks the performance difference between CPU-based data processing (pandas) and GPU-accelerated data processing (cuDF) on large datasets. Using a dataset of 15+ million payment records, the benchmark demonstrates significant performance improvements when leveraging GPU acceleration for data analytics workloads.

## Key Results

- **CPU Processing (Pandas)**: 45.79 seconds on 15.3M rows
- **GPU Processing (cuDF)**: 6.97 seconds on 2M sample
- **Performance Improvement**: 6.57x faster with GPU acceleration

## Features

- **Automated Benchmarking**: Compare pandas vs cuDF performance on identical operations
- **Memory Management**: Intelligent handling of large datasets to avoid GPU memory limitations
- **Real-world Operations**: Filter, groupby aggregation, and sorting operations on financial data
- **Detailed Metrics**: Row counts, processing times, and speedup calculations

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

1. **Data Loading**: Read CSV files with appropriate data type handling
2. **Filtering**: Select records above payment threshold ($1,000)
3. **Aggregation**: Group by state and payment type with sum, mean, count
4. **Sorting**: Order results by total payment amount

### Memory Management

- **CPU**: Processes full dataset (15.3M rows)
- **GPU**: Uses intelligent sampling (2M rows) to avoid memory overflow
- **Production**: Would implement streaming or multi-GPU approaches

### Dependencies

- **cudf-cu12**: GPU-accelerated DataFrame operations
- **cuml-cu12**: GPU-accelerated machine learning
- **dask-cudf-cu12**: Distributed GPU computing
- **pandas**: CPU baseline comparison
- **cupy**: GPU array operations

## Performance Notes

- First GPU run includes JIT compilation overhead
- GPU memory limitations require dataset sampling for this demo
- Production implementations would use distributed computing for larger datasets
- Results vary based on GPU model and available memory

## Contributing

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