import pandas as pd
import time
import os
import cudf
import cupy as cp

# --- Configuration ---
DATA_FILE = './data/merged_open_payment.csv'

# --- 0. Pre-Load the Full Dataset (using Pandas) ---
# Load the dataset once using the robust Pandas reader into CPU RAM.
print(f"Loading full dataset into CPU memory using Pandas...")
try:
    # Use low_memory=False to help Pandas infer types correctly on large files
    FULL_PANDAS_DF = pd.read_csv(DATA_FILE, dtype={'zip_code': 'str'}, low_memory=False)
    print(f"Dataset loaded successfully. Total rows: {len(FULL_PANDAS_DF):,}")
except Exception as e:
    print(f"FATAL ERROR during Pandas load: {e}")
    exit()

# --- 1. Define the Heavy Workload Function ---
# This function measures pure processing time, using the DataFrame already loaded into memory.
def run_analytics(df_library, label, base_df=None):
    """
    Performs a heavy data cleaning and aggregation task designed to challenge the CPU.
    It takes a pre-loaded DataFrame (base_df) or loads a copy of the global Pandas DF.
    """
    print(f"\n--- Running: {label} ({df_library.__name__}) ---")
    
    start_time = time.time()
    
    # 1. DATA PREPARATION: Get the DataFrame copy
    if base_df is not None:
        # For GPU runs: Use the pre-copied GPU DataFrame
        df = base_df.copy() 
    else:
        # For CPU run: Use a copy of the CPU Pandas DataFrame
        global FULL_PANDAS_DF
        df = FULL_PANDAS_DF.copy()
        
    total_rows = len(df)
    
    # --- HIGH-PAIN STRING PROCESSING (BIGGEST GPU ADVANTAGE) ---
    # FIX: Explicitly set regex=False to satisfy cuDF and enable case=False
    df['is_research'] = df['payment_nature'].str.contains('Research', case=False, regex=False)
    
    # 2. FILTERING 
    df = df[df['payment_amount'] > 1000]
    
    # 3. COMPLEX MULTI-KEY GROUPBY AGGREGATION (GPU Advantage)
    if len(df) > 0:
        results_df = df.groupby(['address_state', 'payment_nature', 'is_research']) \
                         .agg({
                             'payment_amount': ['sum', 'mean'],
                             'payment_id': 'count' # Use 'payment_id' as the unique row counter
                         })
        
        # 4. SORTING 
        final_agg = results_df.sort_values(
            ('payment_amount', 'sum'), 
            ascending=False
        )
    else:
        final_agg = df_library.DataFrame()
        
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Clean up (Essential for GPU memory management)
    del df
    del results_df
    del final_agg
    if df_library.__name__ == 'cudf':
        # Ensure GPU memory is fully released and synchronized
        cp.cuda.runtime.deviceSynchronize() 
        
    print(f"  Processed rows: {total_rows:,}, Elapsed Time: {elapsed:.4f} seconds")
    return elapsed

# ----------------------------------------------------------------------------------

# Run 3 iterations for averaging the GPU time (removes startup noise)
ITERATIONS = 3
cpu_times = []
gpu_times = []

# --- 2. Benchmark Execution (Actual Timing Starts Here) ---

# Ensure cuDF/CUDA is ready
cp.cuda.runtime.deviceSynchronize() 

# --- A. CPU Benchmark (Standard Pandas) ---
# Call the CPU function without the 'base_df' argument
print("="*60)
cpu_time = run_analytics(pd, "CPU (Pandas - Full Run)")
cpu_times.append(cpu_time)

# --- B. GPU Preparation ---

print("\n" + "="*60)
print("GPU Preparation: Copying data from CPU RAM to 16GB VRAM...")
GPU_DF_BASE = cudf.from_pandas(FULL_PANDAS_DF) 
cp.cuda.runtime.deviceSynchronize()
print("Copy complete. Starting GPU processing benchmarks.")


print("GPU Warm-up run (JIT compilation)...")
# Run the function once and discard the time
run_analytics(cudf, "GPU Warmup", base_df=GPU_DF_BASE) 
print("Warm-up complete. Starting timed runs.")


# --- C. GPU Timed Runs ---
for i in range(ITERATIONS):
    # Pass the pre-copied GPU DataFrame to the function
    gpu_time = run_analytics(cudf, f"GPU (cuDF) - Run {i+1}", base_df=GPU_DF_BASE)
    gpu_times.append(gpu_time)

# --- 3. Print Final Results ---
avg_gpu_time = sum(gpu_times) / ITERATIONS
speedup = cpu_times[0] / avg_gpu_time

print("\n" + "="*60)
print(f"🔥 GPU ACCELERATION DEMO RESULTS (Averaged over {ITERATIONS} runs) 🔥")
print("Note: The memory copy time is excluded to benchmark pure processing speed.")
print(f"CPU Processing Time (Pandas): {cpu_times[0]:.4f} seconds")
print(f"GPU Processing Time (cuDF)  : {avg_gpu_time:.4f} seconds")
print(f"🚀 GPU SPEEDUP: {speedup:.2f}x faster!")
print("="*60)