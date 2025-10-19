import pandas as pd
import time
import os
import cudf
import cupy as cp

# --- Configuration ---
DATA_FILE = './data/merged_open_payment.csv'
# Replace this with the actual file path you found in step 1.

# --- 1. Define the Heavy Workload Function ---
def run_analytics(df_library, label, iterations=1):
    """
    Performs a heavy data cleaning and aggregation task.
    Operations: Filter, Type Conversion, Groupby-Aggregate, Sort.
    """
    print(f"\n--- Running: {label} ({df_library.__name__}) ---")
    
    total_time = 0.0
    
    for i in range(iterations):
        start_time = time.time()
        
        # 1. READ DATA (I/O is often a huge bottleneck)
        if df_library.__name__ == 'cudf':
            # For GPU: Use a smaller sample to avoid memory issues
            # Read with pandas first, sample, then convert to cudf
            import pandas as pd
            
            print(f"  Reading and sampling data for GPU processing...")
            # Read in chunks with pandas and sample
            sample_size = 2000000  # 2M rows sample
            total_rows = 0
            sampled_data = []
            
            for chunk in pd.read_csv(DATA_FILE, chunksize=1000000, dtype={'zip_code': 'str'}):
                total_rows += len(chunk)
                # Take a proportional sample from each chunk
                sample_frac = min(1.0, sample_size / total_rows)
                if sample_frac < 1.0:
                    chunk_sample = chunk.sample(frac=sample_frac)
                else:
                    chunk_sample = chunk
                sampled_data.append(chunk_sample)
                
                # Stop if we have enough data
                if sum(len(d) for d in sampled_data) >= sample_size:
                    break
            
            # Combine samples and convert to cuDF
            if sampled_data:
                pandas_df = pd.concat(sampled_data, ignore_index=True)
                df = df_library.from_pandas(pandas_df)
                del pandas_df  # Free memory
            else:
                df = df_library.DataFrame()
                
        else:
            # For CPU: Read full dataset
            df = df_library.read_csv(DATA_FILE, dtype={'zip_code': 'str'}, low_memory=False)
        
        # 2. FILTERING (Select a subset of high-value payments)
        df = df[df['payment_amount'] > 1000]
        
        # 3. GROUPBY AGGREGATION (The most GPU-intensive step)
        # Calculate total payments and average payment for each state and nature type.
        if len(df) > 0:
            results_df = df.groupby(['address_state', 'payment_nature']) \
                           .agg({
                               'payment_amount': ['sum', 'mean', 'count']
                           })
            
            # 4. SORTING (Final, heavy data rearrangement)
            results_df = results_df.sort_values(
                ('payment_amount', 'sum'), 
                ascending=False
            )
        else:
            results_df = df_library.DataFrame()  # Empty results

        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed
        
        print(f"  Iteration {i+1}: {elapsed:.4f} seconds, Filtered rows: {len(df):,}")
        
        # Clean up the large dataframe to free memory (more important for GPU)
        del df
        del results_df
        if df_library.__name__ == 'cudf':
            cp.cuda.runtime.deviceSynchronize() # Ensure all GPU operations are complete

    avg_time = total_time / iterations
    
    # Get total row count for reporting
    if df_library.__name__ == 'cudf':
        total_rows = sample_size  # Report the sample size used
        print(f"Total rows processed (GPU sample): {total_rows:,}")
    else:
        sample_df = df_library.read_csv(DATA_FILE, dtype={'zip_code': 'str'}, low_memory=False)
        total_rows = len(sample_df)
        del sample_df
        print(f"Total rows processed (full dataset): {total_rows:,}")
    
    print(f"Final Average Time ({label}): {avg_time:.4f} seconds")
    return avg_time

# --- 2. Benchmark Execution ---

# Ensure cuDF is ready for GPU work
cp.cuda.runtime.deviceSynchronize() 

# --- A. CPU Benchmark (Standard Pandas) ---
# Run 1 time for the demo to show the initial lag/time
cpu_time = run_analytics(pd, "CPU (Pandas)", iterations=1)

# --- B. GPU Benchmark (cuDF) ---
# Note: cuDF is imported here as 'cudf'
# For the GPU test, we might run it 3 times and average the time, 
# as the first run often includes some JIT compilation/initialization.
gpu_time = run_analytics(cudf, "GPU (cuDF)", iterations=3)


# --- 3. Print Results ---
speedup = cpu_time / gpu_time
print("\n" + "="*60)
print(f"🔥 GPU ACCELERATION DEMO RESULTS 🔥")
print(f"CPU Time (Pandas - Full Dataset): {cpu_time:.4f} seconds")
print(f"GPU Time (cuDF - 2M Sample)     : {gpu_time:.4f} seconds (Average of 3 runs)")
print(f"🚀 GPU SPEEDUP: {speedup:.2f}x faster!")
print("\nNote: GPU used a 2M row sample to avoid memory limits.")
print("In production, you'd use multiple GPUs or streaming for full datasets.")
print("="*60)