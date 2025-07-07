import os
import time
import logging
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import griddata
import polars as pl
import h5py
from joblib import Parallel, delayed

# --- 1. Robust Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

# --- 2. Configuration ---
segments = [
    ('sagittal_anterior', 28, 27, -999, 10000),
    ('tangential_anterior', 60, 27, -999, 10000),
    ('gaussian_anterior', 92, 22, -999, 10000),
    ('sagittal_posterior', 124, 26, -999, 10000),
    ('tangential_posterior', 156, 26, -999, 10000),
    ('gaussian_posterior', 188, 22, -999, 10000),
    ('refra_frontal_power_anterior', 220, 27, -999, 10000),
    ('refra_frontal_power_posterior', 252, 26, -999, 10000),
    ('refra_equivalent_power', 284, 23, -999, 10000),
    ('corneal_thickness', 412, 26, -999, 10000),
    ('stromal_thickness', 444, 25, -999, 10000),
    ('epithelial_thickness', 476, 25, -999, 10000),
    ('anterior_chamber_depth', 508, 26, -999, 10000)
]

# --- 3. Optimized Helper Functions ---

def read_all_segments_fast(file_path, segments_config):
    """Reads all required segments from a single CSV in one pass using Polars."""
    all_data = {}
    for name, start_row, num_rows, _, _ in segments_config:
        try:
            df = pl.read_csv(
                file_path, separator=';', has_header=False,
                skip_rows=start_row, n_rows=num_rows, use_pyarrow=True,
                new_columns=[f"col_{i}" for i in range(256)]
            )
            all_data[name] = df.to_numpy(writable=True)
        except Exception as e:
            logging.error(f"Polars failed reading segment {name} from {os.path.basename(file_path)}: {e}")
            all_data[name] = None
    return all_data

def save_to_hdf_fast(results, output_file):
    """Saves all numpy arrays to a single HDF5 file efficiently using h5py."""
    if not results:
        logging.warning("No data to save.")
        return
    try:
        with h5py.File(output_file, 'w') as hf:
            for name, matrix in results.items():
                if matrix is not None:
                    hf.create_dataset(name, data=matrix, compression="gzip")
        logging.info(f"Data successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Could not save HDF5 file {output_file}: {e}")

def polar_to_cartesian_optimized(polar_array, target_size=480):
    """
    Optimized interpolation function that accepts a NumPy array directly.
    The default target_size is 480x480 as a balance of detail and performance.
    """
    periodic_array = np.hstack([polar_array, polar_array[:, [0]]])
    rows, cols = periodic_array.shape
    r = np.linspace(0, 1, rows, endpoint=True)
    theta = np.linspace(0, 2 * np.pi, cols, endpoint=True)
    theta_grid, r_grid = np.meshgrid(theta, r, indexing='xy')
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    valid = ~np.isnan(periodic_array)
    
    if not np.any(valid):
        return np.full((target_size, target_size), np.nan, dtype=np.float32)

    x_flat, y_flat, values_flat = x[valid], y[valid], periodic_array[valid]
    
    xi, yi = np.linspace(-1, 1, target_size), np.linspace(-1, 1, target_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    cart_values = griddata(
        (x_flat, y_flat), values_flat, (xi_grid, yi_grid), method='linear'
    )
    
    mean_value = np.nanmean(values_flat)
    cart_values[np.isnan(cart_values)] = mean_value
    cart_values = cart_values.astype(np.float32, copy=False)
    cart_values = cv2.bilateralFilter(cart_values, d=5, sigmaColor=40, sigmaSpace=50)
    
    R = np.sqrt(xi_grid**2 + yi_grid**2)
    cart_values[R > 1.0] = np.nan
    return cart_values

# --- 4. Main Processing Function for a Single File ---

def process_single_file(filepath, output_dir, segments_config):
    """Complete processing pipeline for one CSV file."""
    file_total_start_time = time.time()
    filename = os.path.basename(filepath)
    logging.info(f"Beginning processing of {filename}")

    # Step 1: Fast I/O
    raw_data_dict = read_all_segments_fast(filepath, segments_config)

    # Step 2: Computation
    results = {}
    for name, data_array in raw_data_dict.items():
        if data_array is None:
            continue
        
        numeric_array = pd.to_numeric(data_array.ravel(), errors='coerce').reshape(data_array.shape)
        numeric_array[numeric_array == -1000] = np.nan
        
        if not np.all(np.isnan(numeric_array)):
            # Call the function with the desired target size
            results[name] = polar_to_cartesian_optimized(numeric_array, target_size=480)

    # Step 3: Efficient Saving
    output_hdf_filename = os.path.splitext(filename)[0] + '.h5'
    output_path = os.path.join(output_dir, output_hdf_filename)
    save_to_hdf_fast(results, output_path)

    total_file_time = time.time() - file_total_start_time
    logging.info(f"*** Finished {filename} in {total_file_time:.2f} seconds ***")

# --- 5. Execution Block with Parallel Processing ---

if __name__ == "__main__":
    logging.info("Script starting...")
    script_start_time = time.time()
    
    # --- Configure Paths ---
    # IMPORTANT: Make sure this path is correct on your machine
    folder_path = r'C:\Users\nassd\OneDrive\Bureau\15-20\RefCartesTest'
    output_folder = os.path.join(folder_path, "output_hdf_optimized")
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            logging.warning(f"No CSV files found in {folder_path}. Exiting.")
        else:
            logging.info(f"Found {len(csv_files)} CSV file(s) to process.")
            Parallel(n_jobs=-1)(
                delayed(process_single_file)(f, output_folder, segments) for f in csv_files
            )
    except FileNotFoundError:
        logging.error(f"Input folder not found: {folder_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    # --- Final Report ---
    total_script_time = time.time() - script_start_time
    logging.info("======================================================")
    logging.info(f"GRAND TOTAL for all files took: {total_script_time:.2f} seconds")
    logging.info("======================================================")