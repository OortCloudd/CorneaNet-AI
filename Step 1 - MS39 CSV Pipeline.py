"""
Matrix Reconstruction: Polar to Cartesian (Version 3.0)
Version 3.0 of the pipeline dedicated to reconstructing polar matrices into Cartesian matrices from the MS39 file.

Key Points:
The data is now circular rather than square, improving its fidelity to reality.
A mask has been applied to filter the data and eliminate square borders, making the shape as close as possible to that of a natural eye.
The exact number of 57,132 NaN values comes from the masked square borders.

Current Status:
Noise has significantly decreased, making this version much more accurate than previous ones.
Next change and final change -> Add -1000 data points as blank
This version is almost final, given its performance and achieved precision.

Possible Improvements:
-1000 values have not been accounted for in this version. It would be interesting to implement them for greater accuracy.
Note: This pipeline represents a major advancement in data fidelity and can be used in applications requiring high precision. 
The primary application here is leveraging Deep Learning methods in ophthalmology.
"""
import pandas as pd
import numpy as np
import os
import time
from scipy.interpolate import griddata
import cv2
"""
# ('name of segment', line where it starts, number of lines to read, low interval to take into account, high interval to take into account)
# There must be a way of improving this
"""
"""
    Elevation maps in corneal topography require special processing. They use a reference surface, typically a Best Fitted Sphere (BFS),
    which is calculated using a least weighted squares method. 
    The elevation data can then be analyzed using Zernike polynomials,
    which are mathematical functions particularly useful for describing optical surfaces and wavefront aberrations.
    However, some kind of unknown data transformation is done by the MS39 Machine, outputing all values between 0 and 3
    Values are meaningful as they have p-values < 0.05 in statistical tests (correlations & mann-whitney keratoconus vs non-keratoconus).
"""
segments = [
    #('sagittal_anterior', 28, 27, -999, 10000),
    #('tangential_anterior', 60, 27, -999, 10000),
    #('gaussian_anterior', 92, 22, -999, 10000),
    #('sagittal_posterior', 124, 26, -999, 10000),
    #('tangential_posterior', 156, 26, -999, 10000),
    #('gaussian_posterior', 188, 22, -999, 10000),
    #('refra_frontal_power_anterior', 220, 27, -999, 10000),
    #('refra_frontal_power_posterior', 252, 26, -999, 10000),
    #('refra_equivalent_power', 284, 23, -999, 10000),
    ('elevation_anterior', 316, 27, -999, 10000),
    ('elevation_posterior', 348, 26, -999, 10000),
    ('elevation_stromal', 380, 25, -999, 10000),
    #('corneal_thickness', 412, 26, -999, 10000),
    #('stromal_thickness', 444, 25, -999, 10000),
    #('epithelial_thickness', 476, 25, -999, 10000),
    #('anterior_chamber_depth', 508, 26, -999, 10000)
]

def lire_segment(fichier, debut, n_lignes):
    """
    Reads the segments. The key here is the skiprows function that allows us to bypass the particular
    CSV format of the MS39 Machine. Indeed, the CSV file is quite particular with patient metadata (strings) everywhere.
    Polars must be faster. However, at the date of creation of the code, there wasn't any equivalent of the skiprows parameter.
    """
    try:
        data = pd.read_csv(
            fichier,
            sep=';',
            header=None,
            skiprows=debut,
            nrows=n_lignes,
            usecols=range(256),
            dtype=float
        )
        return data
    except Exception as e:
        print(f"[ERROR] Reading segment (lines {debut}:{debut+n_lignes}) : {e}")
        return pd.DataFrame()


def polar_to_cartesian(polar_df, target_size=512):
    """
    Convert a polar matrix to cartesian, by admitting:
      - Each column corresponds to a radial division (r = 0 -> 1).
      - Each line corresponds to an angle (0° -> 360°).
    
    Function that transforms our polar matrix into a cartesian matrix (N*256 -> 512*512)

    This function is the main core of the code.
    We considered duplicating the last column for periodicity.
    """
    # 1) Duplicate the first column for angle periodicity
    df_periodic = pd.concat(
        [polar_df, polar_df.iloc[:, [0]]], 
        axis=1,
        ignore_index=True
    )
    
    rows, cols = df_periodic.shape  # Number of radial steps, number of angles + 1

    # 2) Vectors for radius (r) and angle (theta)
    r = np.linspace(0, 1, rows, endpoint=True)
    theta = np.linspace(0, 2 * np.pi, cols, endpoint=True)

    # 3) Polar grids (theta_grid, r_grid)
    theta_grid, r_grid = np.meshgrid(theta, r, indexing='xy')

    # 4) Conversion polar -> cartesian
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)

    # 5) Flatten valid values for interpolation
    values = df_periodic.values
    valid = ~np.isnan(values)
    x_flat = x[valid]
    y_flat = y[valid]
    values_flat = values[valid]

    # 6) Output cartesian grid
    xi = np.linspace(-1, 1, target_size)
    yi = np.linspace(-1, 1, target_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # 7) Interpolation
    cart_values = griddata(
        (x_flat, y_flat),
        values_flat,
        (xi_grid, yi_grid),
        method='linear'
    )

    # Replace NaN with mean value
    mean_value = np.nanmean(values_flat)
    cart_values[np.isnan(cart_values)] = mean_value

    # 8) Smoothing with bilateral filter (OpenCV)
    cart_values = cart_values.astype(np.float32, copy=False)
    cart_values = cv2.bilateralFilter(cart_values, d=5, sigmaColor=40, sigmaSpace=50)

    # 9) Mask for points outside the unit circle
    R = np.sqrt(xi_grid**2 + yi_grid**2)
    cart_values[R > 1.0] = np.nan

    # Return as DataFrame
    return pd.DataFrame(cart_values)

def process_and_interpolate(fichier, segments):
    """
    Create a Mask on the matrix to get only eye data (An eye is round which is incompatible with a block matrix)
    """
    results = {}
    for seg in segments:
        name, start_row, num_rows, min_val, max_val = seg
        start_time = time.time()
        print(f"[INFO] Processing segment: {name}")
        df = lire_segment(fichier, start_row, num_rows)
        if df.empty:
            print(f"[WARNING] Segment {name} empty or impossible to read.")
            continue

        # Filter out-of-bounds values
        df = df.mask((df < min_val) | (df > max_val))
        # -1000 => np.nan
        df = df.replace(-1000, np.nan)
        # Remove rows that are entirely NaN
        df = df.dropna(axis=0, how='all')

        nan_count = df.isna().sum().sum()
        print(f"[DEBUG] {name} - Dimensions before interp: {df.shape}, Number of NaN: {nan_count}")

        # Conversion polar->cartesian (512x512) + smoothing
        cart_df = polar_to_cartesian(df, target_size=512)
        nan_count_cart = cart_df.isna().sum().sum()
        print(f"[DEBUG] {name} - Dimensions after interp: {cart_df.shape}, Number of NaN: {nan_count_cart}")

        results[name] = cart_df
        elapsed = time.time() - start_time
        print(f"[INFO] Segment {name} processed in {elapsed:.2f} seconds.")

    return results


def save_to_hdf(results, output_file):
    """
    Save all matrices to an HDF5 file.
    Each segment is saved as a table (key=name).
    """
    if not results:
        print("[WARNING] No data to save.")
        return

    # Open the HDF5 file in 'w' mode (overwrite) the first time,
    # then 'a' (append) for subsequent segments.
    first = True
    for name, df in results.items():
        mode = 'w' if first else 'a'
        df.to_hdf(output_file, key=name, mode=mode)
        first = False

    print(f"[INFO] Data saved in {output_file}")

def process_folder(folder_path):
    """
    Export all of the matrices into hdf5 format
    .xlsx was too heavy and .npy introduced unknown bugs and variations in the final output.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            fichier = os.path.join(folder_path, filename)
            # Output name .h5 instead of .xlsx
            output_hdf_filename = os.path.splitext(filename)[0] + '.h5'
            output_hdf_path = os.path.join(folder_path, output_hdf_filename)
            print(f"[INFO] Beginning processing of file {filename}")
            results = process_and_interpolate(fichier, segments)
            save_to_hdf(results, output_hdf_path)
            print(f"[INFO] Finished processing {filename}")

if __name__ == "__main__":
    folder_path = r'C:\\Users\\nassd\\OneDrive\\Bureau\\15-20\\zernike'
    process_folder(folder_path)

    """
    Onto Step 2 - Colormaps
    """
