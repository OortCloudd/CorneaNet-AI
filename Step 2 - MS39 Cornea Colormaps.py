"""
Maps 3.0: Deep Learning Version
Version dedicated to integration into the machine learning model.

Key Points:

This version is designed to be used directly within the CNN model.

It is tailored for deep learning tasks in ophthalmology.

Number of colors doubled (40) and scales adjusted.


The MS39 scales are dynamic to highlight deformation for ophthalmologists.
Here, we use static scales compatible with computer vision.

Current Issues:

Missing reconstructions:

Elevations

Possible Improvements:

Consider the -1000 values here in addition to the previous code for greater accuracy.


"""

colors = [
    '#57597e',
    '#71466B', '#8A3259', '#a41f46',
    '#B60F37', '#c80029',
    '#D80017', '#e90005',
    '#EC1B02', '#ef3500',
    '#F05400', '#f27300',
    '#ED8D00', '#e9a700',
    '#EFB500', '#f6c300',
    '#F9CE00', '#fbd900',
    '#A0D900', '#46d900',
    '#4ADE7E', '#4ee2fd',
    '#4BD3FE', '#48c4ff',
    '#45B9FF', '#41aeff',
    '#3CA4FE', '#389afc',
    '#338EF2', '#2e81e8',
    '#2A73E9', '#2664e9',
    '#254DF4', '#2435ff',
    '#2C22F2', '#340ee5',
    '#4C07D0', '#6300bb',
    '#8400BA', '#a400ba'
]
cmap = mcolors.ListedColormap(colors)
cmap.set_bad('white', 1.0)  # NaN areas appear white

# Bounds for each segment (13 segments, without elevation)
map_bounds = {
    'sagittal_anterior': [
        5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
        6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
        7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
        8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.8
    ],
    'tangential_anterior': [
        5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
        6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
        7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
        8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.8
    ],
    'gaussian_anterior': [
        5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
        6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
        7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
        8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.8
    ],
    'sagittal_posterior': [
        3.64, 3.725, 3.81, 3.905, 4.00, 4.105, 4.21, 4.325,
        4.44, 4.575, 4.71, 4.855, 5.00, 5.165, 5.33, 5.52,
        5.71, 5.93, 6.15, 6.41, 6.67, 6.97, 7.27, 7.635,
        8.00, 8.445, 8.89, 9.445, 10.00, 10.715, 11.43, 12.38,
        13.33, 14.665,16.00, 18.00, 20.00, 23.335,26.67, 40.00
    ],
    'tangential_posterior': [
        3.64, 3.725, 3.81, 3.905, 4.00, 4.105, 4.21, 4.325,
        4.44, 4.575, 4.71, 4.855, 5.00, 5.165, 5.33, 5.52,
        5.71, 5.93, 6.15, 6.41, 6.67, 6.97, 7.27, 7.635,
        8.00, 8.445, 8.89, 9.445, 10.00, 10.715, 11.43, 12.38,
        13.33, 14.665,16.00, 18.00, 20.00, 23.335,26.67, 40.00
    ],
    'gaussian_posterior': [
        3.64, 3.725, 3.81, 3.905, 4.00, 4.105, 4.21, 4.325,
        4.44, 4.575, 4.71, 4.855, 5.00, 5.165, 5.33, 5.52,
        5.71, 5.93, 6.15, 6.41, 6.67, 6.97, 7.27, 7.635,
        8.00, 8.445, 8.89, 9.445, 10.00, 10.715, 11.43, 12.38,
        13.33, 14.665,16.00, 18.00, 20.00, 23.335,26.67, 40.00
    ],
    'refra_frontal_power_anterior': [
        58.0, 57.5, 57.0, 56.5, 56.0, 55.5, 55.0, 54.5, 54.0, 53.5,
        53.0, 52.5, 52.0, 51.5, 51.0, 50.5, 50.0, 49.5, 49.0, 48.5,
        48.0, 47.5, 47.0, 46.5, 46.0, 45.5, 45.0, 44.5, 44.0, 43.5,
        43.0, 42.5, 42.0, 41.5, 41.0, 40.5, 40.0, 39.5, 39.0, 38.5
    ],
    'refra_frontal_power_posterior': [
        -9.0, -8.85, -8.7, -8.55, -8.4, -8.25, -8.1, -7.95, -7.8, -7.65,
        -7.5, -7.35, -7.2, -7.05, -6.9, -6.75, -6.6, -6.45, -6.3, -6.15,
        -6.0, -5.85, -5.7, -5.55, -5.4, -5.25, -5.1, -4.95, -4.8, -4.65,
        -4.5, -4.35, -4.2, -4.05, -3.9, -3.75, -3.6, -3.45, -3.3, -3.15
    ],
    'refra_equivalent_power': [
        53.0, 52.5, 52.0, 51.5, 51.0, 50.5, 50.0, 49.5, 49.0, 48.5,
        48.0, 47.5, 47.0, 46.5, 46.0, 45.5, 45.0, 44.5, 44.0, 43.5,
        43.0, 42.5, 42.0, 41.5, 41.0, 40.5, 40.0, 39.5, 39.0, 38.5,
        38.0, 37.5, 37.0, 36.5, 36.0, 35.5, 35.0, 34.5, 34.0, 33.5
    ],
    'corneal_thickness': [
        230, 245, 260, 275, 290, 305, 320, 335, 350, 365,
        380, 395, 410, 425, 440, 455, 470, 485, 500, 515,
        530, 545, 560, 575, 590, 605, 620, 635, 650, 665,
        680, 695, 710, 725, 740, 755, 770, 785, 800, 815
    ],
    'stromal_thickness': [
        230, 245, 260, 275, 290, 305, 320, 335, 350, 365,
        380, 395, 410, 425, 440, 455, 470, 485, 500, 515,
        530, 545, 560, 575, 590, 605, 620, 635, 650, 665,
        680, 695, 710, 725, 740, 755, 770, 785, 800, 815
    ],
    'epithelial_thickness': [
        105, 102.5, 100, 97.5, 95, 92.5, 90, 87.5, 85, 82.5,
        80, 77.5, 75, 72.5, 70, 67.5, 65, 62.5, 60, 57.5,
        55, 52.5, 50, 47.5, 45, 42.5, 40, 37.5, 35, 32.5,
        30, 27.5, 25, 22.5, 20, 17.5, 15, 12.5, 10, 7.5
    ],
    'anterior_chamber_depth': [
        -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25,
         0.5,  0.75,  1.0,  1.25,  1.5,  1.75,  2.0,  2.25, 2.5,  2.75,
         3.0,  3.25,  3.5,  3.75,  4.0,  4.25,  4.5,  4.75, 5.0,  5.25,
         5.5,  5.75,  6.0,  6.25,  6.5,  6.75,  7.0,  7.25, 7.5,  7.75
    ]
}

############################################################################
#                           DISPLAY FUNCTION
############################################################################
def plot_map(matrix, segment_name, output_folder):
    """
    matrix: 2D array (numpy) of dimensions 512x512
    segment_name: segment name (key in map_bounds)
    output_folder: folder where to save the output .png
    """
    # Conversion to float and masking of NaN
    matrix = matrix.astype(float)
    matrix_masked = np.ma.masked_invalid(matrix)

    # Get predefined bounds (if applicable)
    bounds = map_bounds.get(segment_name, None)
    if bounds:
        # Check if the scale is decreasing
        if bounds[0] > bounds[-1]:
            # Decreasing scale
            levels = sorted(bounds)
            cmap_mod = cmap.reversed()  # Inversion of the colormap
            norm = mcolors.BoundaryNorm(levels, cmap_mod.N)
            chosen_cmap = cmap_mod
        else:
            # Increasing scale
            levels = sorted(bounds)
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            chosen_cmap = cmap
    else:
        # No defined bounds -> automatic scale
        levels = np.linspace(np.nanmin(matrix), np.nanmax(matrix), 21)
        norm = None
        chosen_cmap = cmap

    # Creation of the figure
    plt.figure(figsize=(10, 8))
    plt.contourf(matrix_masked, levels=levels, cmap=chosen_cmap, norm=norm, extend='both')

    # Hide axes, ticks, etc.
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.margins(0, 0)

    # Save image
    os.makedirs(output_folder, exist_ok=True)
    out_file = os.path.join(output_folder, f"{segment_name}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"[OK] Map generated: {out_file}")

############################################################################
#            GENERATE MAPS FROM .H5 FILES
############################################################################
def generate_maps_3_0_from_hdf():
    """
    1) We look for all .h5 files in: 'input folder'
    2) For each .h5, we create a subfolder in 'output root'
       with the name of the file (without extension).
    3) We open the .h5, read each key/segment, and generate the corresponding .png
       in this subfolder.
    """
    # Folder containing the .h5 files
    h5_input_folder = r"input folder"
    # Global output folder
    output_root = r"output root"

    # Check existence of folders
    if not os.path.isdir(h5_input_folder):
        print(f"[ERROR] Folder not found: {h5_input_folder}")
        return

    # Create global folder if it doesn't exist
    os.makedirs(output_root, exist_ok=True)

    # List all .h5 files in h5_input_folder (without going into subfolders)
    files_h5 = [f for f in os.listdir(h5_input_folder) if f.lower().endswith('.h5')]
    if not files_h5:
        print(f"[INFO] No .h5 files found in {h5_input_folder}")
        return

    print("[INFO] HDF5 files detected:")
    for f in files_h5:
        print(f"   - {f}")

    # Process each .h5 file
    for h5_name in files_h5:
        h5_path = os.path.join(h5_input_folder, h5_name)
        # Output subfolder name = filename without extension
        base_name = os.path.splitext(h5_name)[0]  # e.g.: "test1.h5" -> "test1"
        output_subfolder = os.path.join(output_root, base_name)
        os.makedirs(output_subfolder, exist_ok=True)

        # Open the .h5 and process each key
        print(f"[INFO] Reading file {h5_path}")
        try:
            with pd.HDFStore(h5_path, mode='r') as store:
                keys = store.keys()  # e.g.: ['/sagittal_anterior', ...]
                if not keys:
                    print(f"[WARN] No keys found in {h5_path}")
                    continue

                print(f"     Keys detected: {keys}")

                # For each key => generate the map
                for key in keys:
                    segment_name = key.strip('/')  # remove the '/'
                    df = store[key]  # DataFrame
                    matrix = df.values  # numpy array
                    # Call plot_map function to save the map
                    plot_map(matrix, segment_name, output_subfolder)

        except Exception as e:
            print(f"[ERROR] Cannot read {h5_path}: {e}")

    print("[INFO] All maps have been generated in:", output_root)


############################################################################
#                                   MAIN
############################################################################
if __name__ == "__main__":
    generate_maps_3_0_from_hdf()
    print("[INFO] Script completed successfully.")
