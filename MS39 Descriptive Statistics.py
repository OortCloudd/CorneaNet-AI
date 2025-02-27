"""
Compute and extract descriptive statistics from MS39 Raw CSV Files
Very useful for ophthalmologists as they work daily on the MS39
Huge impact for writing papers on the Cornea
"""
import os
import glob
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox  # Added for messagebox
from scipy.stats import zscore

# Definition of segments with: 'type', skipped lines in the dataframe, number of lines, min threshold, max threshold
segments = [
    ('sagittal_anterior', 28, 28, -10000, 10000),
    ('tangential_anterior', 60, 28, -10000, 10000),
    ('gaussian_anterior', 92, 23, -10000, 10000),
    ('sagittal_posterior', 124, 27, -10000, 10000),
    ('tangential_posterior', 156, 27, -10000, 10000),
    ('gaussian_posterior', 188, 23, -10000, 10000),
    ('refra_frontal_power_anterior', 220, 28, -10000, 10000),
    ('refra_frontal_power_posterior', 252, 27, -10000, 10000),
    ('refra_equivalent_power', 284, 24, -10000, 10000),
    ('elevation_anterior', 316, 28, -10000, 10000),
    ('elevation_posterior', 348, 27, -10000, 10000),
    ('elevation_stromal', 380, 26, -10000, 10000),
    ('corneal_thickness', 412, 27, -10000, 10000),
    ('stromal_thickness', 444, 26, -10000, 10000),
    ('epithelial_thickness', 476, 26, -10000, 10000),
    ('anterior_chamber_depth', 508, 27, -10000, 10000)
]

# Reading a segment from the CSV
def lire_segment(fichier, debut, n_lignes):
    """
    Reads a specified segment from a CSV file.

    :param fichier: Path to the CSV file
    :param debut: Number of lines to skip
    :param n_lignes: Number of lines to read
    :return: A pandas DataFrame containing the segment data
    """
    data = pd.read_csv(
        fichier, 
        sep=';', 
        header=None, 
        skiprows=range(debut), 
        nrows=n_lignes,
        converters={i: lambda x: pd.to_numeric(x, errors='coerce') for i in range(100)}
    )
    return data

def calculer_statistiques(data, seuil_min, seuil_max, seuil_zscore=1.5, seuil_nan=0.33):
    """
    Calculates descriptive statistics on the data after applying several filters:
      1. Replace -1000 values with NaN
      2. Discard values outside predefined min/max thresholds
      3. Use a robust z-score (based on median and MAD) to identify outliers
      4. Exclude data if NaN percentage exceeds a certain threshold
      5. Return descriptive statistics (min, max, mean, median, etc.)

    :param data: pandas DataFrame containing the data
    :param seuil_min: Minimum threshold
    :param seuil_max: Maximum threshold
    :param seuil_zscore: Robust z-score threshold
    :param seuil_nan: Maximum allowed percentage of NaN values
    :return: A dictionary of descriptive statistics or None if data is excluded
    """

    # Replace all -1000 values with NaN
    data.replace(-1000, np.nan, inplace=True)

    # Remove values outside min/max thresholds by setting them to NaN
    data[(data < seuil_min) | (data > seuil_max)] = np.nan

    # Calculate the median for each column
    median_values = data.median()

    # Calculate the Median Absolute Deviation (MAD) for each column
    mad_values = data.subtract(median_values).abs().median()

    # Scale factor (1.4826) is often used to make MAD consistent with the standard deviation for normal distributions
    scale_factor = 1.4826

    # Compute robust z-scores
    robust_z = data.subtract(median_values).divide(scale_factor * mad_values)

    # Identify outliers based on robust z-scores and replace them with NaN
    abs_robust_z = robust_z.abs()
    data[abs_robust_z > seuil_zscore] = np.nan

    # Check the percentage of NaN values across all columns
    pourcentage_nan = data.isna().mean().mean()

    # If the percentage of NaN values exceeds the threshold, exclude this data (return None)
    if pourcentage_nan > seuil_nan:
        return None

    # Compute descriptive statistics
    return {
        'min': np.nanmin(data),
        'max': np.nanmax(data),
        'mean': np.nanmean(data),
        'median': np.nanmedian(data),
        'std': np.nanstd(data),
        'skew': pd.DataFrame(data).skew(axis=0, skipna=True).mean(skipna=True),
        'kurt': pd.DataFrame(data).kurt(axis=0, skipna=True).mean(skipna=True),
        'quant25': np.nanpercentile(data, 25),
        'quant75': np.nanpercentile(data, 75)
    }

# Creating columns by splitting the MS39 DataFrame name
def traiter_csvs(dossier_csv, fichier_excel_final, segments):
    """
    Processes all CSV files in a folder and compiles their statistics into an Excel file.

    :param dossier_csv: Folder containing CSV files
    :param fichier_excel_final: Output Excel file path
    :param segments: List of segments to read and analyze
    """
    all_results = []
    for fichier in glob.glob(os.path.join(dossier_csv, '*.csv')):
        nom_fichier = os.path.splitext(os.path.basename(fichier))[0]

        # Split the file name to extract relevant information
        resultats = {
            'Nom': nom_fichier.split('^')[1] if '^' in nom_fichier else '',
            'Prenom': nom_fichier.split('^')[2] if len(nom_fichier.split('^')) > 2 else '',
            'IPP/Ncons': nom_fichier.split('^')[3] if len(nom_fichier.split('^')) > 3 else '',
            'DDC': nom_fichier.split('^')[4].split('T')[0] if len(nom_fichier.split('^')) > 4 else '',
            'Nimage': 'T' + nom_fichier.split('^')[4].split('T')[1] if len(nom_fichier.split('^')) > 4 else '',
            'Cote': nom_fichier.split('^')[5] if len(nom_fichier.split('^')) > 5 else ''
        }
        
        # For each segment, read data and compute statistics
        for label, debut, n_lignes, seuil_min, seuil_max in segments:
            data = lire_segment(fichier, debut, n_lignes)
            stats = calculer_statistiques(data, seuil_min, seuil_max)
            if stats is not None:
                # Update the results dictionary with statistics for this segment
                resultats.update({f"{stat}_{label}": value for stat, value in stats.items()})
            else:
                # If any segment is excluded, we discard the entire observation
                resultats = None
                break

        # Append the results to the list only if not excluded
        if resultats is not None:
            all_results.append(resultats)

    # Convert the list of dictionaries into a DataFrame and export to Excel
    df_resultats = pd.DataFrame(all_results)
    df_resultats.to_excel(fichier_excel_final, index=False)

def selectionner_dossier_csv():
    """
    Opens a folder selection dialog to choose the CSV directory.
    """
    dossier_csv = filedialog.askdirectory()
    entry_dossier_csv.delete(0, tk.END)
    entry_dossier_csv.insert(tk.END, dossier_csv)

def enregistrer_fichier_excel():
    """
    Opens a file save dialog to choose the output Excel file.
    """
    fichier_excel_final = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Fichiers Excel", "*.xlsx")])
    entry_fichier_excel.delete(0, tk.END)
    entry_fichier_excel.insert(tk.END, fichier_excel_final)

def executer_pipeline():
    """
    Executes the entire pipeline: read CSVs, process them, and save results to Excel.
    """
    dossier_csv = entry_dossier_csv.get()
    fichier_excel_final = entry_fichier_excel.get()
    traiter_csvs(dossier_csv, fichier_excel_final, segments)
    messagebox.showinfo("Succès", "Le fichier Excel a été créé avec succès.")

# GUI creation
window = tk.Tk()
window.title("Extraction fichiers MS39")

info_label = tk.Label(window, text="Vitesse : Environ 65 fichiers / minute. Veuillez patienter")
info_label.pack(pady=10)

label_dossier_csv = tk.Label(window, text="Dossier CSV avec les fichiers MS39 :")
label_dossier_csv.pack()

entry_dossier_csv = tk.Entry(window, width=50)
entry_dossier_csv.pack()

button_dossier_csv = tk.Button(window, text="Sélectionner", command=selectionner_dossier_csv)
button_dossier_csv.pack()

label_fichier_excel = tk.Label(window, text="Fichier Excel de sortie :")
label_fichier_excel.pack()

entry_fichier_excel = tk.Entry(window, width=50)
entry_fichier_excel.pack()

button_fichier_excel = tk.Button(window, text="Enregistrer sous", command=enregistrer_fichier_excel)
button_fichier_excel.pack()

button_executer = tk.Button(window, text="Exécuter", command=executer_pipeline)
button_executer.pack()

window.mainloop()
