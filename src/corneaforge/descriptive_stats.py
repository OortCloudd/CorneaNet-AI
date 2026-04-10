"""
CorneaForge Descriptive Statistics Pipeline
=============================================

Extracts tabular features from MS39 CSV exports for traditional ML and
clinical research papers.

INPUT:  One MS39 .csv file (or a folder of them)
OUTPUT: One CSV/Excel file with one row per patient, one column per feature

Unlike the NN pipeline (which feeds spatial maps to CNNs), this pipeline
collapses each 2D map into scalar summary statistics. The output is what
XGBoost, logistic regression, random forests, and ophthalmology papers
consume: mean corneal thickness, curvature skewness, etc.

FEATURE STRUCTURE
-----------------
Features are computed per ZONE, not globally. The cornea is divided into:

  Radial zones (from the 0.2mm ring spacing):
    0-1mm, 1-2mm, 2-3mm, 3-4mm, 4-5mm, 5-6mm

  Angular sectors (TABO convention, confirmed against MS39 screen):
    Superior (45-135°), Inferior (225-315°),
    Nasal (315-45°), Temporal (135-225°)
    Nasal/Temporal labels flip for OS (left eye) via CSV metadata.

  Full ring (optional, disabled by default via INCLUDE_FULL_RING):
    All 256 columns combined for a given radial zone.
    Redundant: full_ring_mean = (sup_mean + inf_mean + nas_mean + tem_mean) / 4
    Verified at 0.000000 difference. Useful for clinical papers where global
    zone-level statistics are conventional, but adds collinearity for ML.

  Combined: each feature is localized, e.g., "stromal_thickness_mean_2_3mm_inferior"

This preserves spatial information that global statistics erase:
  - Radial gradients (central thinning vs peripheral thickening)
  - Angular asymmetry (inferior steepening in ectatic diseases)
  - NaN% per zone (peripheral capture quality)

STATISTICS PER ZONE (10 features)
---------------------------------
  - min, max              Extremes (tail behavior, not derivable from mean/std)
  - mean, median          Central tendency (diverge on skewed/ectatic distributions)
  - std                   Spread in original units
  - skew                  Asymmetry of the distribution
  - kurtosis              Tail weight (excess kurtosis, Fisher definition)
  - q25, q75              Quartiles (robust spread, less sensitive to outliers than std)
  - nan_pct               Fraction of missing measurements in this zone (capture quality)

FEATURES NOT INCLUDED (and why)
-------------------------------
  - Coefficient of Variation (std/mean): a boosted tree learns this ratio
    implicitly by splitting on std conditioned on mean. TabPFN handles
    feature interactions natively. Pre-computing CV only helps models that
    cannot learn interactions (e.g., Elastic Net), which are not the target
    production architecture.
  - Radial gradient (zone[n+1] - zone[n]): same reasoning. Trees learn
    differences between adjacent zone features via consecutive splits.
  - IQR (q75 - q25), range (max - min): trivially derivable from features
    already in the set.
  General principle: don't pre-compute what the model can learn. Feature
  engineering should add domain knowledge the model cannot extract from raw
  features (angular sectors, NaN%), not arithmetic it already does.

SEGMENT SELECTION
-----------------
Not all 16 MS39 segments carry independent information. Toggle segments
on/off via SEGMENTS_ENABLED at the top of this file:
  - For tabular ML: disable redundant segments to avoid collinearity
  - For clinical papers: enable everything for comprehensive reporting
  - For custom research: pick exactly what you need

REDUNDANCY FORMULAS — why disabled segments are derivable:

  corneal_thickness = stromal_thickness + epithelial_thickness
      Exact anatomical identity. The MS39 measures stroma and epithelium
      separately; corneal thickness is their sum. Verified: difference < 3µm.

  gaussian_curvature ≈ sqrt(R_sagittal × R_tangential)
      Gaussian curvature is the product of the two principal curvatures
      (κ₁ × κ₂). In the radius domain: R_gauss ≈ sqrt(R_sag × R_tan).
      Not perfectly exact because the MS39 applies asphericity corrections,
      but >90% of variance is explained by sagittal and tangential alone.

  refra_frontal_power_anterior ≈ (n_keratometric - 1) / R_sagittal_anterior
      Refractive power = (n₂ - n₁) / R. Using the standard keratometric
      index n = 1.3375: P_ant ≈ 337.5 / R_sag_ant (R in mm, P in Diopters).
      The MS39 may use a slightly different internal formula, but the
      correlation with 1/R_sagittal is >0.95.

  refra_frontal_power_posterior ≈ (n_aqueous - n_cornea) / R_sagittal_posterior
      P_post ≈ (1.336 - 1.376) / R_sag_post = -40 / R_sag_post
      Same relationship as anterior, different refractive indices.

  refra_equivalent_power ≈ P_ant + P_post - (t / n_cornea) × P_ant × P_post
      The thick lens equation. t = corneal thickness (in meters),
      n_cornea = 1.376. Since P_ant and P_post are themselves derivable
      from curvatures, equivalent power is a second-order redundancy.

  elevation maps — No formula; excluded for a different reason.
      Raw elevation without Best Fit Sphere (BFS) subtraction is dominated
      by the global corneal curvature (~7.8mm radius sphere). The clinically
      interesting signal (local ectasia, surface irregularity) is a tiny
      residual on top of this DC component. Summary statistics on raw elevation
      mostly recapitulate what curvature maps already capture. The Zernike
      pipeline handles elevation properly: BFS subtraction first, then
      polynomial decomposition of the residual.

CURRENT LIMITATIONS
-------------------
1. ACQUISITION QUALITY — The main weakness of MS39-derived features is
   the acquisition success rate. Poor fixation, blinks, eyelid obstruction,
   and tear film instability all produce missing data (-1000 values),
   predominantly in the periphery. The NaN% feature per zone captures this,
   but a built-in acquisition quality score from the device would be a
   stronger quality gate. This is planned for a future version.

2. OUTLIER REJECTION — Non-normal distributions (skewed thickness in
   ectatic corneas, bimodal curvature in high astigmatism) pose a challenge
   for standard outlier detection methods. A global MAD/z-score approach
   risks cutting real anatomical variation in the periphery, where values
   legitimately differ from the central median. No automated outlier
   rejection is applied in the current version — this is a deliberate
   design choice pending a more robust approach.

Usage
-----
    # Single file
    python -m corneaforge.descriptive_stats /path/to/patient.csv -o ./stats/

    # Whole folder → one CSV with all patients
    python -m corneaforge.descriptive_stats /path/to/csv_folder/ -o ./stats/
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from corneaforge.core import (
    parse_csv,
    parse_metadata,
)

logger = logging.getLogger("corneaforge.stats")

# ==========================================================================
# SEGMENT CONFIGURATION
# ==========================================================================
# Toggle segments on/off. True = include in output, False = skip.
#
# Default: 7 independent segments enabled, 9 redundant/problematic disabled.
# Change these for your use case — no code modifications needed.

SEGMENTS_ENABLED = {
    # --- Curvature: anterior surface ---
    "sagittal_anterior": True,
    "tangential_anterior": True,
    "gaussian_anterior": False,  # ≈ f(sagittal, tangential)
    # --- Curvature: posterior surface ---
    "sagittal_posterior": True,
    "tangential_posterior": True,
    "gaussian_posterior": False,  # ≈ f(sagittal, tangential)
    # --- Refractive power ---
    "refra_frontal_power_anterior": False,  # ≈ 1 / sagittal_anterior
    "refra_frontal_power_posterior": False,  # ≈ 1 / sagittal_posterior
    "refra_equivalent_power": False,  # ≈ f(powers, thickness)
    # --- Elevation (use Zernike pipeline instead) ---
    "elevation_anterior": False,  # Unknown MS39 transformation
    "elevation_posterior": False,
    "elevation_stromal": False,
    # --- Thickness ---
    "corneal_thickness": False,  # = stromal + epithelial
    "stromal_thickness": True,
    "epithelial_thickness": True,
    # --- Anterior chamber ---
    "anterior_chamber_depth": True,
}

# Include full-ring statistics (all 256 columns combined per radial zone)?
# These are mathematically redundant with the 4 angular sector stats:
#   full_ring_mean = (superior_mean + inferior_mean + nasal_mean + temporal_mean) / 4
# Verified at 0.000000 difference. Useful for clinical papers where
# global zone statistics are conventional. Adds collinearity for ML.
INCLUDE_FULL_RING = False


# ==========================================================================
# MS39 COORDINATE SYSTEM
# ==========================================================================
# Confirmed by visual comparison: legacy code reconstructions match the MS39
# screen exactly, so the CSV export follows TABO convention with no offset.
#
# TABO (Technische Ausschuss für Brillenoptik):
#   Column 0 = 0° (horizontal)
#   Counterclockwise: 0° → 90° (superior) → 180° → 270° (inferior)
#   256 columns = 256 meridians, 1.40625° per column
#
# Nasal/Temporal depends on which eye (OD/OS):
#   OD (right eye): 0° = nasal,   180° = temporal
#   OS (left eye):  0° = temporal, 180° = nasal

# Radial zones: each row = 0.2mm step from center
# Radii: 0.0, 0.2, 0.4, ..., 5.8, 6.0 (31 values, 0-indexed)
RADIAL_STEP_MM = 0.2
ROWS_PER_MM = 5  # 1mm / 0.2mm = 5 rows

RADIAL_ZONES = [
    ("0_1mm", 0, 5),  # rows 0-4:   0.0-0.8mm
    ("1_2mm", 5, 10),  # rows 5-9:   1.0-1.8mm
    ("2_3mm", 10, 15),  # rows 10-14: 2.0-2.8mm
    ("3_4mm", 15, 20),  # rows 15-19: 3.0-3.8mm
    ("4_5mm", 20, 25),  # rows 20-24: 4.0-4.8mm
    ("5_6mm", 25, 31),  # rows 25-30: 5.0-6.0mm
]

# Maximum radial zone per segment. Segments with fewer than 26 real rows
# physically cannot reach the outer zones. Verified on 1,238 patients:
#   - 25-row segments (posterior curvatures, ACD): never reach 5-6mm
#   - 26-row segments (anterior curvatures, thickness, power): reach all zones
# Columns beyond this limit are never computed — guarantees every patient
# produces the exact same column set regardless of batch size.
MAX_ZONE = {
    "sagittal_anterior": "5_6mm",
    "tangential_anterior": "5_6mm",
    "sagittal_posterior": "4_5mm",
    "tangential_posterior": "4_5mm",
    "stromal_thickness": "5_6mm",
    "epithelial_thickness": "5_6mm",
    "anterior_chamber_depth": "4_5mm",
}

_ZONE_ORDER = [z[0] for z in RADIAL_ZONES]


def _zones_for_segment(seg_name):
    """Return the radial zones this segment can reach."""
    max_zone = MAX_ZONE.get(seg_name, "5_6mm")
    max_idx = _ZONE_ORDER.index(max_zone)
    return RADIAL_ZONES[: max_idx + 1]


# Angular sectors: column ranges for each quadrant
# Each quadrant spans 90° = 64 columns
def _angular_sectors(eye):
    """
    Return angular sector definitions based on which eye (OD/OS).

    Superior and inferior are the same for both eyes — top of the cornea
    is top regardless. Nasal and temporal swap between OD and OS.

    Parameters
    ----------
    eye : str
        'OD' (right) or 'OS' (left). Determines nasal/temporal assignment.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Sector name → list of (start_col, end_col) ranges.
        Some sectors wrap around column 0/256, hence a list of ranges.
    """
    # Superior: 45°-135° → cols 32-96
    # Inferior: 225°-315° → cols 160-224
    # These don't change with OD/OS
    sectors = {
        "superior": [(32, 96)],
        "inferior": [(160, 224)],
    }

    # Horizontal sectors depend on eye laterality
    # 315°-45° → cols 224-256 + 0-32 (wraps around)
    # 135°-225° → cols 96-160
    if eye == "OD":
        sectors["nasal"] = [(224, 256), (0, 32)]
        sectors["temporal"] = [(96, 160)]
    else:
        sectors["temporal"] = [(224, 256), (0, 32)]
        sectors["nasal"] = [(96, 160)]

    return sectors


# ==========================================================================
# STATISTICS COMPUTATION
# ==========================================================================


def _compute_stats(values):
    """
    Compute 10 descriptive features from a 1D array of measurements.

    Returns a dict with:
      - 9 summary statistics (min, max, mean, median, std, skew, kurtosis, Q25, Q75)
      - nan_pct: fraction of input values that were NaN (capture quality indicator)

    If all values are NaN, returns all NaN values with nan_pct = 1.0.
    """
    total = len(values)
    valid = values[~np.isnan(values)]
    nan_pct = 1.0 - (len(valid) / total) if total > 0 else 1.0

    if len(valid) < 2:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "nan_pct": nan_pct,
        }

    mean = np.mean(valid)
    std = np.std(valid, ddof=1)

    # Skewness and kurtosis: computed manually to avoid scipy dependency.
    # scipy.stats.skew/kurtosis would work too, but numpy is already imported.
    n = len(valid)
    if std > 0 and n >= 3:
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((valid - mean) / std) ** 3)
    else:
        skew = 0.0

    if std > 0 and n >= 4:
        kurt = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(
            ((valid - mean) / std) ** 4
        ) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    else:
        kurt = 0.0

    return {
        "min": np.min(valid),
        "max": np.max(valid),
        "mean": mean,
        "median": np.median(valid),
        "std": std,
        "skew": skew,
        "kurtosis": kurt,
        "q25": np.percentile(valid, 25),
        "q75": np.percentile(valid, 75),
        "nan_pct": nan_pct,
    }


def _extract_zone(polar_data, row_start, row_end, col_ranges):
    """
    Extract values from a specific radial × angular zone of the polar matrix.

    Parameters
    ----------
    polar_data : np.ndarray
        Cleaned polar matrix, shape (n_rows, 256).
    row_start, row_end : int
        Radial bounds (row indices).
    col_ranges : list[tuple[int, int]]
        Angular bounds as (start_col, end_col) pairs.
        Multiple ranges handle the wrap-around at 0°/360°.

    Returns
    -------
    np.ndarray
        1D array of values in this zone (may contain NaN).
    """
    # Clamp to actual data dimensions (some segments have fewer rows)
    row_end = min(row_end, polar_data.shape[0])
    if row_start >= row_end:
        return np.array([])

    parts = []
    for c_start, c_end in col_ranges:
        parts.append(polar_data[row_start:row_end, c_start:c_end].ravel())

    return np.concatenate(parts) if parts else np.array([])


# ==========================================================================
# SINGLE FILE PROCESSING
# ==========================================================================


def process_segments(raw_segments, metadata, filename=""):
    """
    Extract all tabular features from pre-parsed MS39 data.

    Use this when you've already called parse_csv() and parse_metadata()
    and want to avoid reading the file again. For single-file convenience,
    use process_single_csv().

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of parse_csv().
    metadata : dict[str, str]
        Output of parse_metadata().
    filename : str
        Original filename (for the output row, cosmetic only).

    Returns
    -------
    dict
        One key per feature, ready to become a DataFrame row.
    """
    # Start the feature row with patient info
    # Names are included in plaintext for now. A future pipeline will handle
    # encryption/pseudonymization for GDPR/HIPAA compliance before any data
    # leaves the on-premise environment.
    row = {
        "filename": filename,
        "patient_last_name": metadata.get("Patient_Last_Name", ""),
        "patient_first_name": metadata.get("Patient_First_Name", ""),
        "patient_id": metadata.get("Patient_ID", ""),
        "patient_dob": metadata.get("Patient_Date_of_Birth", ""),
        "patient_gender": metadata.get("Patient_Gender", ""),
        "exam_eye": metadata.get("Exam_Eye", ""),
        "exam_date": metadata.get("Exam_Scan_Date", ""),
    }

    # Determine angular sectors based on which eye
    eye = metadata.get("Exam_Eye", "OD").strip().upper()
    if eye not in ("OD", "OS"):
        eye = "OD"  # fallback
    sectors = _angular_sectors(eye)

    # All angular columns combined (for radial-only stats)
    all_cols = [(0, 256)]

    # Process each enabled segment
    for seg_name, enabled in SEGMENTS_ENABLED.items():
        if not enabled:
            continue
        if seg_name not in raw_segments:
            continue

        polar_data = raw_segments[seg_name].copy()
        polar_data[polar_data == -1000] = np.nan

        zones = _zones_for_segment(seg_name)

        # --- Full ring stats (optional, redundant with sector stats) ---
        if INCLUDE_FULL_RING:
            for zone_name, r_start, r_end in zones:
                values = _extract_zone(polar_data, r_start, r_end, all_cols)
                if len(values) == 0:
                    continue
                stats = _compute_stats(values)
                for stat_name, stat_val in stats.items():
                    row[f"{seg_name}_{stat_name}_{zone_name}"] = stat_val

        # --- Angular sector stats (per radial zone) ---
        for sector_name, col_ranges in sectors.items():
            for zone_name, r_start, r_end in zones:
                values = _extract_zone(polar_data, r_start, r_end, col_ranges)
                if len(values) == 0:
                    continue
                stats = _compute_stats(values)
                for stat_name, stat_val in stats.items():
                    row[f"{seg_name}_{stat_name}_{zone_name}_{sector_name}"] = stat_val

    return row


def process_single_csv(filepath):
    """
    Extract all tabular features from one MS39 CSV.

    Convenience wrapper: parses the file then calls process_segments().
    For batch processing or when combining with other pipelines,
    call parse_csv() once and pass the result to process_segments() directly.
    """
    metadata = parse_metadata(filepath)
    raw_segments = parse_csv(filepath)
    return process_segments(raw_segments, metadata, filename=os.path.basename(filepath))


# ==========================================================================
# BATCH PROCESSING
# ==========================================================================


def process_folder(input_folder, output_folder, output_format="csv"):
    """
    Process all CSV files in a folder, producing one tabular output file.

    Parameters
    ----------
    input_folder : str
        Folder containing MS39 CSV files.
    output_folder : str
        Where to save the output file.
    output_format : str
        'csv' or 'xlsx'.
    """
    os.makedirs(output_folder, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(".csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", input_folder)
        return

    logger.info("Found %d CSV file(s)", len(csv_files))
    total_start = time.time()

    all_rows = []
    for csv_file in csv_files:
        file_start = time.time()
        filepath = os.path.join(input_folder, csv_file)

        row = process_single_csv(filepath)
        all_rows.append(row)

        elapsed = time.time() - file_start
        n_features = len(row) - 8  # subtract metadata columns
        logger.info("OK %s | %d features | %.2fs", csv_file, n_features, elapsed)

    df = pd.DataFrame(all_rows)

    # Save output
    if output_format == "xlsx":
        output_path = os.path.join(output_folder, "descriptive_stats.xlsx")
        df.to_excel(output_path, index=False)
    else:
        output_path = os.path.join(output_folder, "descriptive_stats.csv")
        df.to_csv(output_path, index=False, sep=";", float_format="%.6f", quoting=1)

    logger.info("Saved %s", output_path)
    logger.info("  %d patients × %d columns", len(all_rows), len(df.columns))
    logger.info("  Total: %.2fs", time.time() - total_start)


# ==========================================================================
# CLI
# ==========================================================================


def main():
    """Entry point for the descriptive stats pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="CorneaForge: Extract tabular features from MS39 CSV exports.",
    )
    parser.add_argument("input", help="Path to a CSV file or folder of CSVs")
    parser.add_argument(
        "-o",
        "--output",
        default="./output_stats",
        help="Output folder (default: ./output_stats)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "xlsx"],
        default="csv",
        help="Output format (default: csv)",
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        os.makedirs(args.output, exist_ok=True)
        t = time.time()
        row = process_single_csv(args.input)
        df = pd.DataFrame([row])
        n_features = len(row) - 8

        if args.format == "xlsx":
            out = os.path.join(args.output, "descriptive_stats.xlsx")
            df.to_excel(out, index=False)
        else:
            out = os.path.join(args.output, "descriptive_stats.csv")
            df.to_csv(out, index=False, sep=";", float_format="%.6f", quoting=1)

        logger.info("Saved %s | %d features | %.2fs", out, n_features, time.time() - t)

    elif os.path.isdir(args.input):
        process_folder(args.input, args.output, args.format)
    else:
        logger.error("'%s' is not a valid file or directory", args.input)
        sys.exit(1)


if __name__ == "__main__":
    main()
