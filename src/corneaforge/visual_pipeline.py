"""
CorneaForge Visual Pipeline
=============================

Converts MS39 CSV exports into clinical colormap images for ophthalmologists.

INPUT:  One MS39 .csv file (or a folder of them)
OUTPUT: One folder per patient containing 13 colormap PNG images

These images are designed for HUMAN clinical reading, not for neural networks.
(For model training, use nn_pipeline.py instead.)

Each PNG uses:
  - A 40-color clinical palette (red → green → blue → purple)
  - Fixed or dynamic scales with clinically meaningful bounds per segment
  - Sharp missing-data cutoffs (interpolate then punch holes where -1000 was)
  - Gray background for areas outside the cornea
  - 512×512 resolution for detailed visual inspection

WHY FIXED SCALES (not auto-scaled per patient)
-----------------------------------------------
The MS39 machine uses dynamic scales that auto-adjust to each patient's
data range. This is great for highlighting individual deformations, but
makes cross-patient comparison impossible — "the same color" means
different values for different patients.

Our static scales are anchored to population-level ranges. This means:
  - A clinician can compare two patients' maps side by side
  - An "abnormal" region stands out the same way in every patient
  - The maps are reproducible: same data always produces the same image

NO POST-PROCESSING FILTER
--------------------------
The maps show the interpolated data as-is, with no additional smoothing.
The Delaunay interpolation from 26×256 polar points to 512×512 Cartesian
already produces smooth output (97% of pixels are interpolated). The MS-39
also applies internal smoothing before export.

A bilateral filter was previously applied but was removed after analysis
showed its parameters (sigmaColor=40) made it equivalent to Gaussian blur
on 11 of 13 segments — destroying the edge preservation it was supposed
to provide. Rather than maintain per-segment filter calibration (which
depends on disease severity), the filter was removed entirely.

Usage
-----
    # Single file
    python visual_pipeline.py /path/to/patient.csv -o ./maps/

    # Whole folder
    python visual_pipeline.py /path/to/csv_folder/ -o ./maps/
"""

import argparse
import json
import logging
import os
import sys
import time

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from corneaforge.core import (
    DEFAULT_SEGMENTS,
    clean_polar_data,
    missing_data_mask,
    parse_csv,
    polar_to_cartesian,
)

logger = logging.getLogger("corneaforge.visual")

VISUAL_SIZE = 512


# ==========================================================================
# CLINICAL PALETTE (loaded from clinical_palette.json)
# ==========================================================================
# Colors, bounds, and units are stored in a JSON file to keep this module
# focused on logic, not data. The JSON contains:
#   - colors: 40 hex values (purple → red → yellow → green → blue → purple)
#   - bounds: per-segment arrays of 40 boundary values for fixed color scales
#   - units: per-segment physical unit labels (mm, µm, D)

_PALETTE_PATH = os.path.join(os.path.dirname(__file__), "clinical_palette.json")
with open(_PALETTE_PATH) as _f:
    _palette = json.load(_f)

CLINICAL_COLORS = _palette["colors"]

CMAP = mcolors.ListedColormap(CLINICAL_COLORS)
CMAP.set_bad("white", 1.0)  # NaN pixels (outside cornea) render white


CLINICAL_BOUNDS = _palette["bounds"]
SEGMENT_UNITS = _palette["units"]


# NOTE: Bilateral filter was removed.
#
# The legacy code applied cv2.bilateralFilter(d=5, sigmaColor=40, sigmaSpace=50)
# to all segments. Analysis showed sigmaColor=40 was 50-130x larger than the
# data standard deviation for curvature maps (std ~0.3-0.7mm), making the filter
# equivalent to a plain Gaussian blur on 11 of 13 segments. Edge preservation
# — the reason bilateral filters exist — was only working on thickness maps.
#
# Rather than calibrate per-segment parameters (which depend on disease severity
# and would need clinical validation), we removed the filter entirely. The
# Delaunay interpolation from 26x256 polar points to 512x512 Cartesian already
# produces smooth output — 97% of pixels are interpolated. The MS-39 also
# applies internal smoothing before export.
#
# If an ophthalmologist needs additional smoothing for a specific clinical use
# case, it can be added as an optional flag in a future version.


# ==========================================================================
# COLORMAP PLOTTING
# ==========================================================================


def _build_color_bar(chosen_cmap, levels, unit, bar_height, bar_width=60):
    """
    Build a vertical color bar image with tick labels.

    The color bar shows the value-to-color mapping so the ophthalmologist
    can read exact values from the map. Without it, the map is unreadable.

    Parameters
    ----------
    chosen_cmap : Colormap
        The colormap used for the map (may be reversed for decreasing scales).
    levels : list[float]
        The boundary values (sorted ascending, 40 values → 39 color bins).
    unit : str
        Physical unit label (mm, µm, D).
    bar_height : int
        Height in pixels (matches the map image height).
    bar_width : int
        Width of the color bar in pixels.

    Returns
    -------
    PIL.Image
        RGBA image of the color bar with tick labels.
    """
    n_colors = len(levels) - 1  # 40 boundaries → 39 bins
    label_margin = 80  # space for tick labels
    pad = 20  # vertical padding so top/bottom labels aren't clipped
    total_width = bar_width + label_margin
    img = Image.new("RGBA", (total_width, bar_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load a readable font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    # Usable area for the gradient (with padding)
    grad_top = pad
    grad_bot = bar_height - pad

    # Draw the color gradient (bottom = low values, top = high values)
    for i in range(n_colors):
        color_rgba = chosen_cmap(i / max(n_colors - 1, 1))
        color_rgb = tuple(int(c * 255) for c in color_rgba[:3])

        y_top = int(grad_top + (grad_bot - grad_top) * (1 - (i + 1) / n_colors))
        y_bot = int(grad_top + (grad_bot - grad_top) * (1 - i / n_colors))
        draw.rectangle([0, y_top, bar_width - 1, y_bot], fill=color_rgb)

    # Draw tick labels — show ~8 evenly spaced values from the scale
    n_ticks = 8
    tick_indices = np.linspace(0, len(levels) - 1, n_ticks, dtype=int)
    for idx in tick_indices:
        value = levels[idx]
        y = int(grad_top + (grad_bot - grad_top) * (1 - idx / (len(levels) - 1)))

        if abs(value) >= 10:
            label = f"{value:.0f}"
        else:
            label = f"{value:.1f}"

        draw.line([(bar_width - 4, y), (bar_width, y)], fill=(0, 0, 0), width=1)
        draw.text((bar_width + 4, y - 7), label, fill=(0, 0, 0), font=font)

    # Unit label above the gradient
    draw.text((bar_width + 4, 2), f"[{unit}]", fill=(80, 80, 80), font=font)

    return img


def plot_map(matrix, segment_name, output_path, dynamic=True):
    """
    Generate a clinical colormap image with color bar for one corneal segment.

    SCALING MODES
    -------------
    dynamic=True (default):
        Maps the patient's actual value range across all 40 colors.
        Maximizes visual contrast for that specific patient — every subtle
        variation becomes visible. This is what ophthalmologists expect
        from clinical topography software.
        Values are clipped to 2nd-98th percentile to prevent outlier pixels
        from compressing the useful range.

    dynamic=False:
        Uses the fixed population-level CLINICAL_BOUNDS for this segment.
        Same color always means the same value across all patients.
        Better for cross-patient comparison and paper figures.

    COLOR ASSIGNMENTS:
      - Values within bounds → one of 40 clinical colors (via BoundaryNorm)
      - Values below range → first color (clamped)
      - Values above range → last color (clamped)
      - NaN (outside cornea + missing data) → white

    Parameters
    ----------
    matrix : np.ndarray
        2D Cartesian matrix (e.g., 512x512) with NaN where no data exists
        (outside the cornea boundary AND where the device couldn't capture).
    segment_name : str
        Internal segment name (key in CLINICAL_BOUNDS / SEGMENT_UNITS).
    output_path : str
        Full path for the output PNG file.
    dynamic : bool
        True = scale to patient's data range. False = fixed clinical bounds.
    """
    if dynamic:
        # Dynamic scaling: use the patient's actual value range.
        # Clip to 2nd-98th percentile to prevent outlier pixels (noise,
        # interpolation artifacts at the boundary) from compressing the
        # useful color range.
        valid_values = matrix[~np.isnan(matrix)]
        if len(valid_values) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = np.percentile(valid_values, 2)
            vmax = np.percentile(valid_values, 98)
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
        levels = np.linspace(vmin, vmax, 41).tolist()  # 41 boundaries → 40 bins
        chosen_cmap = CMAP
    else:
        # Static scaling: use fixed population-level bounds.
        bounds = CLINICAL_BOUNDS.get(segment_name)
        if bounds is not None:
            is_decreasing = bounds[0] > bounds[-1]
            levels = sorted(bounds)
            chosen_cmap = CMAP.reversed() if is_decreasing else CMAP
        else:
            # Fallback to dynamic if no static bounds defined
            valid_values = matrix[~np.isnan(matrix)]
            vmin = np.nanmin(valid_values) if len(valid_values) > 0 else 0.0
            vmax = np.nanmax(valid_values) if len(valid_values) > 0 else 1.0
            levels = np.linspace(vmin, vmax, 41).tolist()
            chosen_cmap = CMAP

    norm = mcolors.BoundaryNorm(levels, chosen_cmap.N)

    # Map values → colors via numpy (no matplotlib figure needed)
    normalized = norm(np.ma.masked_invalid(matrix.astype(float)))
    rgba = chosen_cmap(normalized)  # (H, W, 4) float [0, 1]

    # NaN → neutral gray (outside cornea + missing data gaps)
    # Mid-gray doesn't compete with the clinical palette and approximates
    # the MS-39's grayscale eye photograph background.
    rgba[np.isnan(matrix)] = [0.25, 0.25, 0.25, 1.0]

    # Build the map image
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    map_img = Image.fromarray(rgba_uint8, "RGBA")

    # Build the color bar
    unit = SEGMENT_UNITS.get(segment_name, "")
    bar_img = _build_color_bar(chosen_cmap, levels, unit, bar_height=map_img.height)

    # Compose: map on the left, color bar on the right, white gap between
    gap = 10
    total_width = map_img.width + gap + bar_img.width
    composed = Image.new("RGBA", (total_width, map_img.height), (255, 255, 255, 255))
    composed.paste(map_img, (0, 0))
    composed.paste(bar_img, (map_img.width + gap, 0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    composed.save(output_path)


# ==========================================================================
# PROCESSING
# ==========================================================================


def process_segments(
    raw_segments, output_folder, patient_name="patient", target_size=VISUAL_SIZE, dynamic=True
):
    """
    Generate colormap PNGs from pre-parsed MS39 data.

    Use this when you've already called parse_csv() and want to avoid
    reading the file again. For single-file convenience, use process_single_csv().

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of parse_csv().
    output_folder : str
        Where to save the per-patient subfolder with PNGs.
    patient_name : str
        Subfolder name (typically the CSV filename without extension).
    target_size : int
        Output image resolution (default 512).
    dynamic : bool
        True = dynamic scaling (per-patient contrast). False = static clinical bounds.

    Returns
    -------
    int
        Number of maps generated.
    """
    patient_folder = os.path.join(output_folder, patient_name)
    os.makedirs(patient_folder, exist_ok=True)

    count = 0
    for name in DEFAULT_SEGMENTS:
        if name not in raw_segments:
            continue

        raw_polar = raw_segments[name]

        # Step 1: Interpolate everything (fast path, no NaN → no griddata fallback).
        # This fills -1000 regions with interpolated values — intentionally.
        polar_clean = clean_polar_data(raw_polar, keep_missing=False)
        cartesian = polar_to_cartesian(polar_clean, target_size=target_size)

        # Step 2: Punch holes where the device couldn't capture.
        # Build a binary mask from the original -1000 positions, project it
        # to Cartesian the same way, threshold at 0.5 → sharp binary cutoff.
        # Then set those pixels to NaN so they render as white.
        polar_with_sentinels = clean_polar_data(raw_polar, keep_missing=True)
        mask = missing_data_mask(polar_with_sentinels, target_size=target_size)
        cartesian[mask] = np.nan

        png_path = os.path.join(patient_folder, f"{name}.png")
        plot_map(cartesian, name, png_path, dynamic=dynamic)
        count += 1

    return count


def process_single_csv(filepath, output_folder, target_size=VISUAL_SIZE, dynamic=True):
    """
    Process one MS39 CSV: parse → cartesian → filter → colormap PNGs.

    Convenience wrapper: parses the file then calls process_segments().
    For batch processing or when combining with other pipelines,
    call parse_csv() once and pass the result to process_segments() directly.
    """
    raw_segments = parse_csv(filepath)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return process_segments(raw_segments, output_folder, basename, target_size, dynamic)


def process_folder(input_folder, output_folder, target_size=VISUAL_SIZE, dynamic=True):
    """Process all CSV files in a folder."""
    os.makedirs(output_folder, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(".csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", input_folder)
        return

    scale = "dynamic" if dynamic else "static"
    logger.info("Found %d CSV file(s) [%s scale]", len(csv_files), scale)
    total_start = time.time()

    for csv_file in csv_files:
        file_start = time.time()
        filepath = os.path.join(input_folder, csv_file)

        count = process_single_csv(filepath, output_folder, target_size, dynamic=dynamic)
        elapsed = time.time() - file_start
        logger.info("OK %s -> %d maps | %.2fs", csv_file, count, elapsed)

    logger.info("Done. %d files in %.2fs", len(csv_files), time.time() - total_start)


# ==========================================================================
# CLI
# ==========================================================================


def main():
    """Entry point for the corneaforge-visual console script."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="CorneaForge: Generate clinical colormap images from MS39 CSV exports.",
    )
    parser.add_argument("input", help="Path to a CSV file or folder of CSVs")
    parser.add_argument(
        "-o",
        "--output",
        default="./output_visual",
        help="Output folder for colormap PNGs (default: ./output_visual)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=VISUAL_SIZE,
        help=f"Output image size in pixels (default: {VISUAL_SIZE})",
    )
    parser.add_argument(
        "--static-scale",
        action="store_true",
        help="Use fixed population-level scales instead of per-patient dynamic scaling",
    )
    args = parser.parse_args()
    dynamic = not args.static_scale

    if os.path.isfile(args.input):
        t = time.time()
        count = process_single_csv(args.input, args.output, args.size, dynamic=dynamic)
        logger.info("Generated %d maps in %.2fs", count, time.time() - t)
    elif os.path.isdir(args.input):
        process_folder(args.input, args.output, args.size, dynamic=dynamic)
    else:
        logger.error("'%s' is not a valid file or directory", args.input)
        sys.exit(1)


if __name__ == "__main__":
    main()
