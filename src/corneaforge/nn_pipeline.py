"""
CorneaForge Neural Network Pipeline
=====================================

Converts MS39 CSV exports into model-ready tensors for deep learning.

INPUT:  One MS39 .csv file (or a folder of them)
OUTPUT: One .npz file per patient containing a (13, 224, 224) float32 tensor

Each of the 13 channels is one corneal measurement map, stored in its
original physical units:

    Channel  Segment                        Unit   What it measures
    -------  -----------------------------  ----   ----------------------------------------
     0       sagittal_anterior              mm     Front curvature radius (sagittal)
     1       tangential_anterior            mm     Front curvature radius (tangential)
     2       gaussian_anterior              mm     Front curvature radius (Gaussian mean)
     3       sagittal_posterior             mm     Back curvature radius (sagittal)
     4       tangential_posterior           mm     Back curvature radius (tangential)
     5       gaussian_posterior             mm     Back curvature radius (Gaussian mean)
     6       refra_frontal_power_anterior   D      Front surface refractive power
     7       refra_frontal_power_posterior  D      Back surface refractive power
     8       refra_equivalent_power         D      Total corneal refractive power
     9       corneal_thickness              µm     Full corneal pachymetry
    10       stromal_thickness              µm     Stromal layer thickness
    11       epithelial_thickness           µm     Epithelial layer thickness
    12       anterior_chamber_depth         mm     Depth of anterior chamber

IMPORTANT NOTES FOR TRAINING
-----------------------------
1. VALUES ARE RAW PHYSICAL UNITS — channels have very different scales:
   curvature ~7mm, thickness ~500µm, power ~44D. You MUST normalize
   in your DataLoader. Per-channel standardization (zero-mean, unit-variance)
   computed on your training set is the standard approach.

2. NaN → 0 REPLACEMENT — The cornea is circular but the tensor is square.
   Pixels outside the cornea (the square corners, ~22% of pixels) are set
   to 0.0. This is the same convention as zero-padding in convolutions.
   All valid corneal measurements are non-zero, so the network can
   trivially distinguish "no data" from "real measurement."

3. NO FILTERING APPLIED — Unlike the visual pipeline, there is no bilateral
   filter or any smoothing. The CNN receives the interpolated data as-is,
   preserving all spatial frequency content for the model to learn from.

5. -1000 (MISSING DATA) IS REMOVED, NOT KEPT — The MS39 records -1000
   where it couldn't capture a measurement (eyelid obstruction, blink, poor
   fixation). These gaps are almost exclusively peripheral. While some causes
   of missing data CAN be disease-related (e.g., a patient unable to open
   the eyelid properly may indicate an ocular surface condition), the same
   pattern is also produced by non-clinical factors: technician skill,
   patient fatigue, natural lid anatomy, or simply a blink at the wrong
   moment. On small medical datasets, these confounding variables dominate.
   A CNN would learn "missing periphery = disease" as a shortcut instead
   of learning actual corneal morphology — cutting a small amount of real
   signal but a large amount of noise. Therefore, -1000 points are replaced
   with NaN before interpolation, and the interpolator fills them from
   neighboring valid measurements.
   The visual pipeline DOES show -1000 as gaps for ophthalmologists,
   who have the clinical context to judge what the missing data means.

4. WHY 224×224 — This matches the standard input size for ImageNet-pretrained
   models (ViT, DINOv2, ConvNeXt, EfficientNet-B0, RetFound). Using 224
   directly means no resizing in your DataLoader → no additional interpolation
   artifacts. Given the native data resolution (~54-81 effective pixels across),
   224 is 3-4× oversampled — enough for smooth convolution kernels without
   wasting memory on empty detail.

Usage
-----
    # Single file
    python nn_pipeline.py /path/to/patient.csv -o ./tensors/

    # Whole folder
    python nn_pipeline.py /path/to/csv_folder/ -o ./tensors/

    # Custom resolution (e.g., for EfficientNet-B4 at 380×380)
    python nn_pipeline.py /path/to/csv_folder/ -o ./tensors/ --size 380
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

from corneaforge.core import DEFAULT_SEGMENTS, clean_polar_data, parse_csv, polar_to_cartesian

logger = logging.getLogger("corneaforge.nn")

TARGET_SIZE = 224


def process_segments(raw_segments, target_size=TARGET_SIZE):
    """
    Convert pre-parsed MS39 segments into a stacked multi-channel tensor.

    Use this when you've already called parse_csv() and want to avoid
    reading the file again. For single-file convenience, use process_single_csv().

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of parse_csv(). Keys are segment names, values are polar arrays.
    target_size : int
        Output spatial dimensions (target_size × target_size).

    Returns
    -------
    tensor : np.ndarray or None
        Shape (n_channels, target_size, target_size), dtype float32.
        None if no valid segments were found.
    segments : list[str]
        Ordered list of segment names matching the tensor channels.
    """
    channels = []
    found_segments = []

    for name in DEFAULT_SEGMENTS:
        if name not in raw_segments:
            logger.warning("Segment '%s' not found", name)
            continue

        # Clean: -1000 → NaN, drop all-NaN padding rows
        polar_data = clean_polar_data(raw_segments[name])

        # Convert polar → cartesian image grid
        cartesian = polar_to_cartesian(polar_data, target_size=target_size)

        # Replace NaN with 0 for the neural network
        # (NaN is not valid in PyTorch/TF tensors; 0 = standard conv padding)
        cartesian = np.nan_to_num(cartesian, nan=0.0)

        channels.append(cartesian)
        found_segments.append(name)

    if not channels:
        return None, []

    # Stack into (C, H, W) — the standard tensor format for PyTorch/TF
    return np.stack(channels, axis=0), found_segments


def process_single_csv(filepath, target_size=TARGET_SIZE):
    """
    Convert one MS39 CSV into a stacked multi-channel tensor.

    Convenience wrapper: parses the file then calls process_segments().
    For batch processing or when combining with other pipelines,
    call parse_csv() once and pass the result to process_segments() directly.
    """
    raw_segments = parse_csv(filepath)
    return process_segments(raw_segments, target_size)


def process_folder(input_folder, output_folder, target_size=TARGET_SIZE):
    """Process all CSV files in a folder, saving one .npz per file."""
    os.makedirs(output_folder, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(".csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", input_folder)
        return

    logger.info("Found %d CSV file(s)", len(csv_files))
    total_start = time.time()

    for csv_file in csv_files:
        file_start = time.time()
        filepath = os.path.join(input_folder, csv_file)

        tensor, segments = process_single_csv(filepath, target_size)
        if tensor is None:
            logger.warning("SKIP %s — no valid segments", csv_file)
            continue

        # Save as compressed numpy archive
        # WHY .npz:
        # - Native numpy format — no extra dependencies
        # - Supports multiple named arrays in one file
        # - Compressed: ~2-3× smaller than raw, negligible load time cost
        # - Trivial to use in a PyTorch Dataset: np.load(path)['data']
        output_name = os.path.splitext(csv_file)[0] + ".npz"
        output_path = os.path.join(output_folder, output_name)
        np.savez_compressed(
            output_path,
            data=tensor,  # (13, 224, 224) float32
            segments=np.array(segments),  # segment names for reference
        )

        elapsed = time.time() - file_start
        logger.info("OK %s -> %s | %s | %.2fs", csv_file, output_name, tensor.shape, elapsed)

    logger.info("Done. %d files in %.2fs", len(csv_files), time.time() - total_start)


# ==========================================================================
# CLI
# ==========================================================================
# WHY ARGPARSE:
# Without it, you'd edit the script every time you change a path or setting.
# With it, you run: python nn_pipeline.py /path/to/data -o /path/to/output
# It also gives you --help for free, type checking, and clear error messages.
# This is a small addition that makes the script usable by anyone without
# reading the source code.


def main():
    """Entry point for the corneaforge-nn console script."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="CorneaForge: Convert MS39 CSV exports to model-ready tensors.",
    )
    parser.add_argument("input", help="Path to a CSV file or folder of CSVs")
    parser.add_argument(
        "-o",
        "--output",
        default="./output_nn",
        help="Output folder for .npz files (default: ./output_nn)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=TARGET_SIZE,
        help=f"Output image size in pixels (default: {TARGET_SIZE})",
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        os.makedirs(args.output, exist_ok=True)
        t = time.time()
        tensor, segments = process_single_csv(args.input, args.size)
        if tensor is not None:
            out = os.path.join(
                args.output,
                os.path.splitext(os.path.basename(args.input))[0] + ".npz",
            )
            np.savez_compressed(out, data=tensor, segments=np.array(segments))
            logger.info("Saved %s | %s | %.2fs", out, tensor.shape, time.time() - t)
        else:
            logger.error("No valid segments found.")
    elif os.path.isdir(args.input):
        process_folder(args.input, args.output, args.size)
    else:
        logger.error("'%s' is not a valid file or directory", args.input)
        sys.exit(1)


if __name__ == "__main__":
    main()
