#!/usr/bin/env python3
"""
Profile every operation inside _extract_features to find the bottleneck.

Usage:
    python scripts/profile_extract.py <csv_path> [<csv_path2> ...]
"""

import sys
import time

from corneaforge.computed_indices import (
    compute_abcd_staging,
    compute_epithelial_refraction,
    compute_epithelial_sectors,
    compute_k_readings,
    compute_opd_wavefront,
    compute_screening_extrema,
    compute_screening_indices,
    compute_shape_indices,
    compute_summary_indices,
    compute_zernike_indices,
)
from corneaforge.core import parse_csv, parse_metadata
from corneaforge.descriptive_stats import process_segments as stats_process
from corneaforge.nn_pipeline import process_segments as nn_process


def _timed(label: str, func, *args, **kwargs):
    """Run *func* and print elapsed wall time."""
    t0 = time.monotonic()
    result = func(*args, **kwargs)
    dt = time.monotonic() - t0
    print(f"  {label:40s} {dt:8.4f}s")
    return result, dt


def profile_one(csv_path: str):
    print(f"\n{'=' * 70}")
    print(f"File: {csv_path}")
    print(f"{'=' * 70}")

    total_t0 = time.monotonic()

    # ── Parsing ──────────────────────────────────────────────────────
    print("\n[Parsing]")
    metadata, _ = _timed("parse_metadata", parse_metadata, csv_path)
    raw, _ = _timed("parse_csv", parse_csv, csv_path)

    # ── Descriptive stats ────────────────────────────────────────────
    print("\n[Descriptive stats]")
    stats, dt_stats = _timed("stats_process", stats_process, raw, metadata)

    # ── Computed indices (loop batch) ────────────────────────────────
    print("\n[Computed indices — loop batch]")
    compute_calls = [
        ("compute_summary_indices", compute_summary_indices, (raw,)),
        ("compute_k_readings", compute_k_readings, (raw, metadata)),
        ("compute_shape_indices", compute_shape_indices, (raw,)),
        ("compute_screening_indices", compute_screening_indices, (raw, metadata)),
        ("compute_abcd_staging", compute_abcd_staging, (raw,)),
        ("compute_epithelial_sectors", compute_epithelial_sectors, (raw, metadata)),
    ]

    all_indices = {}
    for name, func, args in compute_calls:
        try:
            result, _ = _timed(name, func, *args)
            all_indices.update(result)
        except Exception as e:
            print(f"  {name:40s} FAILED: {e}")

    # ── Computed indices (sequential) ────────────────────────────────
    print("\n[Computed indices — sequential]")
    zernike = {}
    try:
        zernike, _ = _timed("compute_zernike_indices", compute_zernike_indices, raw, metadata)
        all_indices.update(zernike)
    except Exception as e:
        print(f"  {'compute_zernike_indices':40s} FAILED: {e}")

    try:
        result, _ = _timed(
            "compute_screening_extrema",
            compute_screening_extrema,
            raw,
            metadata,
            zernike_results=zernike,
        )
        all_indices.update(result)
    except Exception as e:
        print(f"  {'compute_screening_extrema':40s} FAILED: {e}")

    try:
        result, _ = _timed("compute_epithelial_refraction", compute_epithelial_refraction, raw, metadata)
        all_indices.update(result)
    except Exception as e:
        print(f"  {'compute_epithelial_refraction':40s} FAILED: {e}")

    try:
        result, _ = _timed("compute_opd_wavefront", compute_opd_wavefront, raw, metadata)
        all_indices.update(result)
    except Exception as e:
        print(f"  {'compute_opd_wavefront':40s} FAILED: {e}")

    # ── NN tensor ────────────────────────────────────────────────────
    print("\n[NN pipeline]")
    tensor, dt_nn = _timed("nn_process", nn_process, raw)

    # ── Summary ──────────────────────────────────────────────────────
    total = time.monotonic() - total_t0
    print(f"\n{'─' * 70}")
    print(f"  TOTAL                                    {total:8.4f}s")
    print(f"  Indices computed: {len(all_indices)}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_path> [<csv_path2> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        profile_one(path)


if __name__ == "__main__":
    main()
