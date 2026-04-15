#!/usr/bin/env python3
"""
Deep-profile compute_opd_wavefront to find where the 4+ seconds go.

Monkey-patches the inner functions to add per-call timing, then prints
a breakdown by diameter and sub-operation.
"""

import sys
import time
from collections import defaultdict

import numpy as np

from corneaforge import computed_indices as ci
from corneaforge.core import parse_csv, parse_metadata

# ── Patching: wrap key inner functions with timing ─────────────────────

_timings = defaultdict(float)
_call_counts = defaultdict(int)


def _make_timed(name, orig_func):
    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        result = orig_func(*args, **kwargs)
        _timings[name] += time.monotonic() - t0
        _call_counts[name] += 1
        return result
    return wrapper


# Wrap the two main sub-functions
_orig_rt_one = ci._opd_raytrace_one_diameter
_orig_rt_dual = ci._opd_raytrace_dual_surface
_orig_biquad_eval = ci._biquad_eval_batch
_orig_biquad_dual = ci._biquad_eval_dual_batch
_orig_newton = ci._newton_posterior_intersection
_orig_snell = ci.snells_law_vector_batch
_orig_focal = ci.estimate_focal_point
_orig_zernike = ci._fit_zernike_coefficients
_orig_ray_grid = ci.create_ray_grid

ci._opd_raytrace_one_diameter = _make_timed("raytrace_one_diameter", _orig_rt_one)
ci._opd_raytrace_dual_surface = _make_timed("raytrace_dual_surface", _orig_rt_dual)
ci._biquad_eval_batch = _make_timed("biquad_eval_batch", _orig_biquad_eval)
ci._biquad_eval_dual_batch = _make_timed("biquad_eval_dual_batch", _orig_biquad_dual)
ci._newton_posterior_intersection = _make_timed("newton_posterior", _orig_newton)
ci.snells_law_vector_batch = _make_timed("snells_law_vector_batch", _orig_snell)
ci.estimate_focal_point = _make_timed("estimate_focal_point", _orig_focal)
ci._fit_zernike_coefficients = _make_timed("fit_zernike", _orig_zernike)
ci.create_ray_grid = _make_timed("create_ray_grid", _orig_ray_grid)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    metadata = parse_metadata(csv_path)
    raw = parse_csv(csv_path)

    _timings.clear()
    _call_counts.clear()

    t0 = time.monotonic()
    result = ci.compute_opd_wavefront(raw, metadata)
    total = time.monotonic() - t0

    print(f"\ncompute_opd_wavefront total: {total:.4f}s  ({len(result)} keys)")
    print(f"\n{'Function':40s} {'Calls':>8s} {'Total(s)':>10s} {'Avg(ms)':>10s} {'% of total':>10s}")
    print("─" * 80)

    for name in sorted(_timings, key=lambda k: _timings[k], reverse=True):
        t = _timings[name]
        n = _call_counts[name]
        avg_ms = (t / n * 1000) if n > 0 else 0
        pct = t / total * 100
        print(f"  {name:38s} {n:8d} {t:10.4f} {avg_ms:10.3f} {pct:9.1f}%")

    print(f"\n  (Unaccounted overhead: {total - sum(_timings.values()):.4f}s)")


if __name__ == "__main__":
    main()
