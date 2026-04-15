#!/usr/bin/env python3
"""
Granular profiler: breaks down time within _biquad_eval_batch
(the real bottleneck inside compute_opd_wavefront).
"""

import sys
import time
from collections import defaultdict

import numpy as np

from corneaforge import computed_indices as ci
from corneaforge.core import parse_csv, parse_metadata

_timings = defaultdict(float)
_call_counts = defaultdict(int)


def _wrap(name, orig):
    def wrapper(*a, **kw):
        t0 = time.monotonic()
        r = orig(*a, **kw)
        _timings[name] += time.monotonic() - t0
        _call_counts[name] += 1
        return r
    return wrapper


# Wrap the sub-steps inside the biquad pipeline
ci._bq_gather_neighbors = _wrap("gather_neighbors", ci._bq_gather_neighbors)
ci._bq_solve_centered_chunked = _wrap("solve_centered_chunked", ci._bq_solve_centered_chunked)
ci._bq_solve_absolute_chunked = _wrap("solve_absolute_chunked", ci._bq_solve_absolute_chunked)
ci._bq_get_surface = _wrap("get_surface (polar→cart+tree)", ci._bq_get_surface)
ci._bq_polar_to_cartesian = _wrap("polar_to_cartesian", ci._bq_polar_to_cartesian)

# Wrap the Newton loop and other parts
ci._newton_posterior_intersection = _wrap("newton_posterior", ci._newton_posterior_intersection)
ci.snells_law_vector_batch = _wrap("snells_law_batch", ci.snells_law_vector_batch)
ci.estimate_focal_point = _wrap("estimate_focal_point", ci.estimate_focal_point)
ci._fit_zernike_coefficients = _wrap("fit_zernike", ci._fit_zernike_coefficients)
ci.create_ray_grid = _wrap("create_ray_grid", ci.create_ray_grid)


def main():
    csv_path = sys.argv[1]
    metadata = parse_metadata(csv_path)
    raw = parse_csv(csv_path)

    _timings.clear()
    _call_counts.clear()

    t0 = time.monotonic()
    result = ci.compute_opd_wavefront(raw, metadata)
    total = time.monotonic() - t0

    print(f"\ncompute_opd_wavefront: {total:.4f}s  ({len(result)} keys)")
    print(f"\n{'Function':40s} {'Calls':>8s} {'Total(s)':>10s} {'Avg(ms)':>10s} {'%':>7s}")
    print("─" * 70)

    for name in sorted(_timings, key=lambda k: _timings[k], reverse=True):
        t = _timings[name]
        n = _call_counts[name]
        avg = (t / n * 1000) if n else 0
        print(f"  {name:38s} {n:8d} {t:10.4f} {avg:10.3f} {t/total*100:6.1f}%")

    print(f"\n  Unaccounted: {total - sum(_timings.values()):.4f}s")


if __name__ == "__main__":
    main()
