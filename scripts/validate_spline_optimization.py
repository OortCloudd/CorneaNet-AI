#!/usr/bin/env python3
"""
Validate spline optimization against biquadratic baseline.

Runs compute_opd_wavefront twice — once with the original biquadratic
surface evaluation, once with the spline replacement — and compares
every output key.

Usage:
    python scripts/validate_spline_optimization.py <csv_path> [<csv_path2> ...]
"""

import sys
import time

import numpy as np

from corneaforge import computed_indices as ci
from corneaforge.core import parse_csv, parse_metadata

# Tolerances for the comparison
# 0.1 µm is λ/5 — below diffraction limit, well within MS-39 repeatability.
# 5% relative covers derived metrics (cylax, lca) that amplify small diffs.
ABS_TOL = 0.1    # absolute tolerance (µm for Zernike coefficients)
REL_TOL = 0.05   # relative tolerance (5%)


def run_opd(raw, metadata, use_spline: bool) -> tuple[dict, float]:
    """Run compute_opd_wavefront with either biquad or spline surface eval.

    The OPD callers now reference spline functions by name.  To get a
    biquad baseline we temporarily point those names at the original
    biquadratic implementations AND restore the scalar Newton loop.
    """
    # Save current (spline) implementations
    orig_eval = ci._spline_eval_batch
    orig_newton_batch = ci._newton_posterior_batch_spline
    orig_dual_rt = ci._opd_raytrace_dual_surface

    if not use_spline:
        # Revert to biquadratic anterior eval
        ci._spline_eval_batch = ci._biquad_eval_batch
        # Revert to the old dual-surface raytrace (biquad posterior + scalar Newton).
        # We reconstruct it by patching the module names the old code path used.
        ci._opd_raytrace_dual_surface = _make_biquad_dual_raytrace()

    try:
        t0 = time.monotonic()
        result = ci.compute_opd_wavefront(raw, metadata)
        elapsed = time.monotonic() - t0
    finally:
        ci._spline_eval_batch = orig_eval
        ci._newton_posterior_batch_spline = orig_newton_batch
        ci._opd_raytrace_dual_surface = orig_dual_rt

    return result, elapsed


def _make_biquad_dual_raytrace():
    """Return the OLD _opd_raytrace_dual_surface that uses biquad + scalar Newton."""

    def _old_dual(ant_map, post_map, fitting_radius, offset_x=0.0, offset_y=0.0, _cache=None):
        if ant_map is None or post_map is None:
            return None, None
        max_data_radius = (ant_map.shape[0] - 1) * ci._R_STEP_MM
        if fitting_radius > max_data_radius:
            return None, None

        n_radial = int(fitting_radius / 0.1) + 1
        ray_xy = ci.create_ray_grid(fitting_radius, n_radial=n_radial, n_meridional=50)
        query_xy = np.column_stack((ray_xy[:, 0] - offset_x, ray_xy[:, 1] - offset_y))

        ant_z, ant_dzdx, ant_dzdy, post_coeffs, valid = ci._biquad_eval_dual_batch(
            ant_map, post_map, query_xy, _cache=_cache
        )
        n_valid = int(np.sum(valid))
        if n_valid < ci._ZERNIKE_MIN_POINTS:
            return None, None

        mask = valid
        x_v = ray_xy[mask, 0]
        y_v = ray_xy[mask, 1]
        xq_v = query_xy[mask, 0]
        yq_v = query_xy[mask, 1]
        ant_z_v = ant_z[mask]
        ant_dzdx_v = ant_dzdx[mask]
        ant_dzdy_v = ant_dzdy[mask]
        post_coeffs_v = post_coeffs[mask]
        nv = n_valid

        ant_points = np.column_stack((xq_v, yq_v, ant_z_v))
        entry_points = np.column_stack((xq_v, yq_v, np.zeros(nv)))
        ant_normals = ci.surface_normals_from_gradients(ant_dzdx_v, ant_dzdy_v)

        incident_dirs = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (nv, 3)).copy()
        try:
            refracted_ant = ci.snells_law_vector_batch(incident_dirs, ant_normals, ci.N_AIR, ci.N_CORNEA)
        except ValueError:
            return None, None

        post_points = np.full((nv, 3), np.nan)
        post_normals = np.full((nv, 3), np.nan)
        ray_valid = np.ones(nv, dtype=bool)
        for i in range(nv):
            result = ci._newton_posterior_intersection(post_coeffs_v[i], refracted_ant[i], ant_points[i])
            if result is None:
                ray_valid[i] = False
                continue
            post_points[i] = result
            pc = post_coeffs_v[i]
            px, py = result[0], result[1]
            pdzdx = 2.0 * pc[0] * px + pc[2] + pc[4] * py
            pdzdy = 2.0 * pc[1] * py + pc[3] + pc[4] * px
            nn = np.array([pdzdx, pdzdy, -1.0])
            nn /= np.linalg.norm(nn)
            post_normals[i] = nn

        if np.sum(ray_valid) < ci._ZERNIKE_MIN_POINTS:
            return None, None

        rv = ray_valid
        x_v2, y_v2 = x_v[rv], y_v[rv]
        entry_points2 = entry_points[rv]
        ant_points2 = ant_points[rv]
        refracted_ant2 = refracted_ant[rv]
        post_points2 = post_points[rv]
        post_normals2 = post_normals[rv]

        try:
            refracted_post = ci.snells_law_vector_batch(refracted_ant2, post_normals2, ci.N_CORNEA, ci.N_AQUEOUS)
        except ValueError:
            return None, None
        try:
            focal_x, focal_y, focal_z = ci.estimate_focal_point(post_points2, refracted_post)
        except ValueError:
            return None, None

        focal_point = np.array([focal_x, focal_y, focal_z])
        dt = (focal_z - post_points2[:, 2]) / refracted_post[:, 2]
        fp_xy = post_points2[:, :2] + dt[:, np.newaxis] * refracted_post[:, :2]
        fp_dist = np.sqrt(fp_xy[:, 0] ** 2 + fp_xy[:, 1] ** 2)
        ray_ok = fp_dist <= 2.0
        if np.sum(ray_ok) < ci._ZERNIKE_MIN_POINTS:
            return None, None

        x_v3, y_v3 = x_v2[ray_ok], y_v2[ray_ok]
        entry_points3 = entry_points2[ray_ok]
        ant_points3 = ant_points2[ray_ok]
        post_points3 = post_points2[ray_ok]
        refracted_post3 = refracted_post[ray_ok]

        try:
            focal_x, focal_y, focal_z = ci.estimate_focal_point(post_points3, refracted_post3)
        except ValueError:
            return None, None
        focal_point = np.array([focal_x, focal_y, focal_z])

        d_air = np.linalg.norm(ant_points3 - entry_points3, axis=1)
        d_cornea = np.linalg.norm(post_points3 - ant_points3, axis=1)
        fp_vec = focal_point[np.newaxis, :] - post_points3
        d_aqueous = np.abs(np.sum(fp_vec * refracted_post3, axis=1))
        opl = ci.N_AIR * d_air + ci.N_CORNEA * d_cornea + ci.N_AQUEOUS * d_aqueous
        opd = opl - np.mean(opl)

        opd_ok = np.abs(opd) <= 1.0
        if np.sum(opd_ok) < ci._ZERNIKE_MIN_POINTS:
            return None, None
        x_final, y_final = x_v3[opd_ok], y_v3[opd_ok]
        opd_final = opd[opd_ok] - np.mean(opd[opd_ok])

        coeffs_um, _ = ci._fit_zernike_coefficients(x_final, y_final, opd_final, fitting_radius, max_order=7)
        if coeffs_um is None:
            return None, None
        coeffs_um[0] = 0.0
        coeffs_um[4] = 0.0
        return coeffs_um, focal_z

    return _old_dual


def compare(baseline: dict, optimized: dict, label: str):
    """Compare two OPD result dicts key by key."""
    all_keys = sorted(set(baseline) | set(optimized))
    n_keys = len(all_keys)
    n_both_none = 0
    n_match = 0
    n_close = 0
    n_differ = 0
    n_missing = 0  # key in one but not the other
    worst_abs = 0.0
    worst_rel = 0.0
    worst_key = ""
    diffs = []

    for key in all_keys:
        bv = baseline.get(key)
        ov = optimized.get(key)

        if key not in baseline or key not in optimized:
            n_missing += 1
            continue

        if bv is None and ov is None:
            n_both_none += 1
            continue

        if bv is None or ov is None:
            n_differ += 1
            diffs.append((key, bv, ov, "None vs value"))
            continue

        abs_diff = abs(bv - ov)
        rel_diff = abs_diff / max(abs(bv), 1e-12)

        if abs_diff < 1e-12:
            n_match += 1
        elif abs_diff <= ABS_TOL or rel_diff <= REL_TOL:
            n_close += 1
        else:
            n_differ += 1
            diffs.append((key, bv, ov, f"abs={abs_diff:.6f} rel={rel_diff:.6f}"))

        if abs_diff > worst_abs:
            worst_abs = abs_diff
            worst_rel = rel_diff
            worst_key = key

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  Total keys:     {n_keys}")
    print(f"  Both None:      {n_both_none}")
    print(f"  Exact match:    {n_match}")
    print(f"  Within tol:     {n_close}  (abs<{ABS_TOL}, rel<{REL_TOL})")
    print(f"  DIFFER:         {n_differ}")
    print(f"  Missing:        {n_missing}")
    print(f"  Worst abs diff: {worst_abs:.8f}  ({worst_key})")
    print(f"  Worst rel diff: {worst_rel:.8f}")

    if diffs:
        print(f"\n  First 10 differences:")
        for key, bv, ov, detail in diffs[:10]:
            print(f"    {key}: baseline={bv}  optimized={ov}  ({detail})")

    return n_differ == 0 and n_missing == 0


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_path> [<csv_path2> ...]")
        sys.exit(1)

    all_pass = True
    for csv_path in sys.argv[1:]:
        print(f"\n{'=' * 60}")
        print(f"  File: {csv_path}")
        print(f"{'=' * 60}")

        metadata = parse_metadata(csv_path)
        raw = parse_csv(csv_path)

        # Run baseline (biquadratic)
        baseline, t_base = run_opd(raw, metadata, use_spline=False)
        print(f"\n  Biquadratic: {t_base:.4f}s  ({len(baseline)} keys)")

        # Run optimized (spline)
        optimized, t_opt = run_opd(raw, metadata, use_spline=True)
        print(f"  Spline:      {t_opt:.4f}s  ({len(optimized)} keys)")
        print(f"  Speedup:     {t_base / t_opt:.2f}x")

        passed = compare(baseline, optimized, "Comparison")

        if passed:
            print(f"\n  ✓ PASS — all {len(baseline)} keys match within tolerance")
        else:
            print(f"\n  ✗ FAIL — results differ beyond tolerance")
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
