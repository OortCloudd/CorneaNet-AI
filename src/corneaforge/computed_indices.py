"""
CorneaForge Computed Indices
============================

Recomputes the 16 topographic-derived summary indices that the MS-39
normally places in the ms39_global CSV export. This removes the need
for the global export (which is slow, recalculates on-the-fly, and
exports all patients instead of one).

All indices are computed from the polar maps in the ms39_individual CSV
using the algorithm documented in formulas_summary_indices.md:

  For each valid cell (r, theta) where map[r, theta] != -1000:
      x = r_mm * cos(theta_rad)
      y = r_mm * sin(theta_rad)
      Track the global min (or max) and its (x, y) location.

Output keys match the ms39_global CSV column names for validation.
"""

import math
from collections.abc import Callable

import numpy as np
from scipy.optimize import least_squares as _scipy_least_squares
from scipy.spatial import cKDTree

# ===========================================================================
# Ray-tracing primitives (formerly ray_trace.py)
# ===========================================================================

N_AIR = 1.0
N_CORNEA = 1.376  # Gullstrand corneal stroma
N_KERATOMETRIC = 1.3375  # keratometric (fictitious) index
N_AQUEOUS = 1.336  # aqueous humor

_DEFAULT_N_RADIAL = 20
_DEFAULT_N_MERIDIONAL = 50


def create_ray_grid(
    fitting_radius: float,
    n_radial: int = _DEFAULT_N_RADIAL,
    n_meridional: int = _DEFAULT_N_MERIDIONAL,
) -> np.ndarray:
    """Create a grid of ray entry points across the pupil aperture."""
    radii = np.linspace(0.0, fitting_radius, n_radial)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_meridional, endpoint=False)
    rr, tt = np.meshgrid(radii, thetas, indexing="ij")
    x = (rr * np.cos(tt)).ravel()
    y = (rr * np.sin(tt)).ravel()
    return np.column_stack((x, y))


def surface_normals_from_gradients(dz_dx: np.ndarray, dz_dy: np.ndarray) -> np.ndarray:
    """Vectorized outward unit surface normals from gradient arrays."""
    N = len(dz_dx)
    normals = np.empty((N, 3))
    normals[:, 0] = dz_dx
    normals[:, 1] = dz_dy
    normals[:, 2] = -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= norms
    return normals


def snells_law_vector_batch(
    incident_dirs: np.ndarray,
    surface_normals: np.ndarray,
    n1: float,
    n2: float,
) -> np.ndarray:
    """Vectorized Snell's law for N rays simultaneously."""
    cos_i = -np.sum(incident_dirs * surface_normals, axis=1)
    ratio = n1 / n2
    sin2_t = ratio * ratio * (1.0 - cos_i * cos_i)
    if np.any(sin2_t > 1.0):
        bad = np.where(sin2_t > 1.0)[0]
        raise ValueError(
            f"Total internal reflection at {len(bad)} ray(s), "
            f"max sin^2(t) = {sin2_t[bad].max():.6f}"
        )
    cos_t = np.sqrt(1.0 - sin2_t)
    factor = ratio * cos_i - cos_t
    refracted = ratio * incident_dirs + factor[:, np.newaxis] * surface_normals
    norms = np.linalg.norm(refracted, axis=1, keepdims=True)
    refracted /= norms
    return refracted


def estimate_focal_point(
    ray_positions: np.ndarray,
    ray_directions: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate the least-squares focal point from refracted rays (3x3 system)."""
    Px, Py, Pz = ray_positions[:, 0], ray_positions[:, 1], ray_positions[:, 2]
    Dz = ray_directions[:, 2]
    tx = ray_directions[:, 0] / Dz
    ty = ray_directions[:, 1] / Dz
    N = len(Px)
    sum_tx = np.sum(tx)
    sum_ty = np.sum(ty)
    A = np.array(
        [
            [-N, 0.0, sum_tx],
            [0.0, -N, sum_ty],
            [-sum_tx, -sum_ty, np.sum(tx * tx + ty * ty)],
        ]
    )
    b = np.array(
        [
            np.sum(Pz * tx - Px),
            np.sum(Pz * ty - Py),
            np.sum(-tx * Px - ty * Py + Pz * (tx * tx + ty * ty)),
        ]
    )
    try:
        focal = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError("Singular system -- cannot estimate focal point")
    return float(focal[0]), float(focal[1]), float(focal[2])


def _rt_compute_opl(
    entry_points: np.ndarray,
    surface_points: np.ndarray,
    n1: float,
    n2: float,
    reference_z: float,
    refracted_dirs: np.ndarray,
) -> np.ndarray:
    """Compute optical path length for rays refracting through a surface."""
    d1 = np.linalg.norm(surface_points - entry_points, axis=1)
    dz = reference_z - surface_points[:, 2]
    dz_dir = refracted_dirs[:, 2]
    dz_dir = np.where(np.abs(dz_dir) < 1e-10, 1e-10, dz_dir)
    dt = dz / dz_dir
    ref_points = surface_points + dt[:, np.newaxis] * refracted_dirs
    d2 = np.linalg.norm(ref_points - surface_points, axis=1)
    return n1 * d1 + n2 * d2


def ray_trace_single_surface(
    ray_positions: np.ndarray,
    surface_eval_fn: Callable,
    n1: float,
    n2: float,
    incident_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Trace collimated rays through a single refracting surface."""
    if incident_dir is None:
        incident_dir = np.array([0.0, 0.0, 1.0])
    N = len(ray_positions)
    x = ray_positions[:, 0]
    y = ray_positions[:, 1]
    z, dz_dx, dz_dy = surface_eval_fn(x, y)
    surface_points = np.column_stack((x, y, z))
    entry_points = np.column_stack((x, y, np.zeros(N)))
    normals = surface_normals_from_gradients(dz_dx, dz_dy)
    incident_dirs = np.broadcast_to(incident_dir, (N, 3)).copy()
    refracted_dirs = snells_law_vector_batch(incident_dirs, normals, n1, n2)
    _, _, focal_z = estimate_focal_point(surface_points, refracted_dirs)
    opl = _rt_compute_opl(entry_points, surface_points, n1, n2, focal_z, refracted_dirs)
    opd = opl - np.mean(opl)
    return opd, refracted_dirs, surface_points, focal_z


def _rt_sphere_surface(R: float):
    """Return a surface_eval_fn for a sphere of radius R (for validation)."""

    def eval_fn(x, y):
        r2 = np.clip(x * x + y * y, 0, R * R - 1e-10)
        denom = np.sqrt(R * R - r2)
        z = R - denom
        dz_dx = x / denom
        dz_dy = y / denom
        return z, dz_dx, dz_dy

    return eval_fn


# ===========================================================================
# Biquadratic local surface evaluation (formerly surface_eval_biquad.py)
# ===========================================================================

_BQ_RADIUS_APEX = 0.41
_BQ_RADIUS_PERIPH = 0.30
_BQ_APEX_THRESHOLD = 0.15
_BQ_MIN_NEIGHBORS = 6
_BQ_FALLBACK_RADIUS = 0.50


def _bq_polar_to_cartesian(
    polar_map: np.ndarray,
    r_step: float = 0.2,
    missing: float = -1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a CSO polar elevation map to Cartesian coordinate arrays."""
    n_rows, n_cols = polar_map.shape
    theta = np.linspace(0, 2 * np.pi, n_cols, endpoint=False)
    radii = np.arange(n_rows) * r_step
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)
    mask = polar_map != missing
    xy = np.column_stack([xx[mask], yy[mask]])
    z = polar_map[mask].astype(np.float64)
    return xy, z


def _bq_find_neighbors(
    x0: float,
    y0: float,
    tree: cKDTree,
    xy: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Find data points within an adaptive radius of (x0, y0)."""
    dist_from_apex = np.hypot(x0, y0)
    radius = _BQ_RADIUS_APEX if dist_from_apex < _BQ_APEX_THRESHOLD else _BQ_RADIUS_PERIPH
    idx = tree.query_ball_point([x0, y0], radius)
    if len(idx) < _BQ_MIN_NEIGHBORS:
        idx = tree.query_ball_point([x0, y0], _BQ_FALLBACK_RADIUS)
    if len(idx) < _BQ_MIN_NEIGHBORS:
        return None
    idx = np.asarray(idx)
    return xy[idx, 0], xy[idx, 1], z[idx]


def _bq_fit(
    x_nbr: np.ndarray,
    y_nbr: np.ndarray,
    z_nbr: np.ndarray,
    x0: float,
    y0: float,
) -> tuple[float, float, float]:
    """Fit a local biquadratic surface centered at (x0, y0). Returns (z, dz_dx, dz_dy)."""
    n = len(x_nbr)
    if n < 6:
        raise ValueError(f"Need >= 6 neighbors for biquadratic fit, got {n}")
    dx = x_nbr - x0
    dy = y_nbr - y0
    A = np.column_stack([np.ones(n), dx, dy, dx * dx, dx * dy, dy * dy])
    coeffs, _, _, _ = np.linalg.lstsq(A, z_nbr, rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def _bq_fit_full(
    x_nbr: np.ndarray,
    y_nbr: np.ndarray,
    z_nbr: np.ndarray,
) -> np.ndarray:
    """Fit biquadratic in ABSOLUTE coordinates. Returns [a,b,c,d,e,f] (CSO order)."""
    n = len(x_nbr)
    if n < 6:
        raise ValueError(f"Need >= 6 neighbors for biquadratic fit, got {n}")
    A = np.column_stack([x_nbr * x_nbr, y_nbr * y_nbr, x_nbr, y_nbr, x_nbr * y_nbr, np.ones(n)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z_nbr, rcond=None)
    return coeffs


def _bq_eval(coeffs: np.ndarray, x: float, y: float) -> float:
    """Evaluate z = a*x^2 + b*y^2 + c*x + d*y + e*x*y + f."""
    return (
        coeffs[0] * x * x
        + coeffs[1] * y * y
        + coeffs[2] * x
        + coeffs[3] * y
        + coeffs[4] * x * y
        + coeffs[5]
    )


def _bq_normal(coeffs: np.ndarray, x: float, y: float) -> np.ndarray:
    """Compute outward unit normal from biquadratic coefficients (CSO convention)."""
    dz_dx = -(2.0 * coeffs[0] * x + coeffs[2] + coeffs[4] * y)
    dz_dy = -(2.0 * coeffs[1] * y + coeffs[3] + coeffs[4] * x)
    norm = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy + 1.0)
    return np.array([dz_dx / norm, dz_dy / norm, 1.0 / norm])


_BQ_TIKHONOV = 1e-8  # regularization for batched normal equations


def _bq_gather_neighbors(query_xy, tree):
    """Batch neighbor finding with adaptive radii (exact radius match).

    Uses ``query_ball_point`` on groups (apex / periphery) then a fallback
    pass for points with fewer than ``_BQ_MIN_NEIGHBORS``.

    Returns
    -------
    neighbor_lists : list[ndarray] — per-valid-point index arrays
    valid_mask : ndarray (M,) bool
    valid_idx  : ndarray
    """
    M = len(query_xy)
    if tree.n < _BQ_MIN_NEIGHBORS:
        empty = np.zeros(M, dtype=bool)
        return [], empty, np.where(empty)[0]

    dist_from_apex = np.hypot(query_xy[:, 0], query_xy[:, 1])
    is_apex = dist_from_apex < _BQ_APEX_THRESHOLD

    raw: list[list | np.ndarray] = [[] for _ in range(M)]

    apex_idx = np.where(is_apex)[0]
    if len(apex_idx):
        for j, nl in zip(apex_idx, tree.query_ball_point(query_xy[apex_idx], _BQ_RADIUS_APEX)):
            raw[j] = nl

    periph_idx = np.where(~is_apex)[0]
    if len(periph_idx):
        for j, nl in zip(
            periph_idx,
            tree.query_ball_point(query_xy[periph_idx], _BQ_RADIUS_PERIPH),
        ):
            raw[j] = nl

    counts = np.array([len(nl) for nl in raw], dtype=np.int32)
    need_fb = np.where(counts < _BQ_MIN_NEIGHBORS)[0]
    if len(need_fb):
        for j, nl in zip(
            need_fb,
            tree.query_ball_point(query_xy[need_fb], _BQ_FALLBACK_RADIUS),
        ):
            raw[j] = nl
            counts[j] = len(nl)

    valid_mask = counts >= _BQ_MIN_NEIGHBORS
    valid_idx = np.where(valid_mask)[0]
    neighbor_lists = [np.asarray(raw[qi]) for qi in valid_idx]
    return neighbor_lists, valid_mask, valid_idx


def _bq_solve_centered_chunked(neighbor_lists, xy, zv, query_xy_valid):
    """Batched centered biquadratic solve, chunked by neighbor count.

    Sorts points by count and processes in chunks so padding stays bounded.
    Returns z, dz_dx, dz_dy arrays of length n_valid.
    """
    n_valid = len(neighbor_lists)
    z_out = np.empty(n_valid)
    dzdx_out = np.empty(n_valid)
    dzdy_out = np.empty(n_valid)

    counts = np.array([len(nl) for nl in neighbor_lists])
    order = np.argsort(counts)

    CHUNK = 256
    for start in range(0, n_valid, CHUNK):
        idx = order[start : start + CHUNK]
        local_max_k = int(counts[idx].max())
        n_chunk = len(idx)

        nbr_xy_c = np.zeros((n_chunk, local_max_k, 2))
        nbr_z_c = np.zeros((n_chunk, local_max_k))
        cnts_c = np.empty(n_chunk, dtype=np.int32)
        for j, vi in enumerate(idx):
            nl = neighbor_lists[vi]
            k = len(nl)
            nbr_xy_c[j, :k] = xy[nl]
            nbr_z_c[j, :k] = zv[nl]
            cnts_c[j] = k

        k_range = np.arange(local_max_k)[None, :]
        mask = k_range < cnts_c[:, None]

        dx = nbr_xy_c[:, :, 0] - query_xy_valid[idx, 0:1]
        dy = nbr_xy_c[:, :, 1] - query_xy_valid[idx, 1:2]
        A = np.stack([np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy], axis=-1)
        A[~mask] = 0.0
        nbr_z_c[~mask] = 0.0

        AtA = np.einsum("bki,bkj->bij", A, A)
        AtA[:, range(6), range(6)] += _BQ_TIKHONOV
        Atz = np.einsum("bki,bk->bi", A, nbr_z_c)
        coeffs = np.linalg.solve(AtA, Atz[..., None])[..., 0]

        z_out[idx] = coeffs[:, 0]
        dzdx_out[idx] = coeffs[:, 1]
        dzdy_out[idx] = coeffs[:, 2]

    return z_out, dzdx_out, dzdy_out


def _bq_solve_absolute_chunked(neighbor_lists, xy, zv):
    """Batched absolute biquadratic solve (CSO: [x², y², x, y, xy, 1]), chunked.

    Returns (n_valid, 6) coefficient array.
    """
    n_valid = len(neighbor_lists)
    coeffs_out = np.empty((n_valid, 6))

    counts = np.array([len(nl) for nl in neighbor_lists])
    order = np.argsort(counts)

    CHUNK = 256
    for start in range(0, n_valid, CHUNK):
        idx = order[start : start + CHUNK]
        local_max_k = int(counts[idx].max())
        n_chunk = len(idx)

        nbr_xy_c = np.zeros((n_chunk, local_max_k, 2))
        nbr_z_c = np.zeros((n_chunk, local_max_k))
        cnts_c = np.empty(n_chunk, dtype=np.int32)
        for j, vi in enumerate(idx):
            nl = neighbor_lists[vi]
            k = len(nl)
            nbr_xy_c[j, :k] = xy[nl]
            nbr_z_c[j, :k] = zv[nl]
            cnts_c[j] = k

        k_range = np.arange(local_max_k)[None, :]
        mask = k_range < cnts_c[:, None]

        x = nbr_xy_c[:, :, 0]
        y = nbr_xy_c[:, :, 1]
        A = np.stack([x * x, y * y, x, y, x * y, np.ones_like(x)], axis=-1)
        A[~mask] = 0.0
        nbr_z_c[~mask] = 0.0

        AtA = np.einsum("bki,bkj->bij", A, A)
        AtA[:, range(6), range(6)] += _BQ_TIKHONOV
        Atz = np.einsum("bki,bk->bi", A, nbr_z_c)
        coeffs_out[idx] = np.linalg.solve(AtA, Atz[..., None])[..., 0]

    return coeffs_out


def _bq_get_surface(polar_map, r_step, missing, _cache):
    """Get or build (xy, zv, tree) for a polar map, with optional caching."""
    key = id(polar_map)
    if _cache is not None and key in _cache:
        return _cache[key]
    xy, zv = _bq_polar_to_cartesian(polar_map, r_step=r_step, missing=missing)
    tree = cKDTree(xy)
    entry = (xy, zv, tree)
    if _cache is not None:
        _cache[key] = entry
    return entry


def _biquad_eval_batch(
    polar_map: np.ndarray,
    query_xy: np.ndarray,
    r_step: float = 0.2,
    missing: float = -1000.0,
    _cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the biquadratic surface at many query points (vectorized)."""
    xy, zv, tree = _bq_get_surface(polar_map, r_step, missing, _cache)

    M = len(query_xy)
    z_out = np.full(M, np.nan)
    dzdx_out = np.full(M, np.nan)
    dzdy_out = np.full(M, np.nan)
    valid = np.zeros(M, dtype=bool)

    nls, valid_mask, valid_idx = _bq_gather_neighbors(query_xy, tree)
    if len(valid_idx) == 0:
        return z_out, dzdx_out, dzdy_out, valid

    z_v, dzdx_v, dzdy_v = _bq_solve_centered_chunked(nls, xy, zv, query_xy[valid_idx])
    z_out[valid_idx] = z_v
    dzdx_out[valid_idx] = dzdx_v
    dzdy_out[valid_idx] = dzdy_v
    valid[valid_idx] = True
    return z_out, dzdx_out, dzdy_out, valid


def _biquad_eval_dual_batch(
    polar_map_ant: np.ndarray,
    polar_map_post: np.ndarray,
    query_xy: np.ndarray,
    r_step: float = 0.2,
    missing: float = -1000.0,
    _cache: dict | None = None,
):
    """Evaluate anterior and posterior surfaces at many query points (vectorized)."""
    xy_ant, z_ant, tree_ant = _bq_get_surface(polar_map_ant, r_step, missing, _cache)
    xy_post, z_post, tree_post = _bq_get_surface(polar_map_post, r_step, missing, _cache)

    M = len(query_xy)
    ant_z_out = np.full(M, np.nan)
    ant_dzdx_out = np.full(M, np.nan)
    ant_dzdy_out = np.full(M, np.nan)
    post_coeffs_out = np.full((M, 6), np.nan)
    valid = np.zeros(M, dtype=bool)

    # --- anterior ---
    a_nls, a_vmask, a_vidx = _bq_gather_neighbors(query_xy, tree_ant)
    if len(a_vidx) == 0:
        return ant_z_out, ant_dzdx_out, ant_dzdy_out, post_coeffs_out, valid

    az, adx, ady = _bq_solve_centered_chunked(a_nls, xy_ant, z_ant, query_xy[a_vidx])
    ant_z_out[a_vidx] = az
    ant_dzdx_out[a_vidx] = adx
    ant_dzdy_out[a_vidx] = ady

    # --- posterior (only query points that passed anterior) ---
    p_nls, p_vmask_sub, p_vidx_sub = _bq_gather_neighbors(query_xy[a_vidx], tree_post)
    if len(p_vidx_sub) == 0:
        return ant_z_out, ant_dzdx_out, ant_dzdy_out, post_coeffs_out, valid

    pcoeffs = _bq_solve_absolute_chunked(p_nls, xy_post, z_post)
    both_idx = a_vidx[p_vidx_sub]
    post_coeffs_out[both_idx] = pcoeffs
    valid[both_idx] = True
    return ant_z_out, ant_dzdx_out, ant_dzdy_out, post_coeffs_out, valid


# Radial step size: 0.2mm between rings
_R_STEP_MM = 0.2

# 256 meridians, TABO convention
_N_COLS = 256
_THETA_RAD = np.arange(_N_COLS) * (2 * np.pi / _N_COLS)
_COS_THETA = np.cos(_THETA_RAD)
_SIN_THETA = np.sin(_THETA_RAD)

# Sentinel value for missing data in MS-39 polar maps
_MISSING = -1000


def _find_extremum(polar_map, mode="min", max_radius=None):
    """
    Find a local min or max of a polar map and its Cartesian location.

    Matches CSO ``GetMin`` / ``GetMax``.  A candidate cell must be a strict local
    extremum: less than (min) or greater than (max) all four cardinal
    neighbours in the polar grid.  Additionally, the cell and its
    surrounding 5x5 block must be hole-free (``CheckPoint``).

    **Why local, not global?**  The naive global argmin/argmax picks up
    peripheral OCT artifacts (e.g. negative stromal thickness at r > 4 mm)
    or boundary effects that CSO filters out via the local-extremum + holes
    check.  Switching to local search fixed StrThkMin (-54 -> 553),
    EpiThkMin coordinates, and DZMaxB coordinates.

    Parameters
    ----------
    polar_map : np.ndarray, shape (n_rows, 256)
        Raw polar data with -1000 for missing cells.
    mode : str
        "min" or "max".
    max_radius : float or None
        CSO's ``Limit`` parameter.  Row ``i`` is eligible only when
        ``(i + 1) * _R_STEP_MM < max_radius``.  CSO applies 3.9 for
        epithelial maps and 4.0 for DZMax / Gaussian posterior.
        When None (default), no limit is applied (CSO's float.MaxValue).
        The ``CheckPoint`` radius is always clamped to 4.0 mm for min
        searches (CSO hardcodes ``4f``), and to ``max_radius`` for max
        searches.

    Returns
    -------
    value : float or None
        The extremum value, or None if the entire map is missing.
    x_mm : float
        X-coordinate from corneal vertex (mm).
    y_mm : float
        Y-coordinate from corneal vertex (mm).
    """
    n_rows, n_cols = polar_map.shape
    vals = polar_map.astype(np.float64)
    # Build a boolean "holes" mask: True where data is missing
    holes = (vals == _MISSING) | np.isnan(vals)

    if np.all(holes):
        return None, 0.0, 0.0

    is_min = mode == "min"
    neighbor_check = 2  # CSO hardcodes N=2

    # CSO's CheckPoint uses MaxR = 4.0f (hardcoded) for GetMin,
    # and MaxR = Limit for GetMax.
    if is_min:
        checkpoint_radius = 4.0
    else:
        checkpoint_radius = max_radius if max_radius is not None else 100.0

    # CSO initialises best to vals[0, 0]; if center is already the extremum
    # and nothing beats it, it returns that value at (0, 0).
    best_row, best_col = 0, 0
    best_val = float(vals[0, 0]) if not holes[0, 0] else (np.inf if is_min else -np.inf)

    for i in range(1, n_rows - 1):
        # CSO: ro[i+1] < Limit
        if max_radius is not None and (i + 1) * _R_STEP_MM >= max_radius:
            continue
        # CSO CheckPoint: ro[i] > MaxR -> reject
        if i * _R_STEP_MM > checkpoint_radius:
            continue

        for j in range(n_cols):
            v = vals[i, j]
            if holes[i, j]:
                continue

            # Strict local extremum check (4 cardinal neighbors)
            jp1 = (j + 1) % n_cols
            jm1 = (j - 1 + n_cols) % n_cols
            if is_min:
                if not (
                    v < vals[i - 1, j]
                    and v < vals[i + 1, j]
                    and v < vals[i, jp1]
                    and v < vals[i, jm1]
                ):
                    continue
                if v >= best_val:
                    continue
            else:
                if not (
                    v > vals[i - 1, j]
                    and v > vals[i + 1, j]
                    and v > vals[i, jp1]
                    and v > vals[i, jm1]
                ):
                    continue
                if v <= best_val:
                    continue

            # CheckPoint: (2*N+1)x(2*N+1) neighborhood must be hole-free
            ok = True
            i_lo = max(0, i - neighbor_check)
            i_hi = min(i + neighbor_check, n_rows)
            for ki in range(i_lo, i_hi):
                for kj_off in range(-neighbor_check, neighbor_check):
                    cj = (j + kj_off + n_cols) % n_cols
                    if holes[ki, cj]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            best_val = v
            best_row = i
            best_col = j

    # If we found a valid local extremum (not the center fallback), return it.
    if best_row != 0 or best_col != 0:
        r_mm = best_row * _R_STEP_MM
        x_mm = float(r_mm * _COS_THETA[best_col])
        y_mm = float(r_mm * _SIN_THETA[best_col])
        return float(best_val), x_mm, y_mm

    # -------------------------------------------------------------------
    # Fallback for severe KC: The strict CSO algorithm (local extremum +
    # 5x5 hole-free check) can fail when the thinnest / steepest point is
    # surrounded by missing data (common in severe KC where OCT loses the
    # posterior surface near the cone tip).
    #
    # Strategy: relax the hole check to N=1 (3x3 neighborhood), still
    # requiring a local extremum.  If that also fails, use a global
    # argmin/argmax within a safe radius (< 4.0 mm) to avoid peripheral
    # artifacts.
    # -------------------------------------------------------------------

    # Pass 2: relaxed hole check (N=1, 3x3 neighborhood)
    relaxed_neighbor = 1
    for i in range(1, n_rows - 1):
        if max_radius is not None and (i + 1) * _R_STEP_MM >= max_radius:
            continue
        if i * _R_STEP_MM > checkpoint_radius:
            continue

        for j in range(n_cols):
            v = vals[i, j]
            if holes[i, j]:
                continue

            jp1 = (j + 1) % n_cols
            jm1 = (j - 1 + n_cols) % n_cols
            if is_min:
                if not (
                    v < vals[i - 1, j]
                    and v < vals[i + 1, j]
                    and v < vals[i, jp1]
                    and v < vals[i, jm1]
                ):
                    continue
                if v >= best_val:
                    continue
            else:
                if not (
                    v > vals[i - 1, j]
                    and v > vals[i + 1, j]
                    and v > vals[i, jp1]
                    and v > vals[i, jm1]
                ):
                    continue
                if v <= best_val:
                    continue

            ok = True
            i_lo = max(0, i - relaxed_neighbor)
            i_hi = min(i + relaxed_neighbor, n_rows)
            for ki in range(i_lo, i_hi):
                for kj_off in range(-relaxed_neighbor, relaxed_neighbor):
                    cj = (j + kj_off + n_cols) % n_cols
                    if holes[ki, cj]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            best_val = v
            best_row = i
            best_col = j

    if best_row != 0 or best_col != 0:
        r_mm = best_row * _R_STEP_MM
        x_mm = float(r_mm * _COS_THETA[best_col])
        y_mm = float(r_mm * _SIN_THETA[best_col])
        return float(best_val), x_mm, y_mm

    # Pass 3: global search within safe radius (no local-extremum requirement)
    # Only used as last resort to avoid returning center value for pathological
    # eyes.  Limited to radius < 4.0 mm to avoid peripheral artifacts.
    safe_radius = min(checkpoint_radius, 4.0)
    for i in range(1, n_rows):
        if i * _R_STEP_MM > safe_radius:
            break
        for j in range(n_cols):
            v = vals[i, j]
            if holes[i, j]:
                continue
            if is_min and v < best_val:
                best_val = v
                best_row = i
                best_col = j
            elif not is_min and v > best_val:
                best_val = v
                best_row = i
                best_col = j

    if best_row == 0 and best_col == 0:
        if holes[0, 0]:
            return None, 0.0, 0.0
        return float(best_val), 0.0, 0.0

    r_mm = best_row * _R_STEP_MM
    x_mm = float(r_mm * _COS_THETA[best_col])
    y_mm = float(r_mm * _SIN_THETA[best_col])
    return float(best_val), x_mm, y_mm


def _corneal_volume(polar_map, elev_ant_map=None):
    """
    Compute corneal volume by integrating the thickness map in polar coords.

    CSO's actual algorithm
    integrates the *difference* between posterior and anterior Z-surface
    elevation maps using a cell-by-cell 4-corner average over each annular
    cell.  The CSV exports perpendicular thickness, NOT the axial
    Z-difference that CSO uses.  On a curved cornea, the axial distance
    between anterior and posterior surfaces is always >= the perpendicular
    thickness, with the ratio growing as 1/cos(alpha) where alpha is the
    anterior surface inclination angle.

    When ``elev_ant_map`` is provided, a per-ring geometric correction is
    applied: the anterior surface slope is estimated from adjacent rings,
    and the thickness integral is multiplied by 1/cos(alpha) to convert
    from perpendicular to axial thickness.  This matches CSO's result to
    within ~1-2%.

    Uses a cell-by-cell approach that mirrors CSO's formula structure:
    for each annular cell, average the 4 corner thickness values and
    multiply by the cell area.  Cells with any NaN corner are skipped,
    but the missing angular fraction is compensated by scaling up each
    ring's contribution proportionally.

    Hard cutoff at r = 5.0 mm (diameter 10 mm), matching CSO.

    Parameters
    ----------
    polar_map : np.ndarray, shape (n_rows, 256)
        CornealThickness map in micrometers, with -1000 for missing.
    elev_ant_map : np.ndarray or None, shape (n_rows, 256)
        ElevationAnterior map in mm, with -1000 for missing.  Used to
        compute the per-ring surface slope correction.  If *None*, the
        raw perpendicular thickness is integrated without correction.

    Returns
    -------
    float
        Corneal volume in mm^3.
    """
    clean = polar_map.astype(np.float64)
    clean[clean == _MISSING] = np.nan
    thickness_mm = clean / 1000.0  # um -> mm

    n_rows = thickness_mm.shape[0]
    max_radius = 5.0  # CSO hard cutoff (diameter 10 mm)

    r_mm = np.arange(n_rows) * _R_STEP_MM

    # ------------------------------------------------------------------
    # Compute per-ring slope correction from the anterior elevation map.
    # CSO integrates the axial Z-difference (posterior - anterior), which
    # equals thickness / cos(alpha) where alpha is the surface angle.
    # slope = dz/dr, cos(alpha) = 1 / sqrt(1 + slope^2)
    # correction = 1 / cos(alpha) = sqrt(1 + slope^2)
    # ------------------------------------------------------------------
    ring_correction = np.ones(n_rows, dtype=np.float64)
    if elev_ant_map is not None:
        elev = elev_ant_map.astype(np.float64)
        elev[elev == _MISSING] = np.nan
        for i in range(1, n_rows - 1):
            valid = ~np.isnan(elev[i]) & ~np.isnan(elev[i + 1])
            if valid.sum() >= 10:
                slopes = (elev[i + 1, valid] - elev[i, valid]) / _R_STEP_MM
                mean_slope = np.mean(np.abs(slopes))
                ring_correction[i] = np.sqrt(1.0 + mean_slope**2)
        # Last usable ring: extrapolate from previous two if possible
        if n_rows >= 3 and ring_correction[n_rows - 2] > 1.0:
            # Use slope between rows (n_rows-3) and (n_rows-2) as fallback
            valid = ~np.isnan(elev[n_rows - 3]) & ~np.isnan(elev[n_rows - 2])
            if valid.sum() >= 10:
                slopes = (elev[n_rows - 2, valid] - elev[n_rows - 3, valid]) / _R_STEP_MM
                mean_slope = np.mean(np.abs(slopes))
                ring_correction[n_rows - 1] = np.sqrt(1.0 + mean_slope**2)

    # Cell-by-cell integration matching CSO's formula structure:
    #   num2 = (r[i+1]^2 - r[i]^2) * 0.5 * 0.25
    #   sum 4-corner values per cell
    #   final *= 2*pi/n_cols
    # For partial-coverage rings, scale by (n_cols / n_valid_cells)
    # to estimate the full-ring volume.
    volume = 0.0
    for i in range(n_rows - 1):
        if r_mm[i + 1] > max_radius:
            break
        cell_area_factor = (r_mm[i + 1] ** 2 - r_mm[i] ** 2) * 0.5 * 0.25
        ring_sum = 0.0
        n_valid_cells = 0
        for j in range(_N_COLS):
            j1 = (j + 1) % _N_COLS
            v00 = thickness_mm[i, j]
            v10 = thickness_mm[i + 1, j]
            v01 = thickness_mm[i, j1]
            v11 = thickness_mm[i + 1, j1]
            if np.isnan(v00) or np.isnan(v10) or np.isnan(v01) or np.isnan(v11):
                continue
            ring_sum += v00 + v10 + v01 + v11
            n_valid_cells += 1
        if n_valid_cells > 0:
            # Scale up to full ring to compensate for missing sectors
            scale = _N_COLS / n_valid_cells
            # Apply geometric correction (perpendicular -> axial thickness)
            corr = 0.5 * (ring_correction[i] + ring_correction[i + 1])
            volume += cell_area_factor * ring_sum * scale * corr

    volume *= 2.0 * np.pi / _N_COLS

    return volume


def compute_summary_indices(raw_segments: dict) -> dict:
    """
    Recompute the 16 topographic-derived summary indices from polar maps.

    These are the same indices the MS-39 puts in the ms39_global CSV
    under "Summary Indices". The 19 OCT-derived indices (ACD, TISA, AOD)
    cannot be reconstructed from the individual CSV and are not included.

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of core.parse_csv(). Keys are segment names, values are
        polar matrices with shape (n_rows, 256).

    Returns
    -------
    dict
        16 named features with keys matching ms39_global column conventions.
    """
    out = {}

    # 1. ThkMin — minimum corneal thickness
    thk_map = raw_segments.get("corneal_thickness")
    if thk_map is not None:
        val, x, y = _find_extremum(thk_map, mode="min")
        out["thk_min_value"] = val
        out["thk_min_x"] = x
        out["thk_min_y"] = y
    else:
        out["thk_min_value"] = None
        out["thk_min_x"] = None
        out["thk_min_y"] = None

    # 2. KMaxF — steepest anterior curvature (minimum radius in mm)
    #    CSO uses the *Gaussian* curvature map (ArcStep.GetGaussianCurvature),
    #    NOT the sagittal curvature map.  Decompiled SummaryIndicesObj line 260:
    #    m_KMaxF = Utils.GetMin(radii, GaussianCurvatureAnterior, maskAnterior)
    gauss_ant = raw_segments.get("gaussian_anterior")
    if gauss_ant is not None:
        val, x, y = _find_extremum(gauss_ant, mode="min")
        out["kmax_front_value"] = val
        out["kmax_front_x"] = x
        out["kmax_front_y"] = y
    else:
        out["kmax_front_value"] = None
        out["kmax_front_x"] = None
        out["kmax_front_y"] = None

    # 3. KMaxB — steepest posterior curvature (minimum radius in mm)
    #    CSO uses Gaussian curvature posterior: GetMin(radii, gaussPost, mask)
    #    with NO explicit Limit parameter (defaults to float.MaxValue).
    #    CheckPoint uses MaxR=4.0 (hardcoded) which limits rows to r <= 4.0mm.
    gauss_post = raw_segments.get("gaussian_posterior")
    if gauss_post is not None:
        val, x, y = _find_extremum(gauss_post, mode="min")
        out["kmax_back_value"] = val
        out["kmax_back_x"] = x
        out["kmax_back_y"] = y
    else:
        out["kmax_back_value"] = None
        out["kmax_back_x"] = None
        out["kmax_back_y"] = None

    # 4. EpiThkMin — minimum epithelial thickness
    #    CSO applies a 3.9mm radius cutoff:
    #    m_EpiThkMin = Utils.GetMin(radii, epiMap, epiMask, 3.9f)
    #    CSO also uses an internal OCT validity mask (bool[,] j()) that filters
    #    unreliable peripheral epithelial segmentations.  This mask is NOT
    #    available in the CSV export, so our result may differ for exams with
    #    peripheral epithelial artifacts.
    epi_map = raw_segments.get("epithelial_thickness")
    if epi_map is not None:
        val, x, y = _find_extremum(epi_map, mode="min", max_radius=3.9)
        out["epi_thk_min_value"] = val
        out["epi_thk_min_x"] = x
        out["epi_thk_min_y"] = y
    else:
        out["epi_thk_min_value"] = None
        out["epi_thk_min_x"] = None
        out["epi_thk_min_y"] = None

    # 5. EpiThkMax — maximum epithelial thickness
    #    Same 3.9mm radius cutoff as EpiThkMin:
    #    m_EpiThkMax = Utils.GetMax(radii, epiMap, epiMask, 3.9f)
    if epi_map is not None:
        val, x, y = _find_extremum(epi_map, mode="max", max_radius=3.9)
        out["epi_thk_max_value"] = val
        out["epi_thk_max_x"] = x
        out["epi_thk_max_y"] = y
    else:
        out["epi_thk_max_value"] = None
        out["epi_thk_max_x"] = None
        out["epi_thk_max_y"] = None

    # 6. CornealVolume — integration of thickness over polar area
    #    CSO integrates the axial Z-difference (posterior - anterior elevation),
    #    not the perpendicular thickness.  When the anterior elevation map is
    #    available, compute a per-ring slope correction to convert from
    #    perpendicular to axial thickness.
    elev_ant = raw_segments.get("elevation_anterior")
    if thk_map is not None:
        out["corneal_volume"] = _corneal_volume(thk_map, elev_ant_map=elev_ant)
    else:
        out["corneal_volume"] = None

    return out


# ---------------------------------------------------------------------------
# ABCD Staging (Belin/Ambrosio)
# ---------------------------------------------------------------------------


def _trimmed_mean_in_disc(polar_map, center_x, center_y, radius=1.5, alpha=0.05, min_count=100):
    """
    Compute a 5%-trimmed mean of *polar_map* values within a disc of the
    given radius (mm) centered at (*center_x*, *center_y*) in Cartesian
    coordinates.

    Exactly mirrors CSO ``Utils.GetCurvAvg`` (see formulas_classification.md
    Section 1.4).

    Parameters
    ----------
    polar_map : np.ndarray, shape (n_rows, 256)
    center_x, center_y : float
        Disc centre in mm (from corneal vertex).
    radius : float
        Disc radius in mm (default 1.5 -> 3 mm diameter zone).
    alpha : float
        Fraction to trim from each tail (default 0.05).
    min_count : int
        Minimum number of valid samples after trimming.  If fewer, return NaN.

    Returns
    -------
    float
        Trimmed mean, or NaN if insufficient data.
    """
    n_rows, n_cols = polar_map.shape
    samples = []
    for j in range(n_cols):
        cos_t = _COS_THETA[j]
        sin_t = _SIN_THETA[j]
        for i in range(n_rows):
            val = polar_map[i, j]
            if val == _MISSING or np.isnan(val):
                continue
            r_mm = i * _R_STEP_MM
            x = r_mm * cos_t
            y = r_mm * sin_t
            if (x - center_x) ** 2 + (y - center_y) ** 2 < radius**2:
                samples.append(val)

    if len(samples) == 0:
        return float("nan")

    samples.sort()
    lo = int(len(samples) * alpha)
    hi = int(len(samples) * (1.0 - alpha))
    trimmed = samples[lo:hi]

    if len(trimmed) < min_count:
        return float("nan")

    return sum(trimmed) / len(trimmed)


def _abcd_grade_a(value):
    """Stage the A parameter (anterior ARC in mm). Lower = worse."""
    if np.isnan(value):
        return "*"
    if value > 7.25:
        return "0"
    if value > 7.05:
        return "1"
    if value > 6.35:
        return "2"
    if value > 6.15:
        return "3"
    return "4"


def _abcd_grade_b(value):
    """Stage the B parameter (posterior PRC in mm). Lower = worse."""
    if np.isnan(value):
        return "*"
    if value > 5.90:
        return "0"
    if value > 5.70:
        return "1"
    if value > 5.15:
        return "2"
    if value > 4.95:
        return "3"
    return "4"


def _abcd_grade_c(value):
    """Stage the C parameter (ThkMin in um). Lower = worse."""
    if np.isnan(value):
        return "*"
    if value > 490:
        return "0"
    if value > 450:
        return "1"
    if value > 400:
        return "2"
    if value > 300:
        return "3"
    return "4"


def compute_abcd_staging(raw_segments: dict) -> dict:
    """
    Compute ABCD keratoconus staging parameters (Belin/Ambrosio system).

    Algorithm (from CSO KeratoconusABCDStaging, see formulas_classification.md):

    1. Locate the thinnest corneal point (ThkMin) on the pachymetry map.
    2. A = 5%-trimmed mean of anterior sagittal curvature within a 1.5 mm
       radius disc centered on ThkMin.
    3. B = same computation on the posterior sagittal curvature map.
    4. C = ThkMin value (minimum corneal thickness in um).
    5. D = BCVA from Rx record -- not available in the individual CSV,
       returned as None.

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of ``core.parse_csv()``.  Required keys:
        ``corneal_thickness``, ``sagittal_anterior``, ``sagittal_posterior``.

    Returns
    -------
    dict
        Keys: ``abcd_a``, ``abcd_b``, ``abcd_c``, ``abcd_d``,
        ``abcd_a_grade``, ``abcd_b_grade``, ``abcd_c_grade``, ``abcd_d_grade``,
        ``abcd_string``.
    """
    out = {}
    thk_map = raw_segments.get("corneal_thickness")
    sag_ant = raw_segments.get("sagittal_anterior")
    sag_post = raw_segments.get("sagittal_posterior")

    # --- C: ThkMin and its location ------------------------------------------
    if thk_map is not None:
        thk_val, thk_x, thk_y = _find_extremum(thk_map, mode="min")
    else:
        thk_val, thk_x, thk_y = None, 0.0, 0.0

    out["abcd_c"] = thk_val
    out["abcd_c_grade"] = _abcd_grade_c(thk_val) if thk_val is not None else "*"

    # --- A: Anterior ARC in 3 mm disc around ThkMin -------------------------
    if sag_ant is not None and thk_val is not None:
        a_val = _trimmed_mean_in_disc(
            sag_ant.astype(np.float64),
            thk_x,
            thk_y,
        )
        out["abcd_a"] = None if np.isnan(a_val) else float(a_val)
    else:
        a_val = float("nan")
        out["abcd_a"] = None

    out["abcd_a_grade"] = _abcd_grade_a(a_val)

    # --- B: Posterior PRC in 3 mm disc around ThkMin -------------------------
    if sag_post is not None and thk_val is not None:
        b_val = _trimmed_mean_in_disc(
            sag_post.astype(np.float64),
            thk_x,
            thk_y,
        )
        out["abcd_b"] = None if np.isnan(b_val) else float(b_val)
    else:
        b_val = float("nan")
        out["abcd_b"] = None

    out["abcd_b_grade"] = _abcd_grade_b(b_val)

    # --- D: BCVA (not available in individual CSV) ---------------------------
    out["abcd_d"] = None
    out["abcd_d_grade"] = "*"

    # --- Composite string (e.g. "A2 B1 C0 D*") ------------------------------
    out["abcd_string"] = (
        f"A{out['abcd_a_grade']} "
        f"B{out['abcd_b_grade']} "
        f"C{out['abcd_c_grade']} "
        f"D{out['abcd_d_grade']}"
    )

    return out


# ---------------------------------------------------------------------------
# Epithelial Sectoral Statistics
# ---------------------------------------------------------------------------

# Column ranges for each sector (OD convention, half-open [start, end)).
# For OS, nasal and temporal swap.
_SECTOR_COLS_OD = {
    "superior": list(range(32, 96)),  # 45-135 deg
    "nasal": list(range(224, 256)) + list(range(0, 32)),  # 315-45 deg
    "temporal": list(range(96, 160)),  # 135-225 deg
    "inferior": list(range(160, 224)),  # 225-315 deg
}

_SECTOR_COLS_OS = {
    "superior": _SECTOR_COLS_OD["superior"],
    "nasal": _SECTOR_COLS_OD["temporal"],  # swapped
    "temporal": _SECTOR_COLS_OD["nasal"],  # swapped
    "inferior": _SECTOR_COLS_OD["inferior"],
}

# Zone definitions: (inner_radius_mm, outer_radius_mm, label)
_EPI_ZONES = [
    (0.0, 1.5, "central"),  # 0 - 3 mm diameter
    (1.5, 3.0, "6mm"),  # 3 - 6 mm diameter annulus
    (3.0, 4.0, "8mm"),  # 6 - 8 mm diameter annulus
]


def compute_epithelial_sectors(raw_segments: dict, metadata: dict) -> dict:
    """
    Compute sectoral epithelial thickness statistics (min, max, mean).

    Mirrors the ~27 columns in the MS-39 global CSV export produced by
    ``EpithelialThicknessObj`` (see formulas_epithelial.md Section 3).

    Zones (concentric annuli defined by radius from corneal vertex):
      * Central : 0 - 1.5 mm  (full circle, no angular sectors)
      * 6 mm    : 1.5 - 3.0 mm  (4 angular sectors)
      * 8 mm    : 3.0 - 4.0 mm  (4 angular sectors)

    Angular sectors (TABO, 90 deg each):
      Superior 45-135 | Nasal 315-45 (OD) | Temporal 135-225 (OD) | Inferior 225-315
      For OS the nasal/temporal labels swap.

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Must contain ``epithelial_thickness`` key.
    metadata : dict
        Must contain ``Exam_Eye`` with value ``"OD"`` or ``"OS"``.

    Returns
    -------
    dict
        27 keys following the pattern:
        ``epi_{stat}_{zone}[_{sector}]`` where stat is mean/min/max.
        Returns None values if the epithelial map is missing.
    """
    out = {}
    epi_map = raw_segments.get("epithelial_thickness")

    if epi_map is None:
        # Return all 27 keys as None
        for _, _, zone_label in _EPI_ZONES:
            if zone_label == "central":
                for stat in ("mean", "max", "min"):
                    out[f"epi_{stat}_central"] = None
            else:
                for sector in ("superior", "nasal", "temporal", "inferior"):
                    for stat in ("mean", "max", "min"):
                        out[f"epi_{stat}_{zone_label}_{sector}"] = None
        return out

    clean = epi_map.astype(np.float64)
    clean[clean == _MISSING] = np.nan
    n_rows = clean.shape[0]

    # Pre-compute radial distances for each row
    r_mm = np.arange(n_rows) * _R_STEP_MM

    # Choose sector column mapping based on laterality
    eye = metadata.get("Exam_Eye", "OD")
    sector_cols = _SECTOR_COLS_OD if eye == "OD" else _SECTOR_COLS_OS

    for r_inner, r_outer, zone_label in _EPI_ZONES:
        # Row mask: rows whose radial distance falls within [r_inner, r_outer)
        # Use <= for outer boundary to match CSO's <= comparison
        row_mask = (r_mm >= r_inner) & (r_mm <= r_outer)
        # Exclude r_inner boundary for annuli (CSO uses strict >)
        if r_inner > 0:
            row_mask = (r_mm > r_inner) & (r_mm <= r_outer)

        zone_rows = np.where(row_mask)[0]
        if len(zone_rows) == 0:
            if zone_label == "central":
                for stat in ("mean", "max", "min"):
                    out[f"epi_{stat}_central"] = None
            else:
                for sector in ("superior", "nasal", "temporal", "inferior"):
                    for stat in ("mean", "max", "min"):
                        out[f"epi_{stat}_{zone_label}_{sector}"] = None
            continue

        zone_data = clean[zone_rows, :]  # shape (n_zone_rows, 256)

        if zone_label == "central":
            # Full circle -- no angular sectors
            vals = zone_data.ravel()
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                out["epi_mean_central"] = None
                out["epi_max_central"] = None
                out["epi_min_central"] = None
            else:
                out["epi_mean_central"] = float(np.mean(valid))
                out["epi_max_central"] = float(np.max(valid))
                out["epi_min_central"] = float(np.min(valid))
        else:
            # 4 angular sectors
            for sector_name, col_indices in sector_cols.items():
                vals = zone_data[:, col_indices].ravel()
                valid = vals[~np.isnan(vals)]
                if len(valid) == 0:
                    out[f"epi_mean_{zone_label}_{sector_name}"] = None
                    out[f"epi_max_{zone_label}_{sector_name}"] = None
                    out[f"epi_min_{zone_label}_{sector_name}"] = None
                else:
                    out[f"epi_mean_{zone_label}_{sector_name}"] = float(np.mean(valid))
                    out[f"epi_max_{zone_label}_{sector_name}"] = float(np.max(valid))
                    out[f"epi_min_{zone_label}_{sector_name}"] = float(np.min(valid))

    return out


# ==========================================================================
# EPITHELIAL REFRACTION (EpiRX) AND EpiRMSonA  (28 columns)
# ==========================================================================
#
# Treats the epithelial thickness map as a refractive "surface" and derives
# the optical contribution of the epithelium via Zernike decomposition at
# two apertures (3mm and 6mm diameter).
#
# Pipeline (matches CSO EpithelialThicknessObj):
#   1. Convert epithelial thickness polar map to mm, extract Cartesian coords
#   2. Fit Zernike directly to the thickness map (no BFS subtraction --
#      the epithelial map is a flat thickness field, not a curved surface)
#   3. Convert geometric Zernike coefficients to OPD via delta_n = -0.401
#   4. Derive refraction (M, J0, J45, Sph, Cyl, CylAx) from low-order OPD
#      using NEGATIVE multipliers (Thibos 2002, matches CSO)
#   5. Apply vertex distance correction (12.5mm, cornea->spectacle)
#   6. Compute EpiRMSonA = HOA_RMS / (pi * R^2)
#
# Reference: formulas_epithelial.md Sections 4-5,
#            formulas_cornealwf_refraction.md Section 6 (ObjRx.FromZernike).
#
# NOTE: CSO applies -0.401 * thickness BEFORE Zernike fitting; we apply
# delta_n AFTER fitting (mathematically equivalent since Zernike fitting
# is linear).  Uses NEGATIVE multipliers (Thibos 2002, matches CSO).
# Remaining error (~5-13% on refraction) is due to pupil-center offset:
# CSO subtracts pupil center from polar coords before fitting.
# ==========================================================================

# Epithelial refractive indices
_EPI_DELTA_N = -0.401  # -(n_epithelium - n_air) = -(1.401 - 1.0), matches CSO
_EPI_VD_MM = 12.5  # standard vertex distance (mm), matches CSO Settings.Default.RxVD
_EPI_VD_M = _EPI_VD_MM / 1000.0  # vertex distance in meters

# Apertures for EpiRX: (diameter_mm, fitting_radius_mm, label)
_EPI_RX_APERTURES = [
    (3.0, 1.5, "3mm"),
    (6.0, 3.0, "6mm"),
]


def _epirx_from_zernike(opd_coeffs_um, fitting_radius_mm):
    """
    Derive clinical refraction from OPD Zernike coefficients.

    Implements CSO's ObjRx.FromZernike power-vector decomposition plus
    vertex distance correction.  Uses NEGATIVE multipliers
    (Thibos 2002, matches CSO).

    Parameters
    ----------
    opd_coeffs_um : np.ndarray
        Zernike coefficients in micrometers (OPD domain).
        Must have at least 6 elements (indices 0-5).
    fitting_radius_mm : float
        Fitting radius in mm.

    Returns
    -------
    dict
        Keys: m, j0, j45, sph_eq, sph_eq_vd, sph, sph_vd,
        cyl, cyl_vd, cyl_ax, vd, pupil_radius, pupil_diam.
    """
    r = fitting_radius_mm
    r2 = r * r

    c3 = opd_coeffs_um[3]  # Z(2,-2) oblique astigmatism
    c4 = opd_coeffs_um[4]  # Z(2, 0) defocus
    c5 = opd_coeffs_um[5]  # Z(2,+2) vertical astigmatism

    # Power vector -- POSITIVE multipliers for epithelial refraction.
    # The delta_n (-0.401) already flips the sign of the OPD coefficients,
    # so positive multipliers here give the correct clinical refraction.
    # (NB: the OPD WF refraction in CornealWF uses negative multipliers
    # because ray-traced OPD has no delta_n flip.)
    M = 4.0 * math.sqrt(3.0) * c4 / r2
    J0 = 2.0 * math.sqrt(6.0) * c5 / r2
    J45 = 2.0 * math.sqrt(6.0) * c3 / r2

    # Clinical refraction (negative cylinder convention)
    Cyl = -2.0 * math.sqrt(J0**2 + J45**2)

    # Cylinder axis in [0, 180)
    # CSO: (atan2(J45, J0) / pi * 180 + 360) / 2 % 180
    CylAx = 0.5 * math.atan2(J45, J0) * (180.0 / math.pi)
    if CylAx < 0.0:
        CylAx += 180.0
    if CylAx >= 180.0:
        CylAx -= 180.0

    Sph = M - Cyl / 2.0
    SphEq = M

    # Vertex distance correction -- corneal plane to spectacle plane.
    # CSO epithelial refraction uses P_vd = P / (1 + P * VD_m).
    # This is the vergence formula for the epithelial case where the
    # sign convention already accounts for delta_n direction.
    d = _EPI_VD_M
    denom_sph = 1.0 + Sph * d
    denom_sph_cyl = 1.0 + (Sph + Cyl) * d

    if abs(denom_sph) < 1e-12 or abs(denom_sph_cyl) < 1e-12:
        SphVD = Sph
        CylVD = Cyl
    else:
        SphVD = Sph / denom_sph
        CylVD = (Sph + Cyl) / denom_sph_cyl - SphVD

    # CSO computes SphEqVD directly as M / (1 + M * VD)
    denom_m = 1.0 + M * d
    if abs(denom_m) < 1e-12:
        SphEqVD = M
    else:
        SphEqVD = M / denom_m

    return {
        "pupil_radius": r,
        "pupil_diam": 2.0 * r,
        "m": M,
        "j0": J0,
        "j45": J45,
        "sph_eq": SphEq,
        "sph_eq_vd": SphEqVD,
        "sph": Sph,
        "sph_vd": SphVD,
        "cyl": Cyl,
        "cyl_vd": CylVD,
        "cyl_ax": CylAx,
        "vd": _EPI_VD_MM,
    }


def compute_epithelial_refraction(raw_segments, metadata):
    """
    Compute epithelial refraction (EpiRX) and EpiRMSonA indices.

    Fits Zernike polynomials to the epithelial thickness map at 3mm and
    6mm diameter apertures, converts to OPD via delta_n, then derives
    clinical refraction (M, J0, J45, Sph, Cyl, etc.) and higher-order
    RMS.

    Mirrors the 28 columns in the MS-39 global CSV:
      - 13 EpiRX columns per aperture (x2 = 26)
      - 2 EpiRMSonA columns (one per aperture)

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Must contain ``epithelial_thickness`` key with shape (n_rows, 256).
        Values are epithelial thickness in micrometers, -1000 for missing.
    metadata : dict
        Must contain ``Exam_Eye`` (not used for EpiRX but kept for
        interface consistency with other compute functions).

    Returns
    -------
    dict
        28 keys:
          epirx_{D}mm_pupil_radius, epirx_{D}mm_pupil_diam,
          epirx_{D}mm_m, epirx_{D}mm_j0, epirx_{D}mm_j45,
          epirx_{D}mm_sph_eq, epirx_{D}mm_sph_eq_vd,
          epirx_{D}mm_sph, epirx_{D}mm_sph_vd,
          epirx_{D}mm_cyl, epirx_{D}mm_cyl_vd, epirx_{D}mm_cyl_ax,
          epirx_{D}mm_vd,
          epi_rmson_a_{D}mm
        for D in {3, 6}. All values are None if the epithelial map is
        missing or the Zernike fit fails.
    """
    out = {}
    epi_map = raw_segments.get("epithelial_thickness")

    # Define all output keys (used for None-fill on failure)
    all_keys = []
    for _, _, label in _EPI_RX_APERTURES:
        for suffix in (
            "pupil_radius",
            "pupil_diam",
            "m",
            "j0",
            "j45",
            "sph_eq",
            "sph_eq_vd",
            "sph",
            "sph_vd",
            "cyl",
            "cyl_vd",
            "cyl_ax",
            "vd",
        ):
            all_keys.append(f"epirx_{label}_{suffix}")
        all_keys.append(f"epi_rmson_a_{label}")

    if epi_map is None:
        for key in all_keys:
            out[key] = None
        return out

    # Convert epithelial thickness from micrometers to mm for Zernike fitting.
    # Keep the sentinel value intact so _polar_elevation_to_cartesian can
    # filter it out (it checks for _MISSING = -1000).
    epi_mm = epi_map.astype(np.float64)
    non_sentinel = epi_mm != _MISSING
    epi_mm[non_sentinel] = epi_mm[non_sentinel] / 1000.0

    # Pupil center offset (mm, in map coordinates).
    # CSO EpithelialThicknessObj.cs:471-472 subtracts PupilCenter before
    # Zernike fitting AND clips by pupil-centered radius, not vertex radius.
    pupil_cx = float(metadata.get("PupilCX", 0.0) or 0.0)
    pupil_cy = float(metadata.get("PupilCY", 0.0) or 0.0)

    for diam_mm, fit_radius, label in _EPI_RX_APERTURES:
        prefix_rx = f"epirx_{label}"
        prefix_rms = f"epi_rmson_a_{label}"

        # Step 1: Extract Cartesian coordinates from the epithelial map.
        # Use a generous extraction radius so points near the edge of the
        # pupil-centered circle are not excluded prematurely.
        extract_radius = fit_radius + max(abs(pupil_cx), abs(pupil_cy)) + 0.5
        x_raw, y_raw, z_raw = _polar_elevation_to_cartesian(epi_mm, extract_radius)

        # Shift to pupil-centered coordinates (CSO subtracts PupilCenter)
        x_pc = x_raw - pupil_cx
        y_pc = y_raw - pupil_cy

        # Filter by pupil-centered radius (matching CSO aperture clipping)
        r2_pc = x_pc**2 + y_pc**2
        mask = r2_pc <= fit_radius**2
        x = x_pc[mask]
        y = y_pc[mask]
        z = z_raw[mask]

        # Step 2: Fit Zernike polynomials directly to the thickness values.
        # No BFS subtraction -- the epithelial map is a flat thickness
        # field (not a curved surface), so BFS is inappropriate.
        # CSO EpithelialThicknessObj uses MMSE.ZernikeFitting with default
        # Npol=36 (order 7).  Using order 8 (45 terms) causes low-order
        # coefficient leakage at the small 3mm aperture.
        coeffs_um, _ = _fit_zernike_coefficients(x, y, z, fit_radius, max_order=7)

        if coeffs_um is None or len(coeffs_um) < 6:
            # Fitting failed -- fill all keys for this aperture with None
            for suffix in (
                "pupil_radius",
                "pupil_diam",
                "m",
                "j0",
                "j45",
                "sph_eq",
                "sph_eq_vd",
                "sph",
                "sph_vd",
                "cyl",
                "cyl_vd",
                "cyl_ax",
                "vd",
            ):
                out[f"{prefix_rx}_{suffix}"] = None
            out[prefix_rms] = None
            continue

        # Step 3: Convert geometric Zernike coefficients to OPD
        # OPD_coefs = geometric_coefs * delta_n
        opd_coeffs_um = coeffs_um * _EPI_DELTA_N

        # Step 4: Derive refraction from OPD low-order coefficients
        rx = _epirx_from_zernike(opd_coeffs_um, fit_radius)

        for suffix, value in rx.items():
            out[f"{prefix_rx}_{suffix}"] = float(value)

        # Step 5: EpiRMSonA -- higher-order RMS of OPD coefficients
        # CSO divides by pi*R^2 (fitting circle area):
        #   EpiRMSonA = GetRMS(HighOrders) / (pi * R^2)
        hoa_rms = _compute_hoa_rms(opd_coeffs_um)
        if hoa_rms is not None:
            hoa_rms = hoa_rms / (math.pi * fit_radius * fit_radius)
        out[prefix_rms] = hoa_rms

    return out


# ==========================================================================
# SHAPE INDICES  (108 columns: 2 surfaces x 9 diameters x 6 values)
# ==========================================================================
#
# Recomputes the 108 shape-index columns that the MS-39 places in the
# global CSV export under "Shape Indices".  Each column describes a
# conic-section fit of the corneal surface at a given analysis diameter.
#
# Algorithm (matches CSO -- see formulas_shape_indices.md):
#
#   1.  Convert the sagittal-radius polar map R_sag(h, theta) to surface
#       heights z using z = R_sag - sqrt(R_sag^2 - h^2).  This is exact
#       for the sagittal curvature definition.
#
#   2.  For each nominal diameter D, select valid (h, z, theta) triples
#       where h <= D/2 and neither R_sag nor z is invalid.
#
#   3.  Fit a 4-parameter toroidal conic:
#           h^2 = -p*z^2 + 2*R_mean*z + 2*dR*cos(2*alpha)*z*cos(2*theta)
#                                       + 2*dR*sin(2*alpha)*z*sin(2*theta)
#       This is linear in the unknowns [p, 2*R_mean, 2*dR*cos(2a), 2*dR*sin(2a)]
#       and solved via a 4x4 symmetric normal-equation system.
#
#   4.  Extract:
#         p       = x[0]
#         R_mean  = x[1] / 2
#         dR      = sqrt(x[2]^2 + x[3]^2) / 2
#         alpha   = atan2(x[3], x[2]) / 2      (physical flat-axis angle)
#         Rflat   = |R_mean| + |dR|
#         Rsteep  = |R_mean| - |dR|
#         RflatAx = alpha in degrees, folded to [0, 180)
#
#   5.  Compute RMS of fit residuals (torus model) in microns.

# The 9 fitting diameters used by the MS-39
_SHAPE_DIAMETERS = [3, 4, 4.5, 5, 6, 7, 8, 9, 10]

# Minimum number of valid cells required for a meaningful conic fit.
# Too few points cause singular / ill-conditioned normal equations.
_MIN_POINTS_FOR_FIT = 10


def _axial_to_height_grid(clean, h_grid, invalid):
    """
    Convert axial curvature radius map to surface height via cumulative
    trapezoidal integration per meridian.

    The MS-39 CSV stores *axial* (ray-optics) radius R_axial = h/sin(phi),
    not the elevation-based sagittal radius R_sag = (h^2+z^2)/(2z).
    For aspherical surfaces these differ, and using the sagittal inversion
    biases asphericity by ~13-80%.

    The correct height reconstruction integrates the slope along each
    meridian (column = fixed angle):

        dz/dh = h / sqrt(R_axial(h)^2 - h^2)
        z(h)  = integral from 0 to h of [rho / sqrt(R(rho)^2 - rho^2)] drho

    Parameters
    ----------
    clean : np.ndarray, shape (n_rows, n_cols)
        Axial radius of curvature in mm.  Sentinel / invalid values must
        be marked in *invalid*.
    h_grid : np.ndarray, shape (n_rows, n_cols)
        Radial distance from optical axis (row 0 = apex = 0 mm).
    invalid : np.ndarray of bool, same shape
        True where the radius value is missing or non-physical.

    Returns
    -------
    z_grid : np.ndarray, same shape
        Surface height in mm.  NaN where integration could not proceed
        (gap in data or h >= R).
    """
    n_rows, n_cols = clean.shape
    z_grid = np.full_like(clean, np.nan)

    for j in range(n_cols):
        col_radii = clean[:, j]
        col_h = h_grid[:, j]
        col_valid = ~invalid[:, j]

        # Need at least row 0 valid (apex)
        if not col_valid[0]:
            continue

        z_col = np.full(n_rows, np.nan)
        z_col[0] = 0.0  # apex height is zero by definition

        for i in range(1, n_rows):
            if not col_valid[i] or np.isnan(z_col[i - 1]):
                break  # stop at first gap

            r_abs = abs(col_radii[i])
            disc = r_abs * r_abs - col_h[i] * col_h[i]
            if disc <= 0.0:
                break

            slope_i = col_h[i] / np.sqrt(disc)

            # Previous point slope
            if col_h[i - 1] == 0.0:
                slope_prev = 0.0  # dz/dh = 0 at apex
            else:
                r_abs_prev = abs(col_radii[i - 1])
                disc_prev = r_abs_prev * r_abs_prev - col_h[i - 1] * col_h[i - 1]
                if disc_prev <= 0.0:
                    break
                slope_prev = col_h[i - 1] / np.sqrt(disc_prev)

            # Trapezoidal rule
            dh = col_h[i] - col_h[i - 1]
            z_col[i] = z_col[i - 1] + 0.5 * (slope_i + slope_prev) * dh

        z_grid[:, j] = z_col

    return z_grid


def _fit_torus_4param(h_valid, z_valid, theta_valid):
    """
    4-parameter toroidal conic fit: exact port of CSO GetBestFitTorus.

    CSO's internal convention uses *negative* sag (z < 0 for a convex
    surface).  Our height conversion produces positive z.  The normal
    equations are therefore solved on ``-z`` so that the extraction
    ``p = x[0]``, ``R = x[1]/2`` yields the correct positive radius and
    physically meaningful asphericity.

    The alpha extraction uses ``atan2(x[3], x[2]) / 2`` because x[2]
    and x[3] encode ``2*dR*cos(2*alpha)`` and ``2*dR*sin(2*alpha)``
    respectively.  CSO uses ``-atan2(x[3], x[2])``
    (without /2) but post-processes via ``CylAx = (360 - 90*alpha/pi)
    % 180`` (formulas_shape_indices.md Section 10.4), which is equivalent to dividing by 2.

    Parameters
    ----------
    h_valid, z_valid, theta_valid : np.ndarray (1-D, same length)
        Radial distances, surface heights (positive), and meridian
        angles of valid data points.

    Returns
    -------
    R_mean : float   -- mean apical radius (mm, positive)
    p      : float   -- asphericity (conic shape factor)
    dR     : float   -- half-difference between flat and steep radii (mm)
    alpha  : float   -- orientation of flattest meridian (radians)

    Raises
    ------
    np.linalg.LinAlgError
        If the 4x4 system is singular (degenerate data).
    """
    # Negate z to match CSO's internal sign convention (see docstring).
    zv = -z_valid

    h2 = h_valid**2
    z2 = zv**2
    z3 = zv * z2
    z4 = z2 * z2

    cos2t = np.cos(2.0 * theta_valid)
    sin2t = np.sin(2.0 * theta_valid)

    # Build 4x4 symmetric normal-equation matrix (upper triangle)
    A = np.zeros((4, 4))
    A[0, 0] = np.sum(z4)
    A[0, 1] = np.sum(z3)
    A[0, 2] = np.sum(z3 * cos2t)
    A[0, 3] = np.sum(z3 * sin2t)
    A[1, 1] = np.sum(z2)
    A[1, 2] = np.sum(z2 * cos2t)
    A[1, 3] = np.sum(z2 * sin2t)
    A[2, 2] = np.sum(z2 * cos2t**2)
    A[2, 3] = np.sum(z2 * sin2t * cos2t)
    A[3, 3] = np.sum(z2 * sin2t**2)

    # Fill symmetric lower triangle
    A[1, 0] = A[0, 1]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    A[3, 0] = A[0, 3]
    A[3, 1] = A[1, 3]
    A[3, 2] = A[2, 3]

    # Right-hand side
    b = np.array(
        [
            -np.sum(h2 * z2),
            -np.sum(h2 * zv),
            -np.sum(h2 * zv * cos2t),
            -np.sum(h2 * zv * sin2t),
        ]
    )

    x = np.linalg.solve(A, b)

    p = x[0]
    R_mean = x[1] / 2.0
    dR = np.sqrt(x[2] ** 2 + x[3] ** 2) / 2.0
    # x[2] = 2*dR*cos(2*alpha), x[3] = 2*dR*sin(2*alpha), so
    # atan2(x[3], x[2]) = 2*alpha.  Divide by 2 to recover alpha.
    alpha = np.arctan2(x[3], x[2]) / 2.0

    return R_mean, p, dR, alpha


def _compute_rms(h2, z_valid, theta_valid, R_mean, p, dR, alpha):
    """
    RMS of fit residuals between measured heights and the torus model.

    z_model = h^2 / ( R(theta) * (1 + sqrt(1 - p*h^2/R(theta)^2)) )

    Returns RMS in *microns* (mm * 1000).
    """
    R_theta = R_mean + dR * np.cos(2.0 * (theta_valid - alpha))
    R_theta2 = R_theta**2

    # Clamp the discriminant to zero for numerical safety (very peripheral
    # or very aspheric points may produce negative values).
    disc = 1.0 - p * h2 / R_theta2
    disc = np.maximum(disc, 0.0)

    z_model = h2 / (R_theta * (1.0 + np.sqrt(disc)))

    residuals = z_valid - z_model
    rms_mm = np.sqrt(np.sum(residuals**2) / max(len(residuals) - 1, 1))
    return rms_mm * 1000.0  # mm -> um


# Gaussian smoothing sigma for BFS residuals (in meridian-sample units).
# CSO applies sigma=2 circular Gaussian smoothing per ring before RMS
# (chapter_best_fit_surfaces.md Sections 10 and 11.2).
_SHAPE_SMOOTH_SIGMA = 2.0


def _circular_gauss_smooth_ring(ring, valid_mask, sigma):
    """
    Circular Gaussian smoothing of one ring, matching CSO CircularFilter.

    Steps (matching CSO CircularFilter):
      1. Fill holes (NaN/invalid) via nearest-neighbour interpolation
      2. Build 1D Gaussian kernel (width = ceil(3*sigma)*2 + 1)
      3. Circular convolution (wrap at 0/255)
      4. Restore NaN at originally-invalid positions

    Parameters
    ----------
    ring : np.ndarray, shape (n_cols,)
    valid_mask : np.ndarray of bool, shape (n_cols,)
    sigma : float

    Returns
    -------
    np.ndarray, shape (n_cols,)
        Smoothed ring with NaN at originally-invalid positions.
    """
    n = len(ring)
    n_valid = np.count_nonzero(valid_mask)
    if n_valid == 0:
        return np.full(n, np.nan)

    # Step 1: fill holes via nearest-neighbour (matches CSO HolesFiller)
    filled = ring.copy()
    if n_valid < n:
        valid_idx = np.where(valid_mask)[0]
        for j in np.where(~valid_mask)[0]:
            dists = np.minimum(
                np.abs(valid_idx - j),
                n - np.abs(valid_idx - j),
            )
            filled[j] = ring[valid_idx[np.argmin(dists)]]

    # Step 2: Gaussian kernel
    hw = int(np.ceil(3.0 * sigma))
    x = np.arange(-hw, hw + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    # Step 3: circular convolution (wrap-around)
    padded = np.concatenate([filled[-hw:], filled, filled[:hw]])
    smoothed = np.convolve(padded, kernel, mode="valid")[:n]

    # Step 4: restore NaN at originally-invalid positions
    smoothed[~valid_mask] = np.nan

    return smoothed


def _compute_rms_smoothed(z_grid, h_grid, theta_grid, in_zone, R_mean, p, dR, alpha):
    """
    RMS of circular-Gaussian-smoothed, mean-centred BFS residuals.

    Matches the CSO pipeline (chapter_best_fit_surfaces.md Sections 10-11):
      1. Compute torus-model heights on the 2D polar grid (valid cells).
      2. Compute residuals = measured - model.
      3. Apply **circular** Gaussian smooth (sigma=2 meridian samples)
         independently on each ring (1D along theta, NOT 2D).
      4. Mean-centre the smoothed residuals within the fitting zone.
      5. RMS = sqrt( sum(f^2) / (N-1) ).

    Returns RMS in *microns* (mm * 1000).
    """
    n_rows, n_cols = z_grid.shape

    # Build the 2D residual grid (NaN for out-of-zone cells)
    resid_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    h2 = h_grid[in_zone] ** 2
    theta_v = theta_grid[in_zone]
    z_v = z_grid[in_zone]

    R_theta = R_mean + dR * np.cos(2.0 * (theta_v - alpha))
    R_theta2 = R_theta**2
    disc = 1.0 - p * h2 / R_theta2
    disc = np.maximum(disc, 0.0)
    z_model = h2 / (R_theta * (1.0 + np.sqrt(disc)))

    resid_grid[in_zone] = z_v - z_model

    # --- Per-ring circular Gaussian smoothing (matches CSO CircularFilter) ---
    smoothed_grid = np.full_like(resid_grid, np.nan)
    for i in range(n_rows):
        ring_valid = in_zone[i, :]
        if not np.any(ring_valid):
            continue
        smoothed_grid[i, :] = _circular_gauss_smooth_ring(
            resid_grid[i, :], ring_valid, _SHAPE_SMOOTH_SIGMA
        )

    # --- Mean-centre within the fitting zone (matches CSO Section 11.1) ---
    valid_after = in_zone & ~np.isnan(smoothed_grid)
    vals = smoothed_grid[valid_after]
    n = len(vals)
    if n < 2:
        return 0.0

    avg = np.mean(vals)
    vals_centred = vals - avg

    # RMS = sqrt( sum((f - avg)^2) / (N - 1) )
    rms_mm = np.sqrt(np.sum(vals_centred**2) / (n - 1))
    return rms_mm * 1000.0  # mm -> um


def _remove_tilt(h_valid, z_valid, theta_valid):
    """
    Remove surface tilt before toroidal conic fitting.

    Matches CSO's preprocessing: fit rotationally symmetric conic
    as reference, then fit/subtract a plane from the residuals.
    """
    # Step 1: Fit 2-parameter conic (R, p) via normal equations
    # Using the same negated-z convention as the torus fit
    zv = -z_valid
    h2 = h_valid**2
    z2 = zv**2
    z3 = zv * z2
    z4 = z2 * z2

    A = np.array([[np.sum(z4), np.sum(z3)], [np.sum(z3), np.sum(z2)]])
    b = np.array([-np.sum(h2 * z2), -np.sum(h2 * zv)])

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return z_valid  # can't remove tilt, return original

    p_ref = x[0]
    R_ref = x[1] / 2.0

    if abs(R_ref) < 1e-6:
        return z_valid

    # Compute reference surface heights (positive z convention)
    R_abs = abs(R_ref)
    disc = 1.0 - p_ref * h2 / (R_abs**2)
    disc = np.maximum(disc, 0.0)
    z_ref = h2 / (R_abs * (1.0 + np.sqrt(disc)))

    # Step 2: Residuals
    z_residual = z_valid - z_ref

    # Step 3: Fit plane to residuals: z_res = a*x + b*y + c
    cos_t = h_valid * np.cos(theta_valid)  # x coordinates
    sin_t = h_valid * np.sin(theta_valid)  # y coordinates

    # Build design matrix [x, y, 1]
    M = np.column_stack([cos_t, sin_t, np.ones(len(z_residual))])
    try:
        plane_coeffs, _, _, _ = np.linalg.lstsq(M, z_residual, rcond=None)
    except np.linalg.LinAlgError:
        return z_valid

    a_tilt, b_tilt = plane_coeffs[0], plane_coeffs[1]

    # Step 4: Subtract tilt from original heights (not the constant)
    z_corrected = z_valid - (a_tilt * cos_t + b_tilt * sin_t)

    return z_corrected


def _fit_biconic(h_valid, z_valid, theta_valid):
    """
    5-parameter biconic fit: Rx, Ry, px, py, alpha.

    Uses Levenberg-Marquardt with initial guess from the torus fit.
    Supplementary research output -- not used for CSO-compatible indices.

    The biconic surface equation is::

        z(x', y') = (x'^2/Rx + y'^2/Ry)
                   / (1 + sqrt(1 - px*x'^2/Rx^2 - py*y'^2/Ry^2))

    where (x', y') are coordinates rotated by alpha to align with the
    principal meridians.

    Reference: Navarro R et al., JOSAA 2006;23:969-977.

    Parameters
    ----------
    h_valid, z_valid, theta_valid : np.ndarray (1-D, same length)
        Radial distances (mm), surface heights (mm, positive), and
        meridian angles (radians) of valid data points.  Same inputs
        as _fit_torus_4param (after tilt removal).

    Returns
    -------
    tuple or None
        (Rx, Ry, px, py, alpha_deg, rms_um) on success.
        Rx, Ry in mm; alpha_deg in degrees [0, 180); rms_um in microns.
        None on failure (insufficient data, singular torus, or
        non-convergent optimizer).
    """
    if len(h_valid) < _MIN_POINTS_FOR_FIT:
        return None

    # ---- Step 1: torus fit for initial guess ----
    try:
        R_mean, p, dR, alpha0 = _fit_torus_4param(h_valid, z_valid, theta_valid)
    except np.linalg.LinAlgError:
        return None

    Rx0 = abs(R_mean) + abs(dR)
    Ry0 = abs(R_mean) - abs(dR)
    # Ensure Ry0 is positive (very low astigmatism can push it near zero)
    if Ry0 < 0.5:
        Ry0 = 0.5

    # Cartesian coordinates of each data point
    x_cart = h_valid * np.cos(theta_valid)
    y_cart = h_valid * np.sin(theta_valid)

    # ---- Step 2: biconic residual function ----
    def _residuals(params):
        Rx, Ry, px, py, alpha = params
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        # Rotate to principal meridian frame
        xp = x_cart * cos_a + y_cart * sin_a
        yp = -x_cart * sin_a + y_cart * cos_a
        xp2 = xp * xp
        yp2 = yp * yp

        numerator = xp2 / Rx + yp2 / Ry
        disc = 1.0 - px * xp2 / (Rx * Rx) - py * yp2 / (Ry * Ry)
        # Clamp discriminant to small positive value for numerical safety
        disc = np.maximum(disc, 1e-12)
        z_model = numerator / (1.0 + np.sqrt(disc))
        return z_valid - z_model

    # ---- Step 3: solve ----
    x0 = np.array([Rx0, Ry0, p, p, alpha0])
    bounds_lower = np.array([3.0, 3.0, -2.0, -2.0, -math.pi])
    bounds_upper = np.array([15.0, 15.0, 5.0, 5.0, math.pi])

    # Clamp initial guess to bounds
    x0 = np.clip(x0, bounds_lower, bounds_upper)

    try:
        result = _scipy_least_squares(
            _residuals,
            x0,
            method="trf",
            bounds=(bounds_lower, bounds_upper),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=2000,
        )
    except (np.linalg.LinAlgError, ValueError):
        return None

    if not result.success and result.cost > 1e-3:
        return None

    Rx, Ry, px, py, alpha = result.x

    # Ensure Rx >= Ry (flat >= steep) by swapping if needed
    if Rx < Ry:
        Rx, Ry = Ry, Rx
        px, py = py, px
        alpha = alpha + math.pi / 2.0

    # Fold alpha into [0, 180) degrees
    alpha_deg = math.degrees(alpha) % 180.0

    # RMS from the optimized residuals (mm -> um)
    rms_um = float(np.sqrt(np.mean(result.fun**2)) * 1000.0)

    return (float(Rx), float(Ry), float(px), float(py), round(alpha_deg, 2), round(rms_um, 4))


def _shape_indices_one_surface(elev_map, surface_label):
    """
    Compute shape indices for one surface (anterior or posterior) at all
    9 analysis diameters.

    Parameters
    ----------
    elev_map : np.ndarray, shape (n_rows, 256)
        Elevation heights in mm (already sign-adjusted by the caller),
        with -1000 for missing data.
    surface_label : str
        "ant" or "post" -- used in output key names.

    Returns
    -------
    dict
        Up to 54 key-value pairs (6 per diameter x 9 diameters).
        Keys follow the pattern  shape_{d}mm_{surface}_{param}.
    """
    clean = elev_map.astype(np.float64)
    n_rows, n_cols = clean.shape

    # Build coordinate grids
    h_grid = np.outer(np.arange(n_rows) * _R_STEP_MM, np.ones(n_cols))
    theta_grid = np.tile(_THETA_RAD[:n_cols], (n_rows, 1))

    # Mark invalid cells: sentinel value only.
    # Elevation values CAN be 0 (at apex) -- do NOT filter by <= 0.
    invalid = (clean == _MISSING) | np.isnan(clean)

    # Elevation values ARE surface heights in mm -- no conversion needed.
    # The caller has already applied sign adjustment (negation) to match
    # CSO convention where positive z = sag into the cornea.
    z_grid = clean.copy()

    # Mask out invalid cells in the height grid
    z_grid[invalid] = np.nan
    invalid = invalid | np.isnan(z_grid)

    results = {}

    for D in _SHAPE_DIAMETERS:
        hmax = D / 2.0

        # Diameter label for key names: "4.5" -> "4p5", integers stay as-is
        if D == int(D):
            d_label = str(int(D))
        else:
            d_label = str(D).replace(".", "p")

        prefix = f"shape_{d_label}mm_{surface_label}"

        # Full analysis zone: all valid points within the diameter.
        # CSO includes row 0 (apex, h=0) in the RMS computation zone.
        in_zone = (h_grid <= hmax) & (~invalid)

        # Fitting zone: exclude row 0 for the torus fit.
        # At h=0, z=0 always, so row 0 contributes only zeros to the
        # normal equations and cannot cause division issues.  However,
        # row 0 IS included in the RMS zone (see below) because CSO's
        # GetRMS iterates over all rows where ro <= Rmax.
        in_zone_fit = in_zone.copy()
        in_zone_fit[0, :] = False

        n_valid = np.count_nonzero(in_zone_fit)

        if n_valid < _MIN_POINTS_FOR_FIT:
            results[f"{prefix}_fitting_diameter"] = None
            results[f"{prefix}_rflat"] = None
            results[f"{prefix}_rsteep"] = None
            results[f"{prefix}_asphericity"] = None
            results[f"{prefix}_rflat_axis"] = None
            results[f"{prefix}_rms"] = None
            continue

        h_valid = h_grid[in_zone_fit]
        z_valid = z_grid[in_zone_fit]
        theta_valid = theta_grid[in_zone_fit]

        # Remove surface tilt before torus fitting (matches CSO preprocessing)
        z_valid = _remove_tilt(h_valid, z_valid, theta_valid)

        # ---- 4-parameter torus fit ----
        try:
            R_mean, p, dR, alpha = _fit_torus_4param(h_valid, z_valid, theta_valid)
        except np.linalg.LinAlgError:
            # Singular matrix -- not enough data diversity
            results[f"{prefix}_fitting_diameter"] = None
            results[f"{prefix}_rflat"] = None
            results[f"{prefix}_rsteep"] = None
            results[f"{prefix}_asphericity"] = None
            results[f"{prefix}_rflat_axis"] = None
            results[f"{prefix}_rms"] = None
            continue

        # Per Section 10.4 correction: absolute values applied
        Rflat = abs(R_mean) + abs(dR)
        Rsteep = abs(R_mean) - abs(dR)

        # Axis of flattest meridian: fold to [0, 180) degrees (TABO)
        # _fit_torus_4param now returns alpha = atan2(x[3],x[2]) / 2,
        # which is the physical flat-axis angle in radians.
        rflat_axis_deg = np.degrees(alpha) % 180.0

        # RMS of Gaussian-smoothed residuals against the torus model (um).
        # CSO applies sigma=2 smoothing to BFS residuals before computing
        # RMS (chapter_best_fit_surfaces.md Section 11.2).  We put the
        # tilt-corrected heights back into a 2D grid so the Gaussian
        # kernel works in polar-pixel space.
        #
        # Use the FULL zone (in_zone, including row 0) for RMS, matching
        # CSO's GetRMS which checks ro <= Rmax (includes row 0 at ro=0).
        # Row 0 has z=0 and z_model=0, so its residual is 0.  Including
        # it matches CSO's point count and slightly reduces the RMS.
        z_grid_tc = z_grid.copy()
        z_grid_tc[in_zone_fit] = z_valid  # tilt-corrected heights
        z_grid_tc[~in_zone] = np.nan  # mark out-of-zone as NaN
        rms = _compute_rms_smoothed(z_grid_tc, h_grid, theta_grid, in_zone, R_mean, p, dR, alpha)

        results[f"{prefix}_fitting_diameter"] = float(D)
        results[f"{prefix}_rflat"] = float(Rflat)
        results[f"{prefix}_rsteep"] = float(Rsteep)
        results[f"{prefix}_asphericity"] = float(p)
        results[f"{prefix}_rflat_axis"] = round(rflat_axis_deg)
        results[f"{prefix}_rms"] = float(rms)

    return results


def compute_shape_indices(raw_segments: dict) -> dict:
    """
    Compute ~108 shape index columns from the elevation polar maps.

    For each of 9 fitting diameters (3, 4, 4.5, 5, 6, 7, 8, 9, 10 mm)
    and 2 surfaces (anterior, posterior), computes 6 values via a
    toroidal conic-section fit:

      - Rflat:           radius of flattest meridian (mm)
      - Rsteep:          radius of steepest meridian (mm)
      - Asphericity (p): conic shape factor (p=1 sphere, p<1 prolate, p>1 oblate)
      - RflatAxis:       axis of flattest meridian in degrees (TABO)
      - FittingDiameter: actual diameter used for fitting (mm)
      - RMS:             root mean square of fit residuals (um)

    The fitting algorithm matches CSO's
    GetBestFitTorus (4-parameter variant).  See formulas_shape_indices.md
    Sections 3.3 and 10.4 for the derivation and correction notes.

    Height source
    -------------
    CSO computes shape indices from raw elevation data
    (``-m_ElevationAnterior``), NOT from sagittal curvature maps.
    The elevation values are surface heights in mm.  CSO negates them
    so the z-axis points into the cornea (positive sag for convex
    surfaces).

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of core.parse_csv().  Must contain 'elevation_anterior'
        and/or 'elevation_posterior' maps with shape (n_rows, 256).

    Returns
    -------
    dict
        Up to 108 named features.  Keys follow the pattern:
          shape_{d}mm_{surface}_{param}
        Examples:
          shape_3mm_ant_rflat
          shape_4p5mm_post_asphericity
          shape_10mm_ant_rms
    """
    out = {}

    # Generate all expected keys (even when maps are missing -> None)
    _shape_params = ["fitting_diameter", "rflat", "rsteep", "asphericity", "rflat_axis", "rms"]
    for surface in ("ant", "post"):
        for D in _SHAPE_DIAMETERS:
            dlabel = str(D).replace(".", "p") + "mm"
            for param in _shape_params:
                out[f"shape_{dlabel}_{surface}_{param}"] = None

    # --- Anterior: pass elevation directly as positive heights ---
    # The MS-39 elevation map stores positive heights (0 at apex,
    # increasing outward for a convex surface).  These are exactly
    # the surface heights z that the conic torus fit needs.
    elev_ant = raw_segments.get("elevation_anterior")
    if elev_ant is not None:
        out.update(_shape_indices_one_surface(elev_ant, "ant"))

    # --- Posterior: re-reference to apex, pass as positive heights ---
    # The posterior elevation map is offset by the corneal thickness
    # at the apex.  Subtract the apex value so height = 0 at the
    # posterior apex.
    elev_post = raw_segments.get("elevation_posterior")
    if elev_post is not None:
        ep = elev_post.astype(np.float64)
        # Apex value: row 0, use mean of valid values across all meridians
        apex_vals = ep[0, :]
        valid_apex = apex_vals[apex_vals != _MISSING]
        if len(valid_apex) > 0:
            apex_ref = np.mean(valid_apex)
        else:
            apex_ref = 0.0
        z_post = ep - apex_ref
        # Preserve sentinel values
        z_post[elev_post == _MISSING] = _MISSING
        out.update(_shape_indices_one_surface(z_post, "post"))

    # --- Shape normative limits (P95 / P99) ---
    # Quadratic normative RMS limits (population-derived thresholds).
    #         (ShapeIndicesModuleT.Get95Limit / Get99Limit).
    #
    # Quadratic RMS limits as a function of fitting diameter Phi (mm):
    #   Anterior P95: 0.0007 * Phi^2 + 0.0059 * Phi
    #   Anterior P99: 0.0011 * Phi^2 + 0.0086 * Phi
    #   Posterior P95: 0.0005 * Phi^2 + 0.0221 * Phi
    #   Posterior P99: 0.0056 * Phi^2 + 0.01 * Phi
    #
    # These are population-derived RMS thresholds: values above P99
    # indicate abnormal corneal irregularity at that diameter.
    for surface_tag, is_anterior in [("ant", True), ("post", False)]:
        for D in _SHAPE_DIAMETERS:
            dlabel = str(D).replace(".", "p") + "mm"
            prefix = f"shape_{dlabel}_{surface_tag}"

            p95 = shape_rms_normative_p95(D, anterior=is_anterior)
            p99 = shape_rms_normative_p99(D, anterior=is_anterior)
            out[f"{prefix}_rms_p95_limit"] = p95
            out[f"{prefix}_rms_p99_limit"] = p99

            rms_val = out.get(f"{prefix}_rms")
            if rms_val is not None:
                out[f"{prefix}_rms_abnormal"] = rms_val > p99
            else:
                out[f"{prefix}_rms_abnormal"] = None

    return out


def shape_rms_normative_p95(phi_mm, anterior=True):
    """
    Compute the P95 normative RMS limit for shape indices at a given diameter.

    Matches CSO ShapeIndicesModuleT.Get95Limit.

    Parameters
    ----------
    phi_mm : float
        Fitting diameter in mm.
    anterior : bool
        True for anterior surface, False for posterior.

    Returns
    -------
    float
        P95 RMS threshold in microns.
    """
    if anterior:
        return 0.0007 * phi_mm * phi_mm + 0.0059 * phi_mm
    return 0.0005 * phi_mm * phi_mm + 0.0221 * phi_mm


def shape_rms_normative_p99(phi_mm, anterior=True):
    """
    Compute the P99 normative RMS limit for shape indices at a given diameter.

    Matches CSO ShapeIndicesModuleT.Get99Limit.

    Parameters
    ----------
    phi_mm : float
        Fitting diameter in mm.
    anterior : bool
        True for anterior surface, False for posterior.

    Returns
    -------
    float
        P99 RMS threshold in microns.
    """
    if anterior:
        return 0.0011 * phi_mm * phi_mm + 0.0086 * phi_mm
    return 0.0056 * phi_mm * phi_mm + 0.01 * phi_mm


# ==========================================================================
# KERATOCONUS SCREENING INDICES
# ==========================================================================
#
# Source: formulas_screening_indices.md
#
# Indices computable from polar maps alone:
#   SIf, SIb, CSIf, CSIb, ThkSI, PTI, PEpiTI
#
# Indices requiring Zernike coefficients (NOT computable from individual CSV):
#   RMSf, RMSb, EIf, EIb, fbLikeness, PDThkSI
# ==========================================================================

# Refractive index deltas for dioptric conversion
_DELTA_N_ANTERIOR = 0.3375  # n_keratometric - 1 = 1.3375 - 1
_DELTA_N_POSTERIOR = 0.04  # |n_aqueous - n_cornea| ~ |1.336 - 1.376|

# Reference angles for laterality (radians).
# Use exact conversion from integer degrees to avoid truncation-induced
# boundary sensitivity in the disc-based SIf/SIb computation.
_P0_ANGLE_OD = np.radians(241)  # 241 deg -- infero-temporal OD
_P0_ANGLE_OS = np.radians(299)  # 299 deg -- infero-temporal OS
_P0_ANGLE_UNSPECIFIED = np.radians(270)  # 270 deg -- straight inferior

# Minimum sample requirement per region
_SCREENING_MIN_SAMPLES = 100

# CSIf/CSIb / ThkSI trimming fraction (5% each tail)
_SCREENING_TRIM_ALPHA = 0.05

# PTI normative percentile arrays (24 rings, from formulas_screening_indices.md Section 16)
_PTI95 = np.array(
    [
        0,
        0.625,
        1.069,
        1.435,
        1.846,
        2.341,
        2.915,
        3.654,
        4.526,
        5.502,
        6.622,
        7.861,
        9.218,
        10.762,
        12.451,
        14.239,
        16.245,
        18.333,
        20.626,
        22.969,
        25.199,
        27.752,
        30.358,
        32.924,
    ]
)
_PTI99 = np.array(
    [
        0,
        1.189,
        1.626,
        2.093,
        2.527,
        3.032,
        3.742,
        4.487,
        5.417,
        6.499,
        7.769,
        9.110,
        10.529,
        12.120,
        13.945,
        15.854,
        18.032,
        20.316,
        22.638,
        25.315,
        28.440,
        31.121,
        33.714,
        36.272,
    ]
)


def _get_p0_angle(exam_eye: str) -> float:
    """Return the reference angle (radians) for the infero-temporal direction."""
    eye = str(exam_eye).strip().upper()
    if eye == "OD":
        return _P0_ANGLE_OD
    elif eye == "OS":
        return _P0_ANGLE_OS
    return _P0_ANGLE_UNSPECIFIED


def _polar_to_cart_cso(r, theta):
    """
    CSO convention: x = r*sin(theta), y = r*cos(theta).

    This is the convention used in GetCurvSI / GetThkSI -- note the
    swapped sin/cos compared to standard math convention.
    """
    return r * np.sin(theta), r * np.cos(theta)


def _screening_trimmed_mean(values, alpha=0.05):
    """
    Compute alpha-trimmed mean: sort, discard bottom and top alpha
    fraction, take mean of remainder.
    """
    arr = np.sort(values)
    n = len(arr)
    lo = int(np.floor(alpha * n))
    hi = n - lo
    if hi <= lo:
        return float(np.mean(arr))
    return float(np.mean(arr[lo:hi]))


def _clean_polar_map(polar_map):
    """Replace sentinel -1000 with NaN, return float64 copy."""
    clean = polar_map.astype(np.float64)
    clean[clean == _MISSING] = np.nan
    return clean


def _compute_curv_si(polar_map, delta_n, exam_eye, r0=1.5):
    """
    Curvature Symmetry Index (SIf or SIb).

    Compares mean curvature in an infero-temporal disc vs the diametrically
    opposite supero-nasal disc.  Returns diopters.

    Source: GetCurvSI() -- formulas_screening_indices.md Sections 2-3.
    """
    clean = _clean_polar_map(polar_map)
    n_rows, n_cols = clean.shape

    # Reference point P0 in infero-temporal direction
    angle = _get_p0_angle(exam_eye)
    p0_x, p0_y = _polar_to_cart_cso(r0, angle)

    theta_rad = np.arange(n_cols) * (2 * np.pi / n_cols)

    list_it = []
    list_sn = []

    for i in range(n_rows):
        r_mm = i * _R_STEP_MM
        for j in range(n_cols):
            if np.isnan(clean[i, j]):
                continue
            x, y = _polar_to_cart_cso(r_mm, theta_rad[j])
            dist = np.sqrt((x - p0_x) ** 2 + (y - p0_y) ** 2)
            if dist < r0:
                j_opp = (j + n_cols // 2) % n_cols
                val_opp = clean[i, j_opp]
                list_it.append(clean[i, j])
                if not np.isnan(val_opp):
                    list_sn.append(val_opp)

    if len(list_it) < _SCREENING_MIN_SAMPLES or len(list_sn) < _SCREENING_MIN_SAMPLES:
        return None

    mean_it = float(np.mean(list_it))
    mean_sn = float(np.mean(list_sn))

    if mean_it == 0.0 or mean_sn == 0.0:
        return None

    # 1000 * (1/R_IT - 1/R_SN) * delta_n  -> dioptric power difference
    return 1000.0 * (1.0 / mean_it - 1.0 / mean_sn) * delta_n


def _compute_simax(polar_map, delta_n, r0=1.5):
    """
    Maximum Symmetry Index: location-agnostic version of SIf/SIb.

    Sweeps the reference point through all meridians and returns
    the maximum absolute curvature asymmetry. Detects ectasia
    regardless of cone location (superior, central, PMD, etc.).

    Also returns the angle of maximum asymmetry (cone direction).

    Parameters
    ----------
    polar_map : np.ndarray, shape (n_rows, 256)
        Gaussian curvature map (radius of curvature in mm).
    delta_n : float
        Refractive index delta (0.3375 anterior, 0.04 posterior).
    r0 : float
        Radius of the sampling disc in mm (default 1.5).

    Returns
    -------
    simax : float or None
        Maximum absolute SI across all meridians (diopters).
    angle : float or None
        Meridian angle (degrees) where the maximum was found.
    """
    clean = _clean_polar_map(polar_map)
    n_rows, n_cols = clean.shape

    max_si = 0.0
    max_angle = 0.0

    # Sweep through meridians (step by 4 for speed: 64 positions)
    step = max(1, n_cols // 64)
    theta_rad = np.arange(n_cols) * (2 * np.pi / n_cols)

    for ref_idx in range(0, n_cols, step):
        angle = theta_rad[ref_idx]
        # Reference point at distance r0 along this meridian
        p0_x, p0_y = _polar_to_cart_cso(r0, angle)

        list_ref = []
        list_opp = []

        for i in range(n_rows):
            r_mm = i * _R_STEP_MM
            for j in range(n_cols):
                if np.isnan(clean[i, j]):
                    continue
                x, y = _polar_to_cart_cso(r_mm, theta_rad[j])
                dist = np.sqrt((x - p0_x) ** 2 + (y - p0_y) ** 2)
                if dist < r0:
                    j_opp = (j + n_cols // 2) % n_cols
                    list_ref.append(clean[i, j])
                    val_opp = clean[i, j_opp]
                    if not np.isnan(val_opp):
                        list_opp.append(val_opp)

        if len(list_ref) < _SCREENING_MIN_SAMPLES or len(list_opp) < _SCREENING_MIN_SAMPLES:
            continue

        mean_ref = float(np.mean(list_ref))
        mean_opp = float(np.mean(list_opp))

        if mean_ref == 0.0 or mean_opp == 0.0:
            continue

        si = abs(1000.0 * (1.0 / mean_ref - 1.0 / mean_opp) * delta_n)
        if si > max_si:
            max_si = si
            max_angle = np.degrees(angle)

    if max_si == 0.0:
        return None, None
    return float(max_si), float(max_angle)


def _compute_curv_csi(polar_map, delta_n, alpha=_SCREENING_TRIM_ALPHA):
    """
    Center-Surround Curvature Index (CSIf or CSIb).

    Compares trimmed-mean curvature in a central disc (0 < r < 1.5 mm)
    vs a peripheral annulus (1.5 <= r < 3.0 mm).  Ring 0 excluded.
    Returns diopters.

    Source: GetCurvCSI() -- formulas_screening_indices.md Sections 4-5.
    """
    clean = _clean_polar_map(polar_map)
    n_rows, n_cols = clean.shape

    center_vals = []
    periph_vals = []

    for i in range(1, n_rows):  # skip ring 0
        r_mm = i * _R_STEP_MM
        for j in range(n_cols):
            v = clean[i, j]
            if np.isnan(v):
                continue
            if r_mm < 1.5:
                center_vals.append(v)
            elif r_mm < 3.0:
                periph_vals.append(v)

    if len(center_vals) < _SCREENING_MIN_SAMPLES or len(periph_vals) < _SCREENING_MIN_SAMPLES:
        return None

    tm_center = _screening_trimmed_mean(np.array(center_vals), alpha)
    tm_periph = _screening_trimmed_mean(np.array(periph_vals), alpha)

    if tm_center == 0.0 or tm_periph == 0.0:
        return None

    return 1000.0 * (1.0 / tm_center - 1.0 / tm_periph) * delta_n


def _compute_thksi(thk_map, r0=1.5):
    """
    Thickness Symmetry Index (ThkSI).

    Compares trimmed-mean thickness in an inferior disc vs superior,
    reference at (0, -R0) = straight inferior (laterality-independent).
    Returns micrometers.

    Source: GetThkSI() -- formulas_screening_indices.md Section 9.
    """
    clean = _clean_polar_map(thk_map)
    n_rows, n_cols = clean.shape

    # Reference point: straight inferior in CSO convention (x=r*sin, y=r*cos).
    # Inferior = 270 deg TABO = theta = 3*pi/2:
    #   x = r0*sin(3*pi/2) = -r0,  y = r0*cos(3*pi/2) = 0.
    p0_x, p0_y = -r0, 0.0

    theta_rad = np.arange(n_cols) * (2 * np.pi / n_cols)

    list_inf = []
    list_sup = []

    for i in range(n_rows):
        r_mm = i * _R_STEP_MM
        if r_mm >= r0:
            continue  # doc says ro[i] < R0
        for j in range(n_cols):
            if np.isnan(clean[i, j]):
                continue
            x, y = _polar_to_cart_cso(r_mm, theta_rad[j])
            dist = np.sqrt((x - p0_x) ** 2 + (y - p0_y) ** 2)
            if dist < r0:
                j_opp = (j + n_cols // 2) % n_cols
                val_opp = clean[i, j_opp]
                list_inf.append(clean[i, j])
                if not np.isnan(val_opp):
                    list_sup.append(val_opp)

    if len(list_inf) < _SCREENING_MIN_SAMPLES or len(list_sup) < _SCREENING_MIN_SAMPLES:
        return None

    tm_sup = _screening_trimmed_mean(np.array(list_sup), _SCREENING_TRIM_ALPHA)
    tm_inf = _screening_trimmed_mean(np.array(list_inf), _SCREENING_TRIM_ALPHA)

    # Positive = thinner inferiorly (consistent with KC)
    return tm_sup - tm_inf


def _bilinear_interp_polar(clean_map, r_mm, theta_rad, r_step=_R_STEP_MM):
    """
    Bilinear interpolation on a clean polar map at (r_mm, theta_rad).

    Returns interpolated value or NaN if any neighbor is NaN or out of bounds.
    """
    n_rows, n_cols = clean_map.shape

    # Fractional ring index
    ri = r_mm / r_step
    if ri < 0 or ri >= n_rows - 1:
        return np.nan
    i0 = int(np.floor(ri))
    i1 = i0 + 1
    fr = ri - i0

    # Fractional meridian index (wrap around)
    theta_norm = theta_rad % (2 * np.pi)
    jf = theta_norm / (2 * np.pi / n_cols)
    j0 = int(np.floor(jf)) % n_cols
    j1 = (j0 + 1) % n_cols
    ft = jf - np.floor(jf)

    # Four corner values
    v00 = clean_map[i0, j0]
    v01 = clean_map[i0, j1]
    v10 = clean_map[i1, j0]
    v11 = clean_map[i1, j1]

    if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
        return np.nan

    return v00 * (1 - fr) * (1 - ft) + v01 * (1 - fr) * ft + v10 * fr * (1 - ft) + v11 * fr * ft


def _compute_ctsp(clean_thk, center_x, center_y, max_radius=3.0):
    """
    Corneal Thickness Spatial Profile (CTSP).

    Re-centers the thickness map on (center_x, center_y) and averages
    concentric rings.  Returns CTSP array (one value per ring) or None.

    CSO uses:
        x = ro[i] * cos(-2*pi*j/length) + ThkMin.X
        y = ro[i] * sin(-2*pi*j/length) - ThkMin.Y
    then converts back to polar for interpolation.

    Source: formulas_screening_indices.md Section 11, Step 1.
    """
    n_rows, n_cols = clean_thk.shape
    n_rings = min(n_rows, int(max_radius / _R_STEP_MM) + 1)

    ctsp = np.full(n_rings, np.nan)
    # CSO requires 2/3 of meridians.  For severe KC with holes near the
    # cone, relax to 1/3 as a minimum -- the annular average is still
    # stable with ~85 points.
    min_valid_relaxed = n_cols // 3

    had_good_ring = False
    consecutive_bad_after_good = 0
    for i in range(n_rings):
        r_mm = i * _R_STEP_MM
        count = 0
        total = 0.0

        for j in range(n_cols):
            angle = -2 * np.pi * j / n_cols
            x = r_mm * np.cos(angle) + center_x
            y = r_mm * np.sin(angle) - center_y

            # Convert back to polar relative to original grid
            ro_new = np.sqrt(x * x + y * y)
            theta_new = np.arctan2(-y, x)
            if theta_new < 0:
                theta_new += 2 * np.pi

            val = _bilinear_interp_polar(clean_thk, ro_new, theta_new)
            if not np.isnan(val):
                total += val
                count += 1

        if count >= min_valid_relaxed:
            ctsp[i] = total / count
            had_good_ring = True
            consecutive_bad_after_good = 0
        else:
            # In severe KC, the innermost rings (near the cone) may have
            # holes from OCT posterior-surface loss.  Don't stop scanning
            # until we've had good rings and then hit a gap.
            if had_good_ring:
                consecutive_bad_after_good += 1
                if consecutive_bad_after_good >= 2:
                    break

    valid_rings = int(np.sum(~np.isnan(ctsp)))
    if valid_rings < 3:
        return None

    return ctsp


def _compute_pti(thk_map, thk_min_x, thk_min_y, normatives=None):
    """
    Pachymetric Thickness Index (PTI).

    Measures how abnormally fast the cornea thickens from its thinnest
    point outward vs normative data.  Returns dimensionless log-scaled
    percentile value.

    Parameters
    ----------
    thk_map : np.ndarray
        Corneal thickness polar map.
    thk_min_x, thk_min_y : float
        Location of minimum thickness (mm).
    normatives : dict or None
        If provided, should contain 'pti95' and 'pti99' arrays
        (24 elements each) for the patient's population group.
        If None, uses CSO's default European normatives.

    Source: formulas_screening_indices.md Section 11.
    """
    clean = _clean_polar_map(thk_map)

    ctsp = _compute_ctsp(clean, thk_min_x, thk_min_y)
    if ctsp is None:
        return None

    # PTI Array: percentage change from center
    if ctsp[0] == 0 or np.isnan(ctsp[0]):
        return None

    pti_array = np.full_like(ctsp, np.nan)
    for k in range(len(ctsp)):
        if not np.isnan(ctsp[k]):
            pti_array[k] = 100.0 * (ctsp[k] - ctsp[0]) / ctsp[0]

    # Select normative arrays: population-adaptive if provided, else CSO defaults
    pti95 = normatives["pti95"] if normatives else _PTI95
    pti99 = normatives["pti99"] if normatives else _PTI99

    # Final PTI: max over rings within 3.0mm where normative data exists
    max_ring = min(len(pti_array), len(pti95))
    pti_val = None

    for k in range(1, max_ring):  # skip ring 0 (PTI95[0] = 0)
        r_mm = k * _R_STEP_MM
        if r_mm > 3.0:
            break
        if np.isnan(pti_array[k]):
            break  # stop at first invalid
        if pti95[k] <= 0 or pti99[k] <= 0:
            continue
        if pti_array[k] <= 0:
            continue
        ratio_95 = pti_array[k] / pti95[k]
        ratio_99_95 = pti99[k] / pti95[k]
        if ratio_95 <= 0 or ratio_99_95 <= 0 or ratio_99_95 == 1.0:
            continue
        val = np.log10(ratio_95) / np.log10(ratio_99_95)
        if pti_val is None or val > pti_val:
            pti_val = val

    return pti_val


def _select_pti_normatives(cct_um):
    """
    Select PTI normative arrays based on central corneal thickness.

    CCT bins (um):
      <510: thin (African/South Asian typical)
      510-530: below average
      530-550: average (European typical)
      550-570: above average
      >570: thick

    Currently returns None (uses CSO defaults).
    Will be populated from the 200k exam dataset.

    Parameters
    ----------
    cct_um : float or None
        Central corneal thickness in micrometers.

    Returns
    -------
    dict or None
        If a population-specific normative table is available, returns
        ``{'pti95': np.ndarray, 'pti99': np.ndarray}``.
        Otherwise returns None, which causes ``_compute_pti`` to use
        the hardcoded CSO European defaults.
    """
    # Phase B: load from JSON normative tables keyed by CCT bin
    # For now, return None to use CSO defaults
    return None


def _compute_pepiti(epi_map, thk_min_x, thk_min_y):
    """
    Pachymetric Epithelial Thickness Index (PEpiTI).

    Measures epithelial thickness progression from the thinnest epithelial
    point, weighted by the inferior location of the minimum.

    Simplified: uses the global EpiThkMin (from summary indices) as the
    center instead of the full Notable Points barycenter search, since
    we lack DZMax (requires Zernike).

    Source: formulas_screening_indices.md Section 12.
    """
    clean = _clean_polar_map(epi_map)

    # Find epithelial minimum with Limit=3.0 mm (CSO's GetEpiMin uses
    # ro[i+1] < 3.0).  Without this cap, peripheral OCT artifacts (e.g.
    # epithelial thinning at r > 3 mm in OS eyes) are picked up, placing
    # the minimum far from the cone and zeroing the location-dependent weight.
    # The full algorithm also restricts the search to a 2.0 mm radius around
    # NotablePtsBarycenter, but we lack DZMax so we skip that constraint.
    epi_min_val, epi_min_x, epi_min_y = _find_extremum(epi_map, mode="min", max_radius=3.0)
    if epi_min_val is None:
        return None

    # Build epithelial CTSP centered on EpiThkMin
    ctsp = _compute_ctsp(clean, epi_min_x, epi_min_y)
    if ctsp is None:
        return None

    if ctsp[0] == 0 or np.isnan(ctsp[0]):
        return None

    # Epithelial PTI array (percentage change from center)
    epi_pti_array = np.full_like(ctsp, np.nan)
    for k in range(len(ctsp)):
        if not np.isnan(ctsp[k]):
            epi_pti_array[k] = 100.0 * (ctsp[k] - ctsp[0]) / ctsp[0]

    # PEpiTI_raw = max over rings within 3.0 mm
    pepiti_raw = None
    for k in range(1, len(epi_pti_array)):
        r_mm = k * _R_STEP_MM
        if r_mm > 3.0:
            break
        if np.isnan(epi_pti_array[k]):
            break
        if pepiti_raw is None or epi_pti_array[k] > pepiti_raw:
            pepiti_raw = epi_pti_array[k]

    if pepiti_raw is None:
        return None

    # Location-dependent weight: clamp((3.0 - CenterY) / 2.0, 0, 1)
    # CenterY positive = inferior in CSO's clinical convention.
    # _find_extremum uses x=r*cos(theta), y=r*sin(theta) (standard math).
    # In standard math, inferior (270 deg) -> sin(270)=-1 -> y<0.
    # CSO's CenterY is positive-inferior, so CenterY = -epi_min_y.
    center_y = -epi_min_y
    weight = float(np.clip((3.0 - center_y) / 2.0, 0.0, 1.0))

    return float(pepiti_raw * weight)


# ==========================================================================
# PDThkSI  (Population-Deviation Thickness Symmetry Index)
# ==========================================================================
#
# Normative Zernike arrays for population-deviation thickness Z-scoring.
#
# The 6 normative arrays (ThkM, ThkSD, EpiThkM, EpiThkSD, StrThkM, StrThkSD)
# each contain 36 Zernike coefficients (order 7) that reconstruct the
# population-average thickness map and its SD on the polar grid.
#
# PDThkSI pipeline:
#   1. Reconstruct normative mean+SD maps from Zernike coefficients.
#   2. Z-score the patient's thickness map: z_map = (thk - mean_norm) / sd_norm
#   3. Apply laterality mirror for OS: col -> (3*N/2 - col) % N
#   4. Run GetThkSIn on the Z-scored map (same as ThkSI but with r0=2.0mm)
# ==========================================================================

# Normative Zernike coefficients (36 terms, order 7).  Used to reconstruct
# population-average thickness maps for Z-scoring.

_NORM_THK_MEAN = np.array(
    [
        601.4998,
        10.661249,
        12.507556,
        -0.64886314,
        29.477129,
        -6.4408226,
        -0.42685696,
        -0.1431971,
        -0.09358688,
        0.5060259,
        -0.11920072,
        -0.13358428,
        0.62308675,
        0.3572415,
        -0.10614243,
        0.059476495,
        0.000125924,
        -0.3027365,
        -0.10990147,
        0.14722215,
        -0.015666874,
        -0.052826717,
        0.017867163,
        0.016722912,
        0.13262406,
        0.17166604,
        -0.006914712,
        -0.016416455,
        -0.00656847,
        -0.026803024,
        0.002118993,
        -0.076883495,
        -0.006653272,
        -0.013670836,
        0.003125859,
        0.022794235,
    ],
    dtype=np.float64,
)

# ThkSD[7] was obfuscated as (float)Math.E * 189f / 286f = 1.7963470824...
# StrThkSD[13] was obfuscated as (float)Math.PI * 113f / 291f = 1.21993116...
_NORM_THK_SD = np.array(
    [
        34.86008,
        5.8851037,
        5.535323,
        1.7590903,
        4.9016113,
        2.792961,
        1.5596858,
        1.7963470824432153,
        0.8928315,
        1.1997348,
        1.054491,
        0.6025547,
        1.1017715,
        1.0023795,
        1.2773968,
        0.82817924,
        0.61279863,
        0.6858587,
        0.3524995,
        0.47860324,
        0.86822355,
        0.72350836,
        0.39985517,
        0.32739222,
        0.4577736,
        0.4080985,
        0.42954874,
        0.66885334,
        0.48644572,
        0.31291568,
        0.26955637,
        0.29510468,
        0.20994735,
        0.25406918,
        0.31850675,
        0.5429866,
    ],
    dtype=np.float64,
)

# Epithelial and stromal normative arrays (included for completeness;
# only ThkM/ThkSD are needed for PDThkSI).
_NORM_EPI_THK_MEAN = np.array(
    [
        52.78585,
        -1.643747,
        0.5815672,
        0.2574188,
        -0.6844717,
        0.6614001,
        0.2285293,
        -0.3128191,
        0.1046776,
        0.1151007,
        -0.004136913,
        0.03423735,
        -0.2735508,
        0.3815731,
        -0.1836213,
        -0.02889536,
        0.000821681,
        -0.09697807,
        0.04753661,
        0.04031591,
        -0.03689218,
        -0.01228882,
        0.003147262,
        -0.02037665,
        0.05111621,
        0.02371641,
        -0.02080496,
        0.01689941,
        -0.01043059,
        -0.003060486,
        0.007960317,
        -0.005914009,
        0.002148654,
        0.00608072,
        0.01346058,
        -0.007114129,
    ],
    dtype=np.float64,
)

_NORM_EPI_THK_SD = np.array(
    [
        6.170407,
        2.020006,
        -0.1759111,
        -0.2652197,
        1.841234,
        -2.21957,
        -1.527079,
        1.663975,
        -0.1164867,
        0.3842285,
        0.3794179,
        -0.1687229,
        1.157805,
        -1.402634,
        1.321936,
        0.7166095,
        -0.8584267,
        0.8817713,
        -0.04170836,
        0.2124647,
        -0.4000874,
        -0.2772595,
        0.2032197,
        -0.09684863,
        0.5090056,
        -0.6533328,
        0.6145517,
        -0.5489709,
        -0.2133382,
        0.2285873,
        -0.2664808,
        0.2784584,
        -0.03884237,
        0.07484196,
        -0.1140139,
        0.1582638,
    ],
    dtype=np.float64,
)

_NORM_STR_THK_MEAN = np.array(
    [
        549.0687,
        12.079648,
        11.919627,
        -0.81474006,
        30.55507,
        -7.043895,
        -0.6039825,
        0.06297211,
        -0.17196378,
        0.41390654,
        -0.10649593,
        -0.18427862,
        0.62917805,
        0.076897226,
        -0.008301925,
        0.061684012,
        0.042709287,
        -0.30087602,
        -0.14525387,
        0.09834461,
        0.028379017,
        -0.02971658,
        -3.57949e-05,
        0.03616533,
        -0.003095726,
        0.17589384,
        -0.017560702,
        -0.015717827,
        0.004918261,
        -0.030355537,
        0.009944238,
        -0.09855505,
        -0.012043016,
        -0.020664455,
        0.000957065,
        0.020520885,
    ],
    dtype=np.float64,
)

_NORM_STR_THK_SD = np.array(
    [
        34.19834,
        6.134349,
        5.374725,
        1.8504745,
        4.949799,
        3.0655887,
        1.7698686,
        1.9848951,
        0.9208401,
        1.2532934,
        1.1093479,
        0.6483807,
        1.200315,
        1.21993116788882,
        1.4263525,
        0.9421415,
        0.75035584,
        0.83655185,
        0.38365155,
        0.5149085,
        0.9152382,
        0.75913423,
        0.43266264,
        0.35628653,
        0.49027288,
        0.50210845,
        0.5250195,
        0.732415,
        0.5450453,
        0.35515717,
        0.3124792,
        0.3306835,
        0.22734568,
        0.26891863,
        0.3394595,
        0.58287257,
    ],
    dtype=np.float64,
)


def _reconstruct_normative_map(zernike_coeffs, n_rows, n_cols, fitting_radius=4.0):
    """
    Reconstruct a normative map from 36 Zernike coefficients on a polar grid.

    Matches CSO's Altimetry.GetZernAltimetry: evaluates all 36 Zernike terms
    (order 7) at each (r, theta) point in the polar grid.

    Parameters
    ----------
    zernike_coeffs : np.ndarray, shape (36,)
        Zernike coefficients (order 7, 36 terms).
    n_rows : int
        Number of radial rings.
    n_cols : int
        Number of angular meridians (256).
    fitting_radius : float
        Normalization radius in mm (CSO uses 4.0).

    Returns
    -------
    np.ndarray, shape (n_rows, n_cols)
        Reconstructed map.  Points outside the fitting zone are NaN.
    """
    modes = _zernike_generate_modes(max_order=7)  # 36 terms
    out_map = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    theta_grid = np.arange(n_cols) * (2 * np.pi / n_cols)

    for i in range(n_rows):
        r_mm = i * _R_STEP_MM
        rho = r_mm / fitting_radius
        if rho > 1.0:
            break

        rho_arr = np.full(n_cols, rho)
        ring_vals = np.zeros(n_cols, dtype=np.float64)
        for k, (j, n, m) in enumerate(modes):
            if k >= len(zernike_coeffs):
                break
            ring_vals += zernike_coeffs[k] * _zernike_polynomial(n, m, rho_arr, theta_grid)
        out_map[i, :] = ring_vals

    return out_map


def _compute_pdthksi(thk_map, exam_eye, r0=2.0):
    """
    Compute PDThkSI (Population-Deviation Thickness Symmetry Index).

    Algorithm (matches CSO KeratoconusScreening):
      1. Reconstruct normative mean+SD maps from Zernike coefficients.
      2. Z-score: z_map = (thk - mean_norm) / sd_norm
      3. For OS eyes, mirror the normative map: col -> (3*N/2 - col) % N
      4. Apply GetThkSIn on the Z-scored map with r0=2.0mm.

    The GetThkSIn function is identical to GetThkSI but
    operates on the Z-scored map and uses a configurable r0.

    Parameters
    ----------
    thk_map : np.ndarray, shape (n_rows, 256)
        Corneal thickness polar map (micrometers), -1000 for missing.
    exam_eye : str
        "OD" or "OS".
    r0 : float
        Disc radius for the SI comparison (CSO uses 2.0 for PDThkSI).

    Returns
    -------
    float or None
        PDThkSI value, or None if insufficient data.
    """
    clean = _clean_polar_map(thk_map)
    n_rows, n_cols = clean.shape

    # Reconstruct normative mean and SD maps
    norm_mean = _reconstruct_normative_map(_NORM_THK_MEAN, n_rows, n_cols)
    norm_sd = _reconstruct_normative_map(_NORM_THK_SD, n_rows, n_cols)

    # For OS eyes, mirror the normative map: col -> (3*N/2 - col) % N
    # Mirror formula: (3*N/2 - col) % N
    is_os = str(exam_eye).strip().upper() == "OS"
    if is_os:
        mirror_idx = np.array([(3 * n_cols // 2 - j) % n_cols for j in range(n_cols)])
        norm_mean = norm_mean[:, mirror_idx]
        norm_sd = norm_sd[:, mirror_idx]

    # Z-score the patient's thickness map
    # z_map = (thk - norm_mean) / norm_sd
    # Guard against division by zero/NaN in normative SD
    z_map = np.full_like(clean, np.nan)
    ok_data = ~np.isnan(clean) & ~np.isnan(norm_mean) & ~np.isnan(norm_sd)
    valid = ok_data & (np.abs(norm_sd) > 1e-6)
    z_map[valid] = (clean[valid] - norm_mean[valid]) / norm_sd[valid]

    # Now run the same SI logic as GetThkSIn on the Z-scored map.
    # GetThkSIn (line 29170) uses the same laterality-specific P0 angle
    # as GetCurvSI, with CSO's swapped sin/cos convention.
    angle = _get_p0_angle(exam_eye)
    p0_x, p0_y = _polar_to_cart_cso(r0, angle)

    theta_rad = np.arange(n_cols) * (2 * np.pi / n_cols)

    list_ref = []
    list_opp = []

    for i in range(n_rows):
        r_mm = i * _R_STEP_MM
        for j in range(n_cols):
            if np.isnan(z_map[i, j]):
                continue
            x, y = _polar_to_cart_cso(r_mm, theta_rad[j])
            dist_sq = (x - p0_x) ** 2 + (y - p0_y) ** 2
            if dist_sq < r0 * r0:
                list_ref.append(z_map[i, j])
                j_opp = (j + n_cols // 2) % n_cols
                val_opp = z_map[i, j_opp]
                if not np.isnan(val_opp):
                    list_opp.append(val_opp)

    if len(list_ref) < _SCREENING_MIN_SAMPLES or len(list_opp) < _SCREENING_MIN_SAMPLES:
        return None

    mean_ref = float(np.mean(list_ref))
    mean_opp = float(np.mean(list_opp))

    # GetThkSIn returns mean_opp - mean_ref (opposite convention from SIf)
    return mean_opp - mean_ref


# ==========================================================================
# KERATOCONUS MORPHOLOGICAL CLASSIFIER
# ==========================================================================
#
# Matches CSO KeratoconusMorphologicalClassifier.
#
# Pure rule-based classifier that categorizes keratoconus cone morphology
# into 6 types based on SimK, asphericity, cone center distance, coma
# Zernike coefficients, cylinder, and axis alignment.
# ==========================================================================


def classify_kc_morphology(
    kavg_mm, asphericity_p, kc_center_x, kc_center_y, coma_z7, coma_z8, cyl, cyl_ax
):
    """
    Classify keratoconus cone morphology into 6 types.

    Matches CSO's KeratoconusMorphologicalClassifier algorithm.

    The 6 types are:
      0 = NippleCentral     -- steep + central cone
      1 = NippleParacentral -- steep + off-center cone
      2 = BowTie            -- astigmatism-dominant
      3 = SnowMan           -- axis misalignment > 60 deg
      4 = Croissant         -- axis alignment < 30 deg
      5 = Duck              -- intermediate axis misalignment

    Parameters
    ----------
    kavg_mm : float
        Mean keratometric radius (mm) from SimK.  < 7.03125 mm = > 48D.
    asphericity_p : float
        Shape asphericity at 4.5mm fitting diameter.  CSO convention:
        p < 1 is prolate, p > 1 is oblate.
    kc_center_x, kc_center_y : float
        Keratoconus cone center coordinates (mm) from the notable-points
        barycenter or DZMax location.
    coma_z7, coma_z8 : float
        Zernike coma coefficients Z(3,-1) and Z(3,+1) from the anterior
        altimetric decomposition (screening Zernike, 4mm diameter, um).
    cyl : float
        Keratometric cylinder (D) from SimK.
    cyl_ax : float
        Cylinder axis (degrees, TABO) from SimK.

    Returns
    -------
    dict
        Keys:
          kc_morphology_type : int (-1 to 5)
              -1 = Unknown (inputs invalid)
          kc_morphology_name : str
              One of "NippleCentral", "NippleParacentral", "BowTie",
              "SnowMan", "Croissant", "Duck", or "Unknown".
    """
    try:
        # Validate inputs
        if any(
            math.isnan(v)
            for v in [
                kavg_mm,
                asphericity_p,
                kc_center_x,
                kc_center_y,
                coma_z7,
                coma_z8,
                cyl,
                cyl_ax,
            ]
        ):
            return {"kc_morphology_type": -1, "kc_morphology_name": "Unknown"}

        # Cone center distance from apex
        kc_dist_mm = math.sqrt(kc_center_x**2 + kc_center_y**2)

        # KAx: angle of cone center from origin
        k_ax = (math.degrees(math.atan2(kc_center_y, kc_center_x)) + 360.0) % 360.0

        # Rule 1: Steep + prolate -> Nipple (Central or Paracentral)
        # kavg_mm < 7.03125 corresponds to > 48D
        if kavg_mm < 7.03125 or asphericity_p < -0.25:
            if kc_dist_mm < 1.25:
                return {"kc_morphology_type": 0, "kc_morphology_name": "NippleCentral"}
            else:
                return {"kc_morphology_type": 1, "kc_morphology_name": "NippleParacentral"}

        # Rule 2: BowTie -- coma/cylinder ratio < 1
        coma_mag = math.sqrt(coma_z7**2 + coma_z8**2)
        if abs(cyl) > 1e-12 and coma_mag / abs(cyl) < 1.0:
            return {"kc_morphology_type": 2, "kc_morphology_name": "BowTie"}

        # Rule 3: Croissant / SnowMan / Duck based on axis alignment
        # angle_diff = min(|cylAx - KAx|, |cylAx - KAx + 180|)
        # (matches CSO axis alignment rule)
        angle_diff = min(abs(cyl_ax - k_ax), abs(cyl_ax - k_ax + 180.0))
        if angle_diff < 30.0:
            return {"kc_morphology_type": 4, "kc_morphology_name": "Croissant"}
        elif angle_diff > 60.0:
            return {"kc_morphology_type": 3, "kc_morphology_name": "SnowMan"}
        else:
            return {"kc_morphology_type": 5, "kc_morphology_name": "Duck"}

    except (ValueError, TypeError, ZeroDivisionError):
        return {"kc_morphology_type": -1, "kc_morphology_name": "Unknown"}


def compute_kc_classification(
    raw_segments,
    metadata,
    zernike_results=None,
    k_readings=None,
    shape_results=None,
    screening_extrema=None,
):
    """
    Compute keratoconus morphological classification from available indices.

    Wires together the inputs needed for ``classify_kc_morphology`` from
    the outputs of other compute_* functions.

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of ``core.parse_csv()``.
    metadata : dict
        Must contain ``Exam_Eye``.
    zernike_results : dict or None
        Output of ``compute_zernike_indices()``.
    k_readings : dict or None
        Output of ``compute_k_readings()``.
    shape_results : dict or None
        Output of ``compute_shape_indices()``.
    screening_extrema : dict or None
        Output of ``compute_screening_extrema()``.

    Returns
    -------
    dict
        Keys: ``kc_morphology_type`` (int), ``kc_morphology_name`` (str).
    """
    unknown = {"kc_morphology_type": -1, "kc_morphology_name": "Unknown"}

    # Auto-compute missing prerequisites from raw data
    if k_readings is None:
        k_readings = compute_k_readings(raw_segments, metadata)
    if shape_results is None:
        shape_results = compute_shape_indices(raw_segments)
    if zernike_results is None:
        zernike_results = compute_zernike_indices(raw_segments, metadata)
    if screening_extrema is None:
        screening_extrema = compute_screening_extrema(
            raw_segments, metadata, zernike_results=zernike_results
        )

    # Extract SimK KAvg (mm) from K-readings
    kavg_mm = None
    if k_readings is not None:
        kavg_mm = k_readings.get("simk_kavg")
    if kavg_mm is None:
        return unknown

    # Extract asphericity at 4.5mm anterior from shape results
    asphericity_p = None
    if shape_results is not None:
        asphericity_p = shape_results.get("shape_4p5mm_ant_asphericity")
    if asphericity_p is None:
        return unknown

    # Extract cone center from screening extrema (use DZMaxF location as
    # the cone center, matching CSO's KCcenter_ parameter)
    kc_x, kc_y = 0.0, 0.0
    if screening_extrema is not None:
        dx = screening_extrema.get("screening_dzmaxf_x")
        dy = screening_extrema.get("screening_dzmaxf_y")
        if dx is not None and dy is not None:
            kc_x, kc_y = dx, dy
        else:
            # Fallback to notable-points barycenter
            bx = screening_extrema.get("screening_notable_pts_bary_x")
            by = screening_extrema.get("screening_notable_pts_bary_y")
            if bx is not None and by is not None:
                kc_x, kc_y = bx, by
            else:
                return unknown

    # Extract coma Z7, Z8 from screening Zernike (anterior, 4mm)
    coma_z7, coma_z8 = None, None
    if zernike_results is not None:
        coma_z7 = zernike_results.get("zernike_ant_4mm_z7_um")
        coma_z8 = zernike_results.get("zernike_ant_4mm_z8_um")
    if coma_z7 is None or coma_z8 is None:
        return unknown

    # Extract cylinder and axis from SimK
    cyl = k_readings.get("simk_cyl")
    cyl_ax = k_readings.get("simk_ax_f")
    if cyl is None or cyl_ax is None:
        return unknown

    return classify_kc_morphology(
        kavg_mm=kavg_mm,
        asphericity_p=asphericity_p,
        kc_center_x=kc_x,
        kc_center_y=kc_y,
        coma_z7=coma_z7,
        coma_z8=coma_z8,
        cyl=cyl,
        cyl_ax=cyl_ax,
    )


def compute_screening_indices(raw_segments: dict, metadata: dict) -> dict:
    """
    Compute keratoconus screening indices from polar maps.

    These are the indices the MS-39 computes for its built-in keratoconus
    screening ANN classifier.  Only indices computable from polar-map data
    are returned; Zernike-dependent indices return None.

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of core.parse_csv(). Keys are segment names, values are
        polar matrices with shape (n_rows, 256).  Must include
        ``gaussian_anterior`` and ``gaussian_posterior`` for the
        curvature symmetry indices (SIf/SIb/CSIf/CSIb).
    metadata : dict
        Must contain 'Exam_Eye' key with value 'OD' or 'OS'.

    Returns
    -------
    dict
        Screening index values.  Keys:
          screening_sif, screening_sib       -- curvature symmetry (D)
          screening_csif, screening_csib     -- center-surround (D)
          screening_thksi                    -- thickness symmetry (um)
          screening_pti                      -- pachymetric thickness index
          screening_pepiti                   -- epithelial pachymetric index
          screening_fb_likeness              -- None (requires Zernike)
          screening_rmsf, screening_rmsb     -- None (requires Zernike)
          screening_eif, screening_eib       -- None (requires Zernike)
          screening_pdthksi                  -- None (requires Zernike + norms)
    """
    exam_eye = metadata.get("Exam_Eye", "")
    if exam_eye not in ("OD", "OS"):
        exam_eye = "OD"  # default, matching CSO convention

    # SIf/SIb and CSIf/CSIb use the **Gaussian** curvature map (geometric
    # mean of sagittal and tangential radii), NOT the sagittal map.
    # Empirically verified against MS-39 reference values for both OD and OS.
    gauss_ant = raw_segments.get("gaussian_anterior")
    gauss_post = raw_segments.get("gaussian_posterior")
    thk_map = raw_segments.get("corneal_thickness")
    epi_map = raw_segments.get("epithelial_thickness")

    out = {}

    # --- Curvature symmetry indices (SIf, SIb) ---
    if gauss_ant is not None:
        out["screening_sif"] = _compute_curv_si(gauss_ant, _DELTA_N_ANTERIOR, exam_eye, r0=1.5)
    else:
        out["screening_sif"] = None

    if gauss_post is not None:
        out["screening_sib"] = _compute_curv_si(gauss_post, _DELTA_N_POSTERIOR, exam_eye, r0=1.5)
    else:
        out["screening_sib"] = None

    # --- Center-surround indices (CSIf, CSIb) ---
    if gauss_ant is not None:
        out["screening_csif"] = _compute_curv_csi(gauss_ant, _DELTA_N_ANTERIOR)
    else:
        out["screening_csif"] = None

    if gauss_post is not None:
        out["screening_csib"] = _compute_curv_csi(gauss_post, _DELTA_N_POSTERIOR)
    else:
        out["screening_csib"] = None

    # --- Thickness symmetry index (ThkSI) ---
    if thk_map is not None:
        out["screening_thksi"] = _compute_thksi(thk_map)
    else:
        out["screening_thksi"] = None

    # --- Pachymetric Thickness Index (PTI) ---
    if thk_map is not None:
        thk_min_val, thk_min_x, thk_min_y = _find_extremum(thk_map, mode="min")
        if thk_min_val is not None:
            # Population-adaptive normatives: select based on CCT
            cct = thk_min_val  # ThkMin as CCT proxy
            normatives = _select_pti_normatives(cct)
            out["screening_pti"] = _compute_pti(thk_map, thk_min_x, thk_min_y, normatives)
        else:
            out["screening_pti"] = None
    else:
        out["screening_pti"] = None

    # --- Pachymetric Epithelial Thickness Index (PEpiTI) ---
    if epi_map is not None:
        # ThkMin location serves as proxy for NotablePtsBarycenter
        if thk_map is not None:
            _, tx, ty = _find_extremum(thk_map, mode="min")
        else:
            tx, ty = 0.0, 0.0
        out["screening_pepiti"] = _compute_pepiti(epi_map, tx, ty)
    else:
        out["screening_pepiti"] = None

    # --- Zernike-dependent indices (NOT computable from individual CSV) ---
    # These require altimetric elevation maps decomposed into Zernike
    # polynomials, which are not available in the individual CSV export.
    out["screening_rmsf"] = None  # HOA RMS front (requires Zernike)
    out["screening_rmsb"] = None  # HOA RMS back (requires Zernike)
    out["screening_eif"] = None  # Ectasia index front (requires Zernike)
    out["screening_eib"] = None  # Ectasia index back (requires Zernike)
    out["screening_fb_likeness"] = None  # Front-back likeness (req. Zernike)
    out["screening_pdthksi"] = None  # Pop-deviation ThkSI (req. Zernike norms)

    return out


# ---------------------------------------------------------------------------
# SCREENING EXTREMA + NOTABLE POINTS  (~29 columns)
# ---------------------------------------------------------------------------
#
# Completes the ~38 columns of the MS-39 "Keratoconus Screening" export
# that are NOT computed indices (SIf..PEpiTI) but rather direct
# measurements / extremum searches / derived locations.
#
# Source: formulas_screening_indices.md Section 14.
# ---------------------------------------------------------------------------

# Typical keratoconus apex location (mm from corneal vertex)
_KC_APEX_OD = (-0.475, -1.04)  # temporal & inferior for OD
_KC_APEX_OS = (+0.475, -1.04)  # temporal & inferior for OS


def _reconstruct_zernike_hoa_on_grid(coeffs_um, fitting_radius, n_rows, n_cols):
    """
    Reconstruct the HOA-only (orders 3+) Zernike altimetric surface on
    a polar grid with the given dimensions.

    Returns a polar map (n_rows, n_cols) in micrometers, or None.
    Points outside the fitting zone or at row 0 are set to NaN.
    """
    if coeffs_um is None or len(coeffs_um) < 7:
        return None

    modes = _zernike_generate_modes(_ZERNIKE_MAX_ORDER)

    # Pre-filter to HOA modes only (j >= 6) with available coefficients
    hoa_modes = [(j, n, m) for (j, n, m) in modes if j >= 6 and j < len(coeffs_um)]
    if not hoa_modes:
        return None

    # Pre-extract HOA coefficient values
    hoa_coeffs = np.array([coeffs_um[j] for j, _n, _m in hoa_modes])

    out_map = np.full((n_rows, n_cols), np.nan)

    # Build evaluation grid: vectorized per ring for speed
    theta_grid = np.arange(n_cols) * (2 * np.pi / n_cols)

    for i in range(1, n_rows):
        r_mm = i * _R_STEP_MM
        rho = r_mm / fitting_radius
        if rho > 1.0:
            break

        # Evaluate all HOA Zernike terms at this ring (vectorized over theta)
        rho_arr = np.full(n_cols, rho)
        ring_vals = np.zeros(n_cols)
        for k, (_j, n, m) in enumerate(hoa_modes):
            ring_vals += hoa_coeffs[k] * _zernike_polynomial(n, m, rho_arr, theta_grid)
        out_map[i, :n_cols] = ring_vals

    return out_map


def _ei2_weight(x, y, exam_eye):
    """
    Location-dependent weight for EI2 (ectasia index v2).

    weight = min(1, 3 * exp(-dist(P, apex_KC)))

    Source: formulas_screening_indices.md Section 14.7.
    """
    eye = str(exam_eye).strip().upper()
    apex = _KC_APEX_OD if eye == "OD" else _KC_APEX_OS
    d = math.sqrt((x - apex[0]) ** 2 + (y - apex[1]) ** 2)
    return min(1.0, 3.0 * math.exp(-d))


def _ectasia_axis(x, y):
    """
    Axis diametrically opposite to the (x, y) point.

    EIAx = (atan2(-y, -x) * 180/pi + 360 + 180) mod 360

    Source: formulas_screening_indices.md Section 14.8.
    """
    return (math.degrees(math.atan2(-y, -x)) + 360 + 180) % 360


def _extract_coeffs_from_zernike_results(zernike_results, surface):
    """
    Extract the Zernike coefficient array from compute_zernike_indices()
    output for a given surface ('ant' or 'post').

    Returns np.ndarray of shape (36,) in micrometers, or None.
    """
    coeffs = []
    for j in range(_ZERNIKE_SCREENING_N_TERMS):
        key = f"zernike_{surface}_4mm_z{j}_um"
        val = zernike_results.get(key)
        if val is None:
            return None
        coeffs.append(val)
    return np.array(coeffs, dtype=np.float64)


def compute_screening_extrema(
    raw_segments: dict, metadata: dict, zernike_results: dict | None = None
) -> dict:
    """
    Compute the ~29 direct-measurement / extremum screening columns.

    These complete the MS-39 "Keratoconus Screening" export alongside
    the computed indices from ``compute_screening_indices()`` and the
    Zernike indices from ``compute_zernike_indices()``.

    Columns produced (29 keys):
      screening_thkmin_{value,x,y}          -- min corneal thickness
      screening_strthkmin_{value,x,y}       -- min stromal thickness
      screening_epithkmin_{value,x,y}       -- min epithelial thickness
      screening_kmaxf_{value,x,y}           -- steepest anterior curvature
      screening_kmaxb_{value,x,y}           -- steepest posterior curvature
      screening_dzmaxf_{value,x,y}          -- max Zernike HOA elevation front
      screening_dzmaxb_{value,x,y}          -- max Zernike HOA elevation back
      screening_ei2f, screening_ei2b        -- location-weighted EI v2
      screening_eifax, screening_eibax      -- ectasia axis (degrees)
      screening_pdthksi                     -- None (requires normative SD)
      screening_notable_pts_bary_{x,y}      -- barycenter of 5 notable pts
      screening_notable_pts_r               -- mean distance from barycenter

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of ``core.parse_csv()``.  Relevant keys:
        ``corneal_thickness``, ``stromal_thickness``,
        ``epithelial_thickness``, ``sagittal_anterior``,
        ``sagittal_posterior``, ``elevation_anterior``,
        ``elevation_posterior``.
    metadata : dict
        Must contain ``Exam_Eye`` with value ``"OD"`` or ``"OS"``.
    zernike_results : dict or None
        Output of ``compute_zernike_indices()``.  If provided, used to
        compute DZMaxF/B, EI2f/b, EIfAx/EIbAx, and the full
        NotablePtsBarycenter.  If None, those columns return None.

    Returns
    -------
    dict
        ~29 keys with ``screening_`` prefix.
    """
    out = {}
    exam_eye = metadata.get("Exam_Eye", "")
    if exam_eye not in ("OD", "OS"):
        exam_eye = "OD"  # default, matching CSO convention

    # ------------------------------------------------------------------
    # 1. ThkMin (3 cols) -- duplicate of summary, screening prefix
    # ------------------------------------------------------------------
    thk_map = raw_segments.get("corneal_thickness")
    if thk_map is not None:
        val, x, y = _find_extremum(thk_map, mode="min")
        out["screening_thkmin_value"] = val
        out["screening_thkmin_x"] = x
        out["screening_thkmin_y"] = y
    else:
        val, x, y = None, 0.0, 0.0
        out["screening_thkmin_value"] = None
        out["screening_thkmin_x"] = None
        out["screening_thkmin_y"] = None
    thkmin_loc = (x, y)

    # ------------------------------------------------------------------
    # 2. StrThkMin (3 cols) -- minimum stromal thickness
    #    CSO uses Utils.GetMin with the HolesEpiThickness mask and
    #    a local-minimum search (4-neighbor + 5x5 hole-free check).
    #    CSO uses Utils.GetMin with stromal mask.
    #    The naive global argmin picks up peripheral OCT artifacts
    #    (negative values at r > 4mm) which CSO correctly rejects.
    # ------------------------------------------------------------------
    str_map = raw_segments.get("stromal_thickness")
    if str_map is not None:
        val, x, y = _find_extremum(str_map, mode="min")
        out["screening_strthkmin_value"] = val
        out["screening_strthkmin_x"] = x
        out["screening_strthkmin_y"] = y
    else:
        out["screening_strthkmin_value"] = None
        out["screening_strthkmin_x"] = None
        out["screening_strthkmin_y"] = None

    # ------------------------------------------------------------------
    # 3. EpiThkMin (3 cols) -- duplicate of summary, screening prefix
    #    CSO applies 3.9mm radius cutoff + internal OCT validity mask.
    # ------------------------------------------------------------------
    epi_map = raw_segments.get("epithelial_thickness")
    if epi_map is not None:
        val, x, y = _find_extremum(epi_map, mode="min", max_radius=3.9)
        out["screening_epithkmin_value"] = val
        out["screening_epithkmin_x"] = x
        out["screening_epithkmin_y"] = y
    else:
        out["screening_epithkmin_value"] = None
        out["screening_epithkmin_x"] = None
        out["screening_epithkmin_y"] = None

    # ------------------------------------------------------------------
    # 4. KMaxF (3 cols) -- steepest anterior curvature
    #    CSO uses Gaussian curvature map, NOT sagittal.
    # ------------------------------------------------------------------
    gauss_ant = raw_segments.get("gaussian_anterior")
    if gauss_ant is not None:
        val, x, y = _find_extremum(gauss_ant, mode="min")
        out["screening_kmaxf_value"] = val
        out["screening_kmaxf_x"] = x
        out["screening_kmaxf_y"] = y
    else:
        val, x, y = None, 0.0, 0.0
        out["screening_kmaxf_value"] = None
        out["screening_kmaxf_x"] = None
        out["screening_kmaxf_y"] = None
    kmaxf_loc = (x, y)

    # ------------------------------------------------------------------
    # 5. KMaxB (3 cols) -- steepest posterior curvature
    #    CSO uses GetMin(radii, gaussPost, mask) with no explicit Limit.
    #    CheckPoint uses MaxR=4.0 (hardcoded).
    # ------------------------------------------------------------------
    gauss_post = raw_segments.get("gaussian_posterior")
    if gauss_post is not None:
        val, x, y = _find_extremum(gauss_post, mode="min")
        out["screening_kmaxb_value"] = val
        out["screening_kmaxb_x"] = x
        out["screening_kmaxb_y"] = y
    else:
        val, x, y = None, 0.0, 0.0
        out["screening_kmaxb_value"] = None
        out["screening_kmaxb_x"] = None
        out["screening_kmaxb_y"] = None
    kmaxb_loc = (x, y)

    # ------------------------------------------------------------------
    # 6. DZMaxF (3 cols) -- max HOA Zernike elevation, anterior
    # ------------------------------------------------------------------
    dzmax_front_val = None
    dzmax_front_loc = (0.0, 0.0)

    if zernike_results is not None:
        coeffs_ant = _extract_coeffs_from_zernike_results(zernike_results, "ant")
        if coeffs_ant is not None:
            # Determine grid size from the elevation map if available
            elev_ant = raw_segments.get("elevation_anterior")
            if elev_ant is not None:
                nr, nc = elev_ant.shape
            else:
                nr, nc = 31, 256
            hoa_map = _reconstruct_zernike_hoa_on_grid(
                coeffs_ant,
                _ZERNIKE_FITTING_RADIUS,
                nr,
                nc,
            )
            if hoa_map is not None:
                # CSO uses 4.0mm cutoff for DZMax
                val, x, y = _find_extremum(hoa_map, mode="max", max_radius=4.0)
                if val is not None:
                    dzmax_front_val = val
                    dzmax_front_loc = (x, y)

    out["screening_dzmaxf_value"] = dzmax_front_val
    out["screening_dzmaxf_x"] = dzmax_front_loc[0]
    out["screening_dzmaxf_y"] = dzmax_front_loc[1]

    # ------------------------------------------------------------------
    # 7. DZMaxB (3 cols) -- max HOA Zernike elevation, posterior
    # ------------------------------------------------------------------
    dzmax_back_val = None
    dzmax_back_loc = (0.0, 0.0)

    if zernike_results is not None:
        coeffs_post = _extract_coeffs_from_zernike_results(zernike_results, "post")
        if coeffs_post is not None:
            elev_post = raw_segments.get("elevation_posterior")
            if elev_post is not None:
                nr, nc = elev_post.shape
            else:
                nr, nc = 31, 256
            hoa_map = _reconstruct_zernike_hoa_on_grid(
                coeffs_post,
                _ZERNIKE_FITTING_RADIUS,
                nr,
                nc,
            )
            if hoa_map is not None:
                # CSO uses 4.0mm cutoff for DZMax
                val, x, y = _find_extremum(hoa_map, mode="max", max_radius=4.0)
                if val is not None:
                    dzmax_back_val = val
                    dzmax_back_loc = (x, y)

    out["screening_dzmaxb_value"] = dzmax_back_val
    out["screening_dzmaxb_x"] = dzmax_back_loc[0]
    out["screening_dzmaxb_y"] = dzmax_back_loc[1]

    # ------------------------------------------------------------------
    # 8. EI2f / EI2b (2 cols) -- location-weighted max elevation
    # ------------------------------------------------------------------
    if dzmax_front_val is not None:
        w = _ei2_weight(dzmax_front_loc[0], dzmax_front_loc[1], exam_eye)
        out["screening_ei2f"] = dzmax_front_val * w
    else:
        out["screening_ei2f"] = None

    if dzmax_back_val is not None:
        w = _ei2_weight(dzmax_back_loc[0], dzmax_back_loc[1], exam_eye)
        out["screening_ei2b"] = dzmax_back_val * w
    else:
        out["screening_ei2b"] = None

    # ------------------------------------------------------------------
    # 9. EIfAx / EIbAx (2 cols) -- ectasia axis
    # ------------------------------------------------------------------
    if dzmax_front_val is not None and dzmax_front_val > 0:
        out["screening_eifax"] = _ectasia_axis(dzmax_front_loc[0], dzmax_front_loc[1])
    else:
        out["screening_eifax"] = None

    if dzmax_back_val is not None and dzmax_back_val > 0:
        out["screening_eibax"] = _ectasia_axis(dzmax_back_loc[0], dzmax_back_loc[1])
    else:
        out["screening_eibax"] = None

    # ------------------------------------------------------------------
    # 10. PDThkSI (1 col) -- Population-Deviation Thickness Symmetry Index
    #
    # Algorithm (matches CSO KeratoconusScreening):
    #   1. Reconstruct normative mean (ThkM) and SD (ThkSD) thickness
    #      maps from 36 Zernike coefficients on the polar grid.
    #   2. Z-score the patient's thickness map: (thk - ThkM_norm) / ThkSD_norm
    #   3. Run GetThkSIn (same logic as ThkSI) on the Z-scored map
    #      with r0=2.0mm (vs 1.5mm for ThkSI), using laterality-specific
    #      P0 reference angle.
    #   For OS eyes, the normative map is mirrored: col -> (3*N/2 - col) % N
    # ------------------------------------------------------------------
    thk_map_for_pd = raw_segments.get("corneal_thickness")
    if thk_map_for_pd is not None:
        out["screening_pdthksi"] = _compute_pdthksi(thk_map_for_pd, exam_eye)
    else:
        out["screening_pdthksi"] = None

    # ------------------------------------------------------------------
    # 11. NotablePtsBarycenter + NotablePtsR (3 cols)
    # ------------------------------------------------------------------
    # The 5 notable points: ThkMin, KMaxF, KMaxB, DZMaxF, DZMaxB
    notable_points = [
        thkmin_loc,
        kmaxf_loc,
        kmaxb_loc,
        dzmax_front_loc,
        dzmax_back_loc,
    ]

    # Only compute if we have real values for all 5 notable points
    have_all = (
        out["screening_thkmin_value"] is not None
        and out["screening_kmaxf_value"] is not None
        and out["screening_kmaxb_value"] is not None
        and dzmax_front_val is not None
        and dzmax_back_val is not None
    )

    if have_all:
        bary_x = sum(p[0] for p in notable_points) / 5.0
        bary_y = sum(p[1] for p in notable_points) / 5.0
        out["screening_notable_pts_bary_x"] = bary_x
        out["screening_notable_pts_bary_y"] = bary_y

        # NotablePtsR = mean distance from barycenter to each notable point
        total_dist = 0.0
        for p in notable_points:
            total_dist += math.sqrt((p[0] - bary_x) ** 2 + (p[1] - bary_y) ** 2)
        out["screening_notable_pts_r"] = total_dist / 5.0
    else:
        out["screening_notable_pts_bary_x"] = None
        out["screening_notable_pts_bary_y"] = None
        out["screening_notable_pts_r"] = None

    return out


# ===================================================================
# K-READINGS  (~104 columns)
# ===================================================================
#
# Matches CSO K-readings algorithm (KReadingsObj):
#   - DetectAxis: discrete cylinder sweep across quarter-circle pairs
#   - SimK (Scheimpflug path): band average in [1.25, 2.05] mm
#   - Meridians 3/5/7 mm: cumulative zones (ro <= 1.5/2.5/3.5)
#   - Hemimeridians 3/5/7 mm: Gaussian smooth + local min/max split
#
# Source: formulas_k_readings.md
# ===================================================================

# Dioptric conversion constants
_DN_ANTERIOR = 337.5  # 1000 * (1.3375 - 1.0)  keratometric index
_DN_POSTERIOR = -40.00008  # 1000 * (1.336 - 1.376)

# Spherical fallback threshold (diopters)
_CYL_THRESHOLD = 0.25


def _detect_axis(profile, counts, threshold):
    """
    Discrete cylinder-detection sweep (matches CSO DetectAxis).

    Sweeps meridians 0..N/4-1.  For each pair of orthogonal full
    meridians, computes the dioptric cylinder.

    Parameters
    ----------
    profile : np.ndarray, shape (N,)
        Mean sagittal radius (mm) at each of 256 meridians.
    counts : np.ndarray, shape (N,), dtype int
        Valid-sample count per meridian.
    threshold : int
        Minimum combined count for an orthogonal pair.

    Returns
    -------
    max_cyl : float
        Maximum absolute cylinder (D).  0.0 if no valid pair.
    flat_index : int
        Meridian index of the flat axis.
    """
    n = len(profile)
    quarter = n // 4
    half = n // 2

    max_cyl = 0.0
    flat_index = 0

    for i in range(quarter):
        i_opp = i + half
        i_perp = i + quarter
        i_perp_opp = (i + 3 * quarter) % n

        if (
            counts[i] + counts[i_opp] <= threshold
            or counts[i_perp] + counts[i_perp_opp] <= threshold
        ):
            continue

        sum_i = profile[i] + profile[i_opp]
        sum_perp = profile[i_perp] + profile[i_perp_opp]

        if sum_i == 0.0 or sum_perp == 0.0:
            continue

        # 675 = 2 * 337.5
        cyl = 675.0 * (1.0 / sum_i - 1.0 / sum_perp)

        if abs(cyl) > max_cyl:
            max_cyl = abs(cyl)
            flat_index = i_perp if cyl >= 0 else i

    return max_cyl, flat_index


def _extract_kf_ks(profile, flat_index):
    """Extract Kflat/Ksteep (mm) at the detected flat axis."""
    n = len(profile)
    half = n // 2
    quarter = n // 4

    kf = (profile[flat_index] + profile[(flat_index + half) % n]) / 2.0
    ks = (profile[(flat_index + quarter) % n] + profile[(flat_index + 3 * quarter) % n]) / 2.0
    return kf, ks


def _index_to_degrees(index, n):
    """Convert meridian index to degrees (0-360)."""
    return float(index) * 360.0 / float(n)


def _build_meridian_dict(kf_mm, ks_mm, flat_axis_deg, anterior, prefix):
    """Build the 8-column dict for a Meridians module."""
    if abs(kf_mm) < 1e-12 or abs(ks_mm) < 1e-12:
        return {
            f"{prefix}_kf": None,
            f"{prefix}_ks": None,
            f"{prefix}_ax_f": None,
            f"{prefix}_ax_s": None,
            f"{prefix}_kavg": None,
            f"{prefix}_cyl": None,
            f"{prefix}_kj0": None,
            f"{prefix}_kj45": None,
        }
    dn = _DN_ANTERIOR if anterior else _DN_POSTERIOR
    ax_f = flat_axis_deg % 180.0
    ax_s = (ax_f + 90.0) % 180.0
    kavg = (kf_mm + ks_mm) / 2.0

    d_flat = dn / kf_mm
    d_steep = dn / ks_mm
    cyl = d_flat - d_steep  # negative for astigmatic corneas (CSO convention)

    alpha = np.radians(ax_f)
    kj0 = (-cyl / 2.0) * np.cos(2.0 * alpha)
    kj45 = (-cyl / 2.0) * np.sin(2.0 * alpha)

    return {
        f"{prefix}_kf": round(float(kf_mm), 6),
        f"{prefix}_ks": round(float(ks_mm), 6),
        f"{prefix}_ax_f": round(float(ax_f), 1),
        f"{prefix}_ax_s": round(float(ax_s), 1),
        f"{prefix}_kavg": round(float(kavg), 6),
        f"{prefix}_cyl": round(float(cyl), 6),
        f"{prefix}_kj0": round(float(kj0), 6),
        f"{prefix}_kj45": round(float(kj45), 6),
    }


def _build_hemi_dict(kf1, ks1, axf1, axs1, kf2, ks2, axf2, axs2, prefix):
    """Build the 8-column dict for a Hemimeridians module."""
    return {
        f"{prefix}_kf1": round(float(kf1), 6),
        f"{prefix}_axf1": round(float(axf1), 1),
        f"{prefix}_ks1": round(float(ks1), 6),
        f"{prefix}_axs1": round(float(axs1), 1),
        f"{prefix}_kf2": round(float(kf2), 6),
        f"{prefix}_axf2": round(float(axf2), 1),
        f"{prefix}_ks2": round(float(ks2), 6),
        f"{prefix}_axs2": round(float(axs2), 1),
    }


def _nan_meridian_dict(prefix):
    """Return 8 meridian keys all set to None."""
    return {
        f"{prefix}_kf": None,
        f"{prefix}_ks": None,
        f"{prefix}_ax_f": None,
        f"{prefix}_ax_s": None,
        f"{prefix}_kavg": None,
        f"{prefix}_cyl": None,
        f"{prefix}_kj0": None,
        f"{prefix}_kj45": None,
    }


def _nan_hemi_dict(prefix):
    """Return 8 hemimeridian keys all set to None."""
    return {
        f"{prefix}_kf1": None,
        f"{prefix}_axf1": None,
        f"{prefix}_ks1": None,
        f"{prefix}_axs1": None,
        f"{prefix}_kf2": None,
        f"{prefix}_axf2": None,
        f"{prefix}_ks2": None,
        f"{prefix}_axs2": None,
    }


def _build_zone_profiles(sagittal_map, r_max, anterior, exclude_inner=True):
    """
    Build a mean curvature profile over a cumulative radial zone.

    Averages all valid samples with ro <= r_max for each meridian.
    When ``exclude_inner`` is True (default), the posterior surface
    excludes ro <= 0.5 mm.  CSO applies this exclusion for meridians
    but NOT for hemimeridians on the posterior surface.

    Returns (profile, counts, global_mean, total_count).
    """
    n_rows, n_cols = sagittal_map.shape
    profile = np.zeros(n_cols, dtype=np.float64)
    counts = np.zeros(n_cols, dtype=np.int64)
    global_sum = 0.0
    total_count = 0

    for i in range(1, n_rows):  # skip ring 0
        r = i * _R_STEP_MM
        if r > r_max:
            break
        if exclude_inner and not anterior and r <= 0.5:
            continue
        for j in range(n_cols):
            val = sagittal_map[i, j]
            if np.isnan(val):
                continue
            profile[j] += val
            counts[j] += 1
            global_sum += val
            total_count += 1

    valid = counts > 0
    profile[valid] = profile[valid] / counts[valid]
    profile[~valid] = 0.0  # temporary, will be interpolated below
    profile = _k_fill_holes(profile, counts)

    global_mean = global_sum / total_count if total_count > 0 else np.nan
    return profile, counts, global_mean, total_count


def _k_gaussian_kernel(sigma):
    """Build a 1D Gaussian kernel (width = 6*sigma)."""
    hw = int(np.ceil(3.0 * sigma))
    x = np.arange(-hw, hw + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _k_circular_convolve(profile, kernel):
    """Circular convolution of a 1D profile with a kernel."""
    n = len(profile)
    hw = len(kernel) // 2
    padded = np.concatenate([profile[-hw:], profile, profile[:hw]])
    result = np.convolve(padded, kernel, mode="valid")
    return result[:n]


def _k_fill_holes(profile, counts):
    """Fill missing meridians (count==0) via nearest-neighbor."""
    out = profile.copy()
    n = len(profile)
    holes = counts == 0
    if not np.any(holes) or np.all(holes):
        return out
    valid_idx = np.where(~holes)[0]
    for j in np.where(holes)[0]:
        dists = np.minimum(
            np.abs(valid_idx - j),
            n - np.abs(valid_idx - j),
        )
        out[j] = profile[valid_idx[np.argmin(dists)]]
    return out


def _compute_hemimeridians(profile, counts, prefix):
    """
    Compute hemimeridians via Gaussian smooth + local min/max search.

    Algorithm (formulas_k_readings.md Section 6):
      1. Fill holes, Gaussian-smooth (sigma=4)
      2. Find global max (flattest) and min (steepest)
      3. Find local minima
      4. Split into two topology-driven halves
    """
    n = len(profile)

    if np.sum(counts) == 0:
        return _nan_hemi_dict(prefix)

    # Step 1: fill holes and smooth
    filled = _k_fill_holes(profile, counts)
    kernel = _k_gaussian_kernel(4.0)
    smoothed = _k_circular_convolve(filled, kernel)

    # Step 2: global extrema
    idx_max = int(np.argmax(smoothed))  # flattest
    idx_min = int(np.argmin(smoothed))  # steepest
    val_max = smoothed[idx_max]
    val_min = smoothed[idx_min]

    # Step 3: local minima
    local_minima = []
    for k in range(n):
        pk = (k - 1) % n
        nk = (k + 1) % n
        if smoothed[k] < smoothed[pk] and smoothed[k] < smoothed[nk]:
            local_minima.append(k)

    # Step 4: split
    if len(local_minima) <= 1:
        axf = _index_to_degrees(idx_max, n)
        axs = _index_to_degrees(idx_min, n)
        return _build_hemi_dict(
            val_max,
            val_min,
            axf,
            axs,
            val_max,
            val_min,
            axf,
            axs,
            prefix,
        )

    # Find local minimum farthest from global minimum.
    # CSO uses clockwise distance: (item - globalMinIdx + N) % N
    best_split = local_minima[0]
    best_dist = -1
    for lm in local_minima:
        d = (lm - idx_min + n) % n
        if d > best_dist:
            best_dist = d
            best_split = lm

    def _arc_indices(start, end):
        if start <= end:
            return list(range(start, end + 1))
        return list(range(start, n)) + list(range(0, end + 1))

    arc_a = set(_arc_indices(idx_min, best_split))
    arc_b = set(_arc_indices(best_split, idx_min))

    # Half 1: global extrema
    h1_flat_idx, h1_flat_val = idx_max, val_max
    h1_steep_idx, h1_steep_val = idx_min, val_min

    # Half 2: secondary max in the opposite arc
    search_arc = sorted(arc_b if idx_max in arc_a else arc_a)
    if len(search_arc) > 0:
        arc_vals = smoothed[search_arc]
        h2_flat_idx = search_arc[int(np.argmax(arc_vals))]
        h2_flat_val = smoothed[h2_flat_idx]
    else:
        h2_flat_idx, h2_flat_val = idx_max, val_max

    h2_steep_idx = best_split
    h2_steep_val = smoothed[best_split]

    return _build_hemi_dict(
        h1_flat_val,
        h1_steep_val,
        _index_to_degrees(h1_flat_idx, n),
        _index_to_degrees(h1_steep_idx, n),
        h2_flat_val,
        h2_steep_val,
        _index_to_degrees(h2_flat_idx, n),
        _index_to_degrees(h2_steep_idx, n),
        prefix,
    )


_FOURIER_SIMK_KEYS = [
    "simk_fourier_kf",
    "simk_fourier_ks",
    "simk_fourier_ax_f",
    "simk_fourier_ax_s",
    "simk_fourier_kavg",
    "simk_fourier_cyl",
    "simk_fourier_kj0",
    "simk_fourier_kj45",
]


def _simk_fourier(profile, counts, anterior=True):
    """
    Fourier frequency-2 SimK: extract regular astigmatism via
    cos(2theta)/sin(2theta) decomposition of the angular curvature profile.

    Unlike DetectAxis (which finds max cylinder from 4-point samples),
    Fourier decomposition uses ALL 256 data points simultaneously and
    orthogonally rejects higher-frequency irregularities.

    Parameters
    ----------
    profile : np.ndarray, shape (N,)
        Mean sagittal radius (mm) per meridian (N=256).
    counts : np.ndarray, shape (N,)
        Valid-sample count per meridian.
    anterior : bool
        True for anterior surface (dn=337.5), False for posterior.

    Returns
    -------
    dict with keys: simk_fourier_{kf,ks,ax_f,ax_s,kavg,cyl,kj0,kj45}
        or None if insufficient data.
    """
    n = len(profile)
    # Only use meridians with valid data
    valid = counts > 0
    if np.sum(valid) < n // 2:
        return None

    # Use filled profile for Fourier (fill holes same as hemimeridians)
    filled = _k_fill_holes(profile, counts)

    theta = 2.0 * np.pi * np.arange(n) / n

    r_mean = float(np.mean(filled))
    C2 = (2.0 / n) * float(np.sum(filled * np.cos(2.0 * theta)))
    S2 = (2.0 / n) * float(np.sum(filled * np.sin(2.0 * theta)))

    amplitude = np.sqrt(C2**2 + S2**2)
    axis_flat_rad = 0.5 * np.arctan2(S2, C2)

    kf_mm = r_mean + amplitude  # larger radius = flatter
    ks_mm = r_mean - amplitude  # smaller radius = steeper

    ax_f = np.degrees(axis_flat_rad) % 180.0
    ax_s = (ax_f + 90.0) % 180.0

    dn = _DN_ANTERIOR if anterior else _DN_POSTERIOR
    d_flat = dn / kf_mm
    d_steep = dn / ks_mm
    cyl = d_flat - d_steep  # negative convention

    alpha = np.radians(ax_f)
    kj0 = (-cyl / 2.0) * np.cos(2.0 * alpha)
    kj45 = (-cyl / 2.0) * np.sin(2.0 * alpha)

    return {
        "simk_fourier_kf": round(float(kf_mm), 6),
        "simk_fourier_ks": round(float(ks_mm), 6),
        "simk_fourier_ax_f": round(float(ax_f), 1),
        "simk_fourier_ax_s": round(float(ax_s), 1),
        "simk_fourier_kavg": round(float(r_mean), 6),
        "simk_fourier_cyl": round(float(cyl), 6),
        "simk_fourier_kj0": round(float(kj0), 6),
        "simk_fourier_kj45": round(float(kj45), 6),
    }


def compute_k_readings(raw_segments, metadata=None):
    """
    Compute ~104 K-reading columns from the sagittal curvature polar maps.

    Matches CSO's K-readings algorithm (KReadingsObj):
      - SimK (anterior only): Scheimpflug path, radial band [1.25, 2.05] mm
      - Meridians at 3/5/7 mm: cumulative zones (ro <= 1.5/2.5/3.5)
      - Hemimeridians at 3/5/7 mm: Gaussian smooth + local min/max split

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of core.parse_csv().  Must contain ``sagittal_anterior``
        and/or ``sagittal_posterior`` with shape (n_rows, 256).
    metadata : dict, optional
        Parsed metadata.  Reserved for future cross-validation against
        CSO SimK values (SimKKf, SimKAxf, etc.).

    Returns
    -------
    dict
        Up to 114 named features:
          simk_{kf,ks,ax_f,ax_s,kavg,cyl,kj0,kj45}                (8)
          simk_fourier_{kf,ks,ax_f,ax_s,kavg,cyl,kj0,kj45}       (8)
          simk_irregularity                                        (1)
          meridian_{3,5,7}mm_{ant,post}_{kf,ks,ax_f,...}           (48)
          hemi_{3,5,7}mm_{ant,post}_{kf1,axf1,ks1,...}            (48)
          gullstrand_ratio                                         (1)
    """
    out = {}
    sag_ant = raw_segments.get("sagittal_anterior")
    sag_post = raw_segments.get("sagittal_posterior")

    # (label, r_max_mm, min_total_multiplier, detect_threshold)
    zones = [
        ("3mm", 1.5, 7, 8),
        ("5mm", 2.5, 12, 12),
        ("7mm", 3.5, 17, 18),
    ]

    for surface_tag, sag_map, anterior in [
        ("ant", sag_ant, True),
        ("post", sag_post, False),
    ]:
        if sag_map is None:
            if anterior:
                out.update(_nan_meridian_dict("simk"))
                for k in _FOURIER_SIMK_KEYS:
                    out[k] = None
                out["simk_irregularity"] = None
            for label, _, _, _ in zones:
                out.update(
                    _nan_meridian_dict(
                        f"meridian_{label}_{surface_tag}",
                    )
                )
                out.update(
                    _nan_hemi_dict(
                        f"hemi_{label}_{surface_tag}",
                    )
                )
            continue

        clean = sag_map.astype(np.float64)
        clean[clean == _MISSING] = np.nan
        n_rows, n_cols = clean.shape

        # -- SimK (anterior only, Scheimpflug path) --------------------
        if anterior:
            sim_profile = np.zeros(n_cols, dtype=np.float64)
            sim_counts = np.zeros(n_cols, dtype=np.int64)
            sim_sum = 0.0
            sim_total = 0

            for i in range(1, n_rows):
                r = i * _R_STEP_MM
                if r < 1.25 or r > 2.05:
                    continue
                for j in range(n_cols):
                    val = clean[i, j]
                    if np.isnan(val):
                        continue
                    sim_profile[j] += val
                    sim_counts[j] += 1
                    sim_sum += val
                    sim_total += 1

            valid_m = sim_counts > 0
            sim_profile[valid_m] = sim_profile[valid_m] / sim_counts[valid_m]
            sim_profile[~valid_m] = 0.0

            min_samples = 7 * n_cols // 2  # 896 for N=256
            if sim_total < min_samples:
                out.update(_nan_meridian_dict("simk"))
            else:
                sim_mean = sim_sum / sim_total
                max_cyl, flat_idx = _detect_axis(
                    sim_profile,
                    sim_counts,
                    threshold=6,
                )
                if max_cyl < _CYL_THRESHOLD:
                    out.update(
                        _build_meridian_dict(
                            sim_mean,
                            sim_mean,
                            0.0,
                            anterior=True,
                            prefix="simk",
                        )
                    )
                else:
                    kf, ks = _extract_kf_ks(sim_profile, flat_idx)
                    axis = _index_to_degrees(flat_idx, n_cols)
                    out.update(
                        _build_meridian_dict(
                            kf,
                            ks,
                            axis,
                            anterior=True,
                            prefix="simk",
                        )
                    )

            # -- Fourier SimK (supplementary, scientifically standard) --
            if sim_total >= min_samples:
                fourier_result = _simk_fourier(sim_profile, sim_counts, anterior=True)
                if fourier_result is not None:
                    out.update(fourier_result)
                    # Irregularity index: difference between DetectAxis and Fourier cylinder
                    cso_cyl = out.get("simk_cyl")
                    fourier_cyl = fourier_result.get("simk_fourier_cyl")
                    if cso_cyl is not None and fourier_cyl is not None:
                        out["simk_irregularity"] = round(abs(cso_cyl - fourier_cyl), 6)
                    else:
                        out["simk_irregularity"] = None
                else:
                    for k in _FOURIER_SIMK_KEYS:
                        out[k] = None
                    out["simk_irregularity"] = None
            else:
                for k in _FOURIER_SIMK_KEYS:
                    out[k] = None
                out["simk_irregularity"] = None

        # -- SimK passthrough from firmware metadata ---------------------
        # New firmware (2025+) exports Placido-derived SimK directly in
        # the CSV header (SimKKf, SimKAxf, SimKKs, SimKAxs). These are
        # more accurate than our Scheimpflug-derived SimK because they
        # use the Placido rings. When available, override the Scheimpflug
        # values with the firmware values.
        if anterior and metadata is not None:
            simk_kf = metadata.get("SimKKf")
            simk_ks = metadata.get("SimKKs")
            simk_axf = metadata.get("SimKAxf")
            simk_axs = metadata.get("SimKAxs")
            if simk_kf is not None and simk_ks is not None:
                kf = float(simk_kf)
                ks = float(simk_ks)
                axf = float(simk_axf) if simk_axf is not None else 0.0
                axs = float(simk_axs) if simk_axs is not None else 90.0
                dn = _DN_ANTERIOR  # 337.5
                out["simk_kf"] = kf
                out["simk_ks"] = ks
                out["simk_ax_f"] = axf
                out["simk_ax_s"] = axs
                out["simk_kavg"] = (dn / kf + dn / ks) / 2.0
                out["simk_cyl"] = dn / kf - dn / ks
                axf_rad = math.radians(axf)
                out["simk_kj0"] = -(out["simk_cyl"] / 2.0) * math.cos(2.0 * axf_rad)
                out["simk_kj45"] = -(out["simk_cyl"] / 2.0) * math.sin(2.0 * axf_rad)

        # -- Meridians + Hemimeridians at 3/5/7 mm ---------------------
        for label, r_max, min_mult, det_thresh in zones:
            prefix_m = f"meridian_{label}_{surface_tag}"
            prefix_h = f"hemi_{label}_{surface_tag}"

            profile, counts, global_mean, total_count = _build_zone_profiles(clean, r_max, anterior)
            min_samples = min_mult * n_cols // 2

            # Meridians
            if total_count < min_samples:
                out.update(_nan_meridian_dict(prefix_m))
            else:
                max_cyl, flat_idx = _detect_axis(
                    profile,
                    counts,
                    threshold=det_thresh,
                )
                if max_cyl < _CYL_THRESHOLD:
                    out.update(
                        _build_meridian_dict(
                            global_mean,
                            global_mean,
                            0.0,
                            anterior=anterior,
                            prefix=prefix_m,
                        )
                    )
                else:
                    kf, ks = _extract_kf_ks(profile, flat_idx)
                    axis = _index_to_degrees(flat_idx, n_cols)
                    out.update(
                        _build_meridian_dict(
                            kf,
                            ks,
                            axis,
                            anterior=anterior,
                            prefix=prefix_m,
                        )
                    )

            # Hemimeridians
            # CSO does NOT apply the inner-ring exclusion (r <= 0.5mm)
            # for hemimeridians on the posterior surface — build a
            # separate profile without that guard.
            if not anterior:
                hemi_profile, hemi_counts, _, hemi_total = _build_zone_profiles(
                    clean,
                    r_max,
                    anterior,
                    exclude_inner=False,
                )
            else:
                hemi_profile, hemi_counts, hemi_total = profile, counts, total_count

            if hemi_total < min_samples:
                out.update(_nan_hemi_dict(prefix_h))
            else:
                out.update(
                    _compute_hemimeridians(
                        hemi_profile,
                        hemi_counts,
                        prefix_h,
                    )
                )

    # -- Gullstrand ratio (Meridians 3mm anterior / posterior) ---------
    ant_kavg = out.get("meridian_3mm_ant_kavg")
    post_kavg = out.get("meridian_3mm_post_kavg")
    if ant_kavg is not None and post_kavg is not None and post_kavg != 0.0:
        out["gullstrand_ratio"] = round(float(ant_kavg / post_kavg), 6)
    else:
        out["gullstrand_ratio"] = None

    return out


# ==========================================================================
# ZERNIKE POLYNOMIAL FITTING AND DERIVED INDICES
# ==========================================================================
#
# Fits Zernike polynomials (OSA/ANSI, up to order 8 = 45 terms) to the
# elevation_anterior and elevation_posterior polar maps, then derives:
#
#   - Per-coefficient outputs (45 per surface, in micrometers)
#   - BFS parameters (radius, RMS residual)
#   - RMSf / RMSb (higher-order aberration RMS, indices 6..44)
#   - EIf / EIb (ectasia index, weighted linear discriminant)
#   - fbLikeness (front-back correlation metric)
#
# These are the screening indices that were previously set to None in
# compute_screening_indices because they require Zernike decomposition.
#
# Algorithm:
#   1. Extract elevation data within a 4.0 mm diameter fitting zone
#   2. Convert polar (row, col) -> Cartesian (x, y)
#   3. Fit 1-parameter axis-centered BFS (CSO closed-form method)
#   4. Subtract BFS -> residual elevation
#   5. Re-center coordinates on BFS apex (no-op for axis-centered BFS)
#   6. Normalize rho = r / fitting_radius
#   7. Build Zernike design matrix (order 8, 45 terms)
#   8. Solve via SVD least-squares
#   9. Store coefficients in micrometers (mm * 1000)
#
# Source: legacy/legacy_zernike_reconstructions.py (proven algorithms),
#         formulas_screening_indices.md (EIf/EIb/RMS/fbLikeness formulas),
#         formulas_zernike_comparison.md (OSA/ANSI convention verification).
# ==========================================================================

# Zernike fitting constants
_ZERNIKE_MAX_ORDER = 8  # 45 terms (j=0..44) — used by OPD pipeline
_ZERNIKE_N_TERMS = 45  # OPD pipeline term count
_ZERNIKE_SCREENING_MAX_ORDER = 7  # 36 terms, CSO GetAltDecomposition
_ZERNIKE_SCREENING_N_TERMS = 36
_ZERNIKE_FITTING_RADIUS = 4.0  # mm -- CSO passes RFitting=4.0 (radius, NOT diameter)
_ZERNIKE_MIN_POINTS = 100  # minimum valid points for fitting
_ZERNIKE_BFS_R_MIN = 4.0  # plausible BFS radius range (mm)
_ZERNIKE_BFS_R_MAX = 20.0
_ZERNIKE_BFS_CENTER_MAX = 5.0  # max |cx| or |cy| (mm)
_ZERNIKE_CONIC_P_ANTERIOR = 0.8  # asphericity weight for anterior surface
_ZERNIKE_CONIC_P_POSTERIOR = 0.7  # asphericity weight for posterior surface


def _zernike_radial_polynomial(n, abs_m, rho):
    """
    Compute the Zernike radial polynomial R_n^|m|(rho).

    R_n^|m|(rho) = sum_{s=0}^{(n-|m|)/2} (-1)^s * (n-s)! /
        (s! * ((n+|m|)/2-s)! * ((n-|m|)/2-s)!) * rho^(n-2s)
    """
    if (n - abs_m) % 2 != 0:
        return np.zeros_like(rho)
    R = np.zeros_like(rho)
    for s in range((n - abs_m) // 2 + 1):
        coeff = (
            (-1) ** s
            * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial((n + abs_m) // 2 - s)
                * math.factorial((n - abs_m) // 2 - s)
            )
        )
        R += coeff * (rho ** (n - 2 * s))
    return R


def _zernike_polynomial(n, m, rho, theta):
    """
    Evaluate a single Zernike polynomial Z_n^m(rho, theta) with
    OSA/ANSI normalization.

    Normalization: sqrt(2(n+1)) for m != 0, sqrt(n+1) for m == 0.
    Angular part: cos(m*theta) for m >= 0, -sin(m*theta) for m < 0.
    """
    R = _zernike_radial_polynomial(n, abs(m), rho)
    if m == 0:
        norm = math.sqrt(n + 1)
    else:
        norm = math.sqrt(2 * (n + 1))
    if m >= 0:
        return norm * R * np.cos(m * theta)
    else:
        return norm * R * np.sin(abs(m) * theta)


def _zernike_generate_modes(max_order=_ZERNIKE_MAX_ORDER):
    """
    Generate (j, n, m) tuples for all Zernike terms up to max_order.

    Returns list of (j, n, m) where j is the OSA/ANSI single index.
    """
    modes = []
    j = 0
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            modes.append((j, n, m))
            j += 1
    return modes


def _fit_conic_bfs_radius(x, y, z, p=1.0, z_apex=None):
    """
    Asphericity-weighted BFS radius (CSO GetAltDecomposition formula).

    R = sum(s * (h^2 + p * s^2)) / (2 * sum(s^2))

    where s = |z - z_apex|, h = sqrt(x^2 + y^2).
    When p=1.0, reduces to standard sphere BFS.

    Parameters
    ----------
    x, y, z : np.ndarray
        Cartesian coordinates and elevation values.
    p : float
        Conic asphericity parameter. 0.8 for anterior, 0.7 for posterior.
    z_apex : float or None
        Elevation at the corneal apex (row 0 of polar map).  When provided,
        used directly instead of estimating from the innermost data point.
        CSO uses the actual row-0 value (GetAltDecomposition line 41).

    Returns
    -------
    float or None
        Conic BFS radius in mm, or None on failure.
    """
    h2 = x**2 + y**2
    if z_apex is None:
        z_apex = z[np.argmin(h2)]  # fallback: innermost non-row-0 point
    s = np.abs(z - z_apex)
    s2 = s**2
    denom = 2.0 * np.sum(s2)
    if denom < 1e-30:
        return None
    R = float(np.sum(s * (h2 + p * s2)) / denom)
    if R < _ZERNIKE_BFS_R_MIN or R > _ZERNIKE_BFS_R_MAX:
        return None
    return R


def _conic_sag(r, R, p):
    """
    Conic section sagittal height.

    sag = (R / p) * (1 - sqrt(1 - p * r^2 / R^2))

    Parameters
    ----------
    r : np.ndarray
        Radial distance from optical axis (mm).
    R : float
        Conic BFS radius (mm).
    p : float
        Asphericity parameter.

    Returns
    -------
    np.ndarray
        Sagittal height values.
    """
    disc = 1.0 - p * r**2 / R**2
    disc = np.maximum(disc, 0.0)
    return (R / p) * (1.0 - np.sqrt(disc))


def _bfs_sphere_z(params, x, y):
    """
    Compute z-values on a sphere surface given parameters [cx, cy, cz, R].

    z = cz - sqrt(R^2 - (x-cx)^2 - (y-cy)^2)

    This models the sagittal height of a sphere of radius R centered
    at (cx, cy, cz).
    """
    cx, cy, cz, R = params
    d = R**2 - (x - cx) ** 2 - (y - cy) ** 2
    return cz - np.sqrt(np.maximum(d, 0.0))


def _fit_bfs_1param(x, y, z, z_apex=0.0):
    """
    1-parameter axis-centered BFS (CSO method).

    R = sum((h^2 + z^2) * z) / (2 * sum(z^2))

    The sphere center is forced on the optical axis at (0, 0, cz)
    where cz = z_apex + R.  ``z_apex`` is the elevation at the
    map's geometric center (row 0 of the polar map).  For the
    anterior surface this is 0.  For the posterior surface it is
    the corneal thickness at vertex (~0.5-0.6 mm) because the
    posterior map is referenced to the anterior apex.

    Parameters
    ----------
    x, y, z : np.ndarray
        Cartesian coordinates and elevation values.
    z_apex : float
        Elevation at the surface apex (h=0).  The formula
        is applied to ``z - z_apex`` so that the apex sits at 0.

    Returns (params_dict, rms_residual) or (None, None) on failure.
    """
    if len(x) < _ZERNIKE_MIN_POINTS:
        return None, None

    h2 = x**2 + y**2

    # Re-center z so the apex is at z=0
    z_rc = z - z_apex

    z2 = z_rc**2

    denom = 2.0 * np.sum(z2)
    if denom < 1e-30:
        return None, None

    R = float(np.sum((h2 + z2) * z_rc) / denom)

    if R < _ZERNIKE_BFS_R_MIN or R > _ZERNIKE_BFS_R_MAX:
        return None, None

    # BFS: z_sphere = z_apex + R - sqrt(R^2 - h^2)
    # center at (0, 0, z_apex + R)
    disc = R**2 - h2
    disc = np.maximum(disc, 0.0)
    z_sphere = z_apex + R - np.sqrt(disc)
    residuals = z - z_sphere
    rms = float(np.sqrt(np.mean(residuals**2)))

    params = {
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": float(z_apex + R),
        "radius": float(R),
    }
    return params, rms


def _fit_zernike_coefficients(
    x_centered, y_centered, residuals, fitting_radius, max_order=_ZERNIKE_MAX_ORDER
):
    """
    Fit Zernike polynomials to residual elevation data via SVD.

    Parameters
    ----------
    x_centered, y_centered : np.ndarray
        Cartesian coordinates re-centered on BFS apex.
    residuals : np.ndarray
        Elevation residuals after BFS subtraction (mm).
    fitting_radius : float
        Normalization radius (mm).
    max_order : int
        Maximum Zernike radial order (default 8 -> 45 terms).

    Returns
    -------
    coefficients_um : np.ndarray or None
        Zernike coefficients in micrometers (45 terms).
    rms_fit : float or None
        RMS of Zernike fit residual in mm.
    """
    # Filter to finite values
    valid = np.isfinite(x_centered) & np.isfinite(y_centered) & np.isfinite(residuals)
    xv, yv, zv = x_centered[valid], y_centered[valid], residuals[valid]

    if len(xv) < _ZERNIKE_MIN_POINTS:
        return None, None

    # Normalize to unit disc
    rho = np.sqrt(xv**2 + yv**2) / fitting_radius
    theta = np.arctan2(yv, xv)

    # Keep only points within the unit disc (with small tolerance)
    inside = rho <= 1.001
    rho, theta, zv = rho[inside], theta[inside], zv[inside]

    if len(rho) < _ZERNIKE_MIN_POINTS:
        return None, None

    # Build design matrix
    modes = _zernike_generate_modes(max_order)
    n_modes = len(modes)
    A = np.empty((len(rho), n_modes))
    for col_idx, (j, n, m) in enumerate(modes):
        A[:, col_idx] = _zernike_polynomial(n, m, rho, theta)

    # Solve via SVD least-squares
    try:
        coeffs_mm, _, _, _ = np.linalg.lstsq(A, zv, rcond=None)
        if not np.all(np.isfinite(coeffs_mm)):
            return None, None
    except np.linalg.LinAlgError:
        return None, None

    # Convert mm to micrometers
    coefficients_um = coeffs_mm * 1000.0

    # Compute RMS of Zernike fit residual
    fitted = A @ coeffs_mm
    rms_fit = float(np.sqrt(np.mean((zv - fitted) ** 2)))

    return coefficients_um, rms_fit


def _polar_map_apex_z(polar_map):
    """Return the elevation at the map center (row 0), or 0.0 on failure.

    For the anterior surface this is 0.0 (map referenced to itself).
    For the posterior surface this is ~corneal thickness at vertex
    (~0.5-0.6 mm) because the posterior map is referenced to the
    anterior apex.
    """
    if polar_map is None:
        return 0.0
    row0 = polar_map[0].astype(np.float64)
    row0[row0 == _MISSING] = np.nan
    valid = row0[np.isfinite(row0)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid))


def _polar_elevation_to_cartesian(polar_map, fitting_radius):
    """
    Extract (x, y, z) from a polar elevation map within fitting_radius.

    Parameters
    ----------
    polar_map : np.ndarray, shape (n_rows, 256)
        Elevation data in mm, with -1000 for missing.
    fitting_radius : float
        Maximum radial distance to include (mm).

    Returns
    -------
    x, y, z : np.ndarray
        Cartesian coordinates and elevation values for valid points.
    """
    clean = polar_map.astype(np.float64)
    clean[clean == _MISSING] = np.nan

    n_rows, n_cols = clean.shape
    max_row = min(n_rows, int(fitting_radius / _R_STEP_MM) + 1)

    # Build coordinate grids for the fitting zone (skip row 0: no info)
    rad = np.arange(1, max_row) * _R_STEP_MM  # (max_row-1,)
    ang = np.linspace(0, 2 * np.pi, n_cols, endpoint=False)  # (256,)
    rg, tg = np.meshgrid(rad, ang, indexing="ij")  # (max_row-1, 256)

    zone_data = clean[1:max_row, :n_cols]
    valid_mask = ~np.isnan(zone_data)

    x = (rg * np.cos(tg))[valid_mask]
    y = (rg * np.sin(tg))[valid_mask]
    z = zone_data[valid_mask]

    return x, y, z


def _compute_hoa_rms(coeffs_um):
    """
    Higher-order aberration RMS: sqrt(sum(c[j]^2 for j >= 6)).

    Indices 0-5 are orders 0-2 (piston, tilt, defocus, astigmatism).
    """
    if coeffs_um is None or len(coeffs_um) < 7:
        return None
    return float(np.sqrt(np.sum(coeffs_um[6:] ** 2)))


def _compute_eif(coeffs_um, exam_eye):
    """
    Ectasia Index, Front (EIf).

    EIf = max(0, 2.071 * (
          Z[7]*0.213986 + Z[12]*0.091203 + Z[17]*(-0.730099)
        + hOARMS*0.060440
        + sign_LR * (Z[11]*(-0.191587) + Z[18]*(-0.313218))
      ) - 0.5)

    Source: formulas_screening_indices.md Section 7.
    """
    if coeffs_um is None or len(coeffs_um) < 19:
        return None

    sign_lr = 1.0 if str(exam_eye).strip().upper() == "OD" else -1.0

    # Residual HOA RMS: HOA RMS excluding indices 7, 11, 12, 17, 18
    excluded = {7, 11, 12, 17, 18}
    residual_sum = 0.0
    for k in range(6, len(coeffs_um)):
        if k not in excluded:
            residual_sum += coeffs_um[k] ** 2
    hoa_rms_residual = math.sqrt(residual_sum)

    val = (
        2.071
        * (
            coeffs_um[7] * 0.213986
            + coeffs_um[12] * 0.091203
            + coeffs_um[17] * (-0.730099)
            + hoa_rms_residual * 0.060440
            + sign_lr * (coeffs_um[11] * (-0.191587) + coeffs_um[18] * (-0.313218))
        )
        - 0.5
    )

    return max(0.0, val)


def _compute_eib(coeffs_um, exam_eye):
    """
    Ectasia Index, Back (EIb).

    EIb = max(0, 2.0 * (
          Z[7]*0.119849 + Z[12]*0.049943 + Z[17]*(-0.281500)
        + Z[31]*0.198876
        + sign_LR * (Z[11]*(-0.112375) + Z[18]*(-0.223590))
      ) + 0.0)

    Requires order 8 (Z[31] = tertiary coma Z(7,+1)).
    Source: formulas_screening_indices.md Section 8.
    """
    if coeffs_um is None or len(coeffs_um) < 32:
        return None

    sign_lr = 1.0 if str(exam_eye).strip().upper() == "OD" else -1.0

    val = (
        2.0
        * (
            coeffs_um[7] * 0.119849
            + coeffs_um[12] * 0.049943
            + coeffs_um[17] * (-0.281500)
            + coeffs_um[31] * 0.198876
            + sign_lr * (coeffs_um[11] * (-0.112375) + coeffs_um[18] * (-0.223590))
        )
        + 0.0
    )

    return max(0.0, val)


def _compute_fb_likeness(coeffs_ant_um, coeffs_post_um, rmsf, rmsb):
    """
    Front-Back Likeness.

    fbLikeness = ( |2*Z_f[7]  - Z_b[7] |
                 + |2*Z_f[8]  - Z_b[8] |
                 + |2*Z_f[12] - Z_b[12]|
                 + |2*Z_f[17] - Z_b[17]|
                 + |2*Z_f[24] - Z_b[24]|
                 ) / (RMSf + RMSb)

    Source: formulas_screening_indices.md Section 13.
    """
    if coeffs_ant_um is None or coeffs_post_um is None or rmsf is None or rmsb is None:
        return None

    if len(coeffs_ant_um) < 25 or len(coeffs_post_um) < 25:
        return None

    denom = rmsf + rmsb
    if denom <= 0:
        return None

    indices = [7, 8, 12, 17, 24]
    numerator = sum(abs(2.0 * coeffs_ant_um[k] - coeffs_post_um[k]) for k in indices)

    return float(numerator / denom)


def _zernike_fit_one_surface(
    polar_map, fitting_radius=_ZERNIKE_FITTING_RADIUS, surface_type="anterior"
):
    """
    Fit Zernike polynomials to conic-BFS-subtracted elevation on a single surface.

    Matches CSO GetAltDecomposition:
      1. Extract elevation within fitting radius (skip row 0)
      2. Compute conic BFS radius with asphericity weighting
         (p=0.8 anterior, p=0.7 posterior)
      3. Subtract conic reference surface
      4. Fit 36 Zernike terms (order 7) to residuals via SVD

    Parameters
    ----------
    polar_map : np.ndarray or None
        Polar elevation map (n_rows, 256) in mm.
    fitting_radius : float
        Fitting zone radius in mm.
    surface_type : str
        'anterior' (p=0.8) or 'posterior' (p=0.7).

    Returns
    -------
    coefficients_um : np.ndarray or None
        36 Zernike coefficients in micrometers.
    bfs_params : dict or None
        Conic BFS radius (and center at origin).
    bfs_rms : float or None
        Conic BFS residual RMS in mm.
    zernike_rms : float or None
        Zernike fit residual RMS in mm.
    """
    if polar_map is None:
        return None, None, None, None

    # Step 1: Extract Cartesian coordinates from the polar elevation map
    x, y, z = _polar_elevation_to_cartesian(polar_map, fitting_radius)
    if len(x) < _ZERNIKE_MIN_POINTS:
        return None, None, None, None

    # Step 2: Compute conic BFS radius (CSO GetAltDecomposition)
    asphericity = (
        _ZERNIKE_CONIC_P_ANTERIOR if surface_type == "anterior" else _ZERNIKE_CONIC_P_POSTERIOR
    )
    # Use row-0 apex value from the polar map (CSO convention)
    apex_z = _polar_map_apex_z(polar_map)
    R_conic = _fit_conic_bfs_radius(x, y, z, p=asphericity, z_apex=apex_z)
    if R_conic is None:
        return None, None, None, None

    bfs_params = {
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": float(R_conic),
        "radius": float(R_conic),
    }

    # Step 3: Subtract conic reference surface
    # CSO works with unsigned sag: sag = |z - z_apex|, then
    # residual = conic_sag(r) - sag_measured.
    # We must match this sign convention so the HOA Zernike
    # coefficients (and thus DZMax, EIf, EIb) have the correct sign.
    h2 = x**2 + y**2
    h = np.sqrt(h2)
    sag_measured = np.abs(z - apex_z)
    z_ref = _conic_sag(h, R_conic, asphericity)
    residuals = z_ref - sag_measured
    bfs_rms = float(np.sqrt(np.mean(residuals**2)))

    # Step 4: Fit 36 Zernike terms (order 7) to residuals
    coefficients_um, zernike_rms = _fit_zernike_coefficients(
        x,
        y,
        residuals,
        fitting_radius,
        max_order=_ZERNIKE_SCREENING_MAX_ORDER,
    )

    return coefficients_um, bfs_params, bfs_rms, zernike_rms


def compute_zernike_indices(raw_segments, metadata):
    """
    Fit Zernike polynomials to elevation maps and compute derived indices.

    Matches CSO GetAltDecomposition: fits OSA/ANSI Zernike polynomials up
    to order 7 (36 terms, j=0..35) to conic-BFS-subtracted elevation at a
    fixed 4.0 mm radius fitting zone. Derives screening-relevant indices:
    RMSf, RMSb, EIf, EIb, and fbLikeness.

    Algorithm:
      1. Extract elevation within 4mm radius, convert polar->Cartesian
      2. Compute conic BFS radius with asphericity weighting
         (p=0.8 anterior, p=0.7 posterior)
      3. Subtract conic reference surface
      4. Fit 36 Zernike terms (order 7) to residuals via SVD
      5. Zero low-order coefficients j=0..5 (piston, tilt, defocus, astigmatism)
      6. Compute RMSf/b, EIf/b, fbLikeness from HOA coefficient vectors (j>=6)

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of core.parse_csv(). Must contain 'elevation_anterior'
        and/or 'elevation_posterior' with shape (n_rows, 256).
        Values are in mm with -1000 as missing sentinel.
    metadata : dict
        Must contain 'Exam_Eye' with value 'OD' or 'OS'.

    Returns
    -------
    dict
        Keys:
          Zernike coefficients (36 + 36 = 72 keys):
            zernike_ant_4mm_z{j}_um  (j=0..35)
            zernike_post_4mm_z{j}_um (j=0..35)
          BFS parameters (4 keys):
            zernike_ant_4mm_bfs_radius
            zernike_ant_4mm_bfs_rms
            zernike_post_4mm_bfs_radius
            zernike_post_4mm_bfs_rms
          Zernike fit quality (2 keys):
            zernike_ant_4mm_fit_rms
            zernike_post_4mm_fit_rms
          Derived screening indices (5 keys):
            zernike_rmsf    -- HOA RMS anterior (um)
            zernike_rmsb    -- HOA RMS posterior (um)
            zernike_eif     -- Ectasia Index front
            zernike_eib     -- Ectasia Index back
            zernike_fb_likeness -- Front-back likeness
    """
    out = {}
    exam_eye = metadata.get("Exam_Eye", "")
    if exam_eye not in ("OD", "OS"):
        exam_eye = "OD"  # default, matching CSO convention

    elev_ant = raw_segments.get("elevation_anterior")
    elev_post = raw_segments.get("elevation_posterior")

    # --- Fit anterior surface (conic p=0.8) ---
    coeffs_ant, bfs_ant, bfs_rms_ant, fit_rms_ant = _zernike_fit_one_surface(
        elev_ant, surface_type="anterior"
    )

    # --- Fit posterior surface (conic p=0.7) ---
    coeffs_post, bfs_post, bfs_rms_post, fit_rms_post = _zernike_fit_one_surface(
        elev_post, surface_type="posterior"
    )

    # --- Zero low-order terms j=0..5 (CSO GetAltDecomposition convention) ---
    # Piston, tilt, defocus, and astigmatism are absorbed by the conic BFS
    # and carry no screening-relevant information.
    if coeffs_ant is not None:
        coeffs_ant[:6] = 0.0
    if coeffs_post is not None:
        coeffs_post[:6] = 0.0

    # --- Output: Zernike coefficients per surface ---
    for j in range(_ZERNIKE_SCREENING_N_TERMS):
        out[f"zernike_ant_4mm_z{j}_um"] = float(coeffs_ant[j]) if coeffs_ant is not None else None
        out[f"zernike_post_4mm_z{j}_um"] = (
            float(coeffs_post[j]) if coeffs_post is not None else None
        )

    # --- Output: BFS parameters (conic radius) ---
    out["zernike_ant_4mm_bfs_radius"] = bfs_ant["radius"] if bfs_ant is not None else None
    out["zernike_ant_4mm_bfs_rms"] = bfs_rms_ant
    out["zernike_post_4mm_bfs_radius"] = bfs_post["radius"] if bfs_post is not None else None
    out["zernike_post_4mm_bfs_rms"] = bfs_rms_post

    # --- Output: Zernike fit quality ---
    out["zernike_ant_4mm_fit_rms"] = fit_rms_ant
    out["zernike_post_4mm_fit_rms"] = fit_rms_post

    # --- Output: Derived screening indices ---
    rmsf = _compute_hoa_rms(coeffs_ant)
    rmsb = _compute_hoa_rms(coeffs_post)
    out["zernike_rmsf"] = rmsf
    out["zernike_rmsb"] = rmsb

    out["zernike_eif"] = _compute_eif(coeffs_ant, exam_eye)
    out["zernike_eib"] = _compute_eib(coeffs_post, exam_eye)

    out["zernike_fb_likeness"] = _compute_fb_likeness(
        coeffs_ant,
        coeffs_post,
        rmsf,
        rmsb,
    )

    return out


# ===================================================================
# OPD WAVEFRONT (1936 columns)
# ===================================================================
#
# Recomputes the 1,936 OPD (Optical Path Difference) wavefront columns
# that the MS-39 places in the global CSV export.
#
# Layout:
#   Pupil-centered:  2 surfaces x 11 diameters x 44 params = 968
#   CV-centered:     2 surfaces x 11 diameters x 44 params = 968
#   Total: 1,936
#
# Per diameter, per surface/variant: Radius, Diameter,
#   WFCoeff[0..35], Cyl, CylAx, LSA, LCA, HOA RMS, CornealPower
#   = 44 values
#
# Algorithm:
#   1. Extract elevation within disc of radius d/2
#   2. BFS fit (4-parameter)
#   3. Zernike fit on residuals (45 terms internally, output first 36)
#   4. Convert geometric coefficients to OPD via refractive index
#   5. Derive: Cyl, CylAx, LSA, LCA, HOA RMS, CornealPower
#
# Source: formulas_opd_wavefront.md, formulas_cornealwf_refraction.md
# ===================================================================

# 11 analysis diameters from 2.0 to 7.0 mm in 0.5 mm steps
_OPD_DIAMETERS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

# Number of Zernike coefficients exported per fit (order 7, j=0..35)
_OPD_N_EXPORT_COEFS = 36

# Refractive index differences for OPD conversion
_OPD_DN_ANTERIOR = -0.376  # (n_air - n_cornea) = 1.000 - 1.376
_OPD_DN_POSTERIOR = 0.040  # (n_cornea - n_aqueous) = 1.376 - 1.336


def _opd_diameter_label(d):
    """
    Convert a diameter float to the key-friendly label.

    2.0 -> '2mm', 2.5 -> '2p5mm', 3.0 -> '3mm', etc.
    """
    if d == int(d):
        return f"{int(d)}mm"
    return f"{str(d).replace('.', 'p')}mm"


def _opd_derived_metrics(coeffs_opd, bfs_radius, fitting_radius, corneal_power=None):
    """
    Compute the 6 derived OPD metrics from Zernike coefficients.

    Parameters
    ----------
    coeffs_opd : np.ndarray
        OPD-domain Zernike coefficients (at least 36 terms), in um.
    bfs_radius : float
        Best-fit sphere radius in mm.
    fitting_radius : float
        Fitting radius in mm (= diameter / 2).
    corneal_power : float or None
        If provided, use this value instead of computing from bfs_radius.
        Used by the ray-tracing pipeline (power from focal distance).

    Returns
    -------
    dict with keys: cyl, cylax, lsa, lca, hoa_rms, cornealpower
    """
    r = fitting_radius  # mm
    r2 = r * r

    c3 = coeffs_opd[3]  # Z(2,-2) oblique astigmatism
    c5 = coeffs_opd[5]  # Z(2,+2) vertical astigmatism
    c7 = coeffs_opd[7]  # Z(3,-1) vertical coma
    c8 = coeffs_opd[8]  # Z(3,+1) horizontal coma
    c12 = coeffs_opd[12]  # Z(4,0) spherical aberration

    # Cyl = -4*sqrt(6) * sqrt(c3^2 + c5^2) / r^2
    cyl = -4.0 * math.sqrt(6.0) * math.sqrt(c3**2 + c5**2) / r2

    # CylAx = 0.5 * atan2(c3, c5) in degrees, TABO convention [0, 180)
    cylax_rad = 0.5 * math.atan2(c3, c5)
    cylax = math.degrees(cylax_rad) % 180.0

    # LSA = -24*sqrt(5) * c12 / r^2
    lsa = -24.0 * math.sqrt(5.0) * c12 / r2

    # LCA (actually longitudinal coma aberration) = 24*sqrt(2) * |coma| / r^2
    coma_mag = math.sqrt(c7**2 + c8**2)
    lca = 24.0 * math.sqrt(2.0) * coma_mag / r2

    # HOA RMS = sqrt(sum(c[j]^2 for j >= 6))
    hoa_rms = float(np.sqrt(np.sum(coeffs_opd[6:_OPD_N_EXPORT_COEFS] ** 2)))

    # CornealPower: from ray-traced focal distance or from BFS radius
    if corneal_power is not None:
        cp = corneal_power
    elif bfs_radius is not None and bfs_radius > 0:
        cp = 337.5 / bfs_radius
    else:
        cp = None

    return {
        "cyl": float(cyl),
        "cylax": float(cylax),
        "lsa": float(lsa),
        "lca": float(lca),
        "hoa_rms": float(hoa_rms),
        "cornealpower": float(cp) if cp is not None else None,
    }


def _opd_raytrace_one_diameter(
    polar_map, fitting_radius, n1, n2, offset_x=0.0, offset_y=0.0, _cache=None
):
    """
    Ray-trace through a single corneal surface and fit Zernike to the OPD.

    Matches CSO's ``FromRayTracing`` (single-surface) pipeline:
    biquadratic eval -> Snell's law -> OPL -> OPD -> Zernike -> um.

    Parameters
    ----------
    polar_map : ndarray, shape (n_rows, 256)
        Elevation map in mm, -1000 for missing.
    fitting_radius : float
        Half the analysis diameter (mm).
    n1, n2 : float
        Refractive indices (incident and refracting media).
    offset_x, offset_y : float
        Coordinate shift (same convention as _opd_fit_one_diameter).
        Pupil-centered: (0, 0). CV-centered: (-PupilCX, -PupilCY).

    Returns
    -------
    coeffs_opd_um : ndarray (36,) or None
        OPD Zernike coefficients in um (piston zeroed). Already physical --
        do NOT multiply by delta_n.
    focal_z : float or None
        Focal distance from least-squares ray convergence (mm).
    """
    if polar_map is None:
        return None, None

    max_data_radius = (polar_map.shape[0] - 1) * _R_STEP_MM
    if fitting_radius > max_data_radius:
        return None, None

    # 1. Create ray grid (20 radial x 50 angular = 1000 rays, matching CSO)
    ray_xy = create_ray_grid(fitting_radius)

    # 2. Evaluate surface at map-frame positions.
    #    The map center ~ pupil center (or vertex for Scheimpflug).
    #    analysis_coords = map_coords + offset  =>  map = analysis - offset
    query_xy = np.column_stack((ray_xy[:, 0] - offset_x, ray_xy[:, 1] - offset_y))
    z, dz_dx, dz_dy, valid = _biquad_eval_batch(polar_map, query_xy, _cache=_cache)

    # Filter to valid rays
    n_valid = int(np.sum(valid))
    if n_valid < _ZERNIKE_MIN_POINTS:
        return None, None

    mask = valid
    # Analysis-centered positions for Zernike fitting
    x_v = ray_xy[mask, 0]
    y_v = ray_xy[mask, 1]
    # Query (map-frame) positions for physical ray geometry
    xq_v = query_xy[mask, 0]
    yq_v = query_xy[mask, 1]
    z_v = z[mask]
    dz_dx_v = dz_dx[mask]
    dz_dy_v = dz_dy[mask]
    nv = n_valid

    # 3. Build 3D positions using QUERY coordinates for physical geometry.
    #    The surface was evaluated at (xq, yq), so the physical intersection
    #    is at (xq, yq, z). The collimated ray enters at (xq, yq, 0).
    #    CSO places rays at PupilCenter + (rho, theta).
    surface_points = np.column_stack((xq_v, yq_v, z_v))
    entry_points = np.column_stack((xq_v, yq_v, np.zeros(nv)))

    # 4. Surface normals (outward, toward incoming light)
    normals = surface_normals_from_gradients(dz_dx_v, dz_dy_v)

    # 5. Vectorial Snell's law
    incident_dir = np.array([0.0, 0.0, 1.0])
    incident_dirs = np.broadcast_to(incident_dir, (nv, 3)).copy()
    try:
        refracted_dirs = snells_law_vector_batch(incident_dirs, normals, n1, n2)
    except ValueError:
        return None, None

    # 6. 3D least-squares focal point (formulas Section 5b)
    try:
        focal_x, focal_y, focal_z = estimate_focal_point(
            surface_points,
            refracted_dirs,
        )
    except ValueError:
        return None, None

    # 7. OPL using per-ray reference plane (CSO method, formulas Step 9-10).
    #    Each ray's reference plane is perpendicular to the refracted ray
    #    and passes through the 3D focal point. The distance from the
    #    surface point to this plane along the refracted ray is:
    #      d2 = |(F - P) . d_refracted|
    #    This automatically removes the tilt introduced by any lateral
    #    focal offset (Fx, Fy != 0).
    d1 = np.linalg.norm(surface_points - entry_points, axis=1)  # air path
    focal_point = np.array([focal_x, focal_y, focal_z])
    fp_vec = focal_point[np.newaxis, :] - surface_points  # (N, 3)
    d2 = np.abs(np.sum(fp_vec * refracted_dirs, axis=1))  # cornea path
    opl = n1 * d1 + n2 * d2

    # 8. OPD = OPL - mean(OPL)
    opd = opl - np.mean(opl)

    # 9. Fit Zernike to OPD (mm) -- order 7 = 36 terms (matching CSO NPol=36)
    coeffs_um, _ = _fit_zernike_coefficients(
        x_v,
        y_v,
        opd,
        fitting_radius,
        max_order=7,
    )
    if coeffs_um is None:
        return None, None

    # 10. Zero piston (CSO: zernike[0] = 0)
    coeffs_um[0] = 0.0

    return coeffs_um, focal_z


# ---------------------------------------------------------------------------
# Newton iteration for posterior surface intersection
# ---------------------------------------------------------------------------


def _newton_posterior_intersection(post_coeffs, refracted_dir, ant_point, max_iter=20, tol=1e-6):
    """
    Find where a refracted ray intersects the posterior biquadratic surface.

    Matches CSO's ``NewtonFindPosteriorIntersection`` (formulas Section 4b).

    The system:
        x = ant.x + (z - ant.z) * tx     where tx = dir.x / dir.z
        y = ant.y + (z - ant.z) * ty     where ty = dir.y / dir.z
        z = a*x^2 + b*y^2 + c*x + d*y + e*x*y + f

    Uses Cramer's rule for the 3x3 Newton update.

    Parameters
    ----------
    post_coeffs : ndarray (6,)
        Biquadratic coefficients [a, b, c, d, e, f] (CSO order).
    refracted_dir : ndarray (3,)
        Direction of the ray refracted from the anterior surface.
    ant_point : ndarray (3,)
        3D point on the anterior surface where the ray was refracted.

    Returns
    -------
    ndarray (3,) or None
        Posterior intersection point [x, y, z], or None if no convergence.
    """
    a, b, c, d, e, f = post_coeffs
    if abs(refracted_dir[2]) < 1e-15:
        return None
    tx = refracted_dir[0] / refracted_dir[2]
    ty = refracted_dir[1] / refracted_dir[2]
    ax, ay, az = ant_point

    # Initial guess: project the anterior point onto the posterior surface
    x = ax
    y = ay
    z = a * x * x + b * y * y + c * x + d * y + e * x * y + f

    for _ in range(max_iter):
        # Residual
        r0 = x - (ax + tx * (z - az))
        r1 = y - (ay + ty * (z - az))
        r2 = z - (a * x * x + b * y * y + c * x + d * y + e * x * y + f)

        # Jacobian
        # Row 0: [1, 0, -tx]
        # Row 1: [0, 1, -ty]
        # Row 2: [-(2*a*x + c + e*y), -(2*b*y + d + e*x), 1]
        j20 = -(2.0 * a * x + c + e * y)
        j21 = -(2.0 * b * y + d + e * x)

        # Solve J * delta = -r using Cramer's rule for 3x3
        # J = [[1, 0, -tx], [0, 1, -ty], [j20, j21, 1]]
        # det(J) = 1*(1 - (-ty)*j21) - 0 + (-tx)*(0 - j20)
        #        = 1 + ty*j21 - tx*(-j20) ... let me just expand
        # Actually, using cofactor expansion along row 0:
        # det = 1*(1*1 - (-ty)*j21) - 0 + (-tx)*(0*j21 - 1*j20)
        #     = 1 + ty*j21 + tx*j20
        det = 1.0 + ty * j21 + tx * j20

        if abs(det) < 1e-12:
            return None

        # Cramer's rule: replace column i with -r, compute det
        nr0, nr1, nr2 = -r0, -r1, -r2

        # dx: replace col 0 of J with -r
        # [[nr0, 0, -tx], [nr1, 1, -ty], [nr2, j21, 1]]
        dx = nr0 * (1.0 + ty * j21) - 0.0 + (-tx) * (nr1 * j21 - nr2)
        dx /= det

        # dy: replace col 1 of J with -r
        # [[1, nr0, -tx], [0, nr1, -ty], [j20, nr2, 1]]
        dy = (nr1 - (-ty) * nr2) - nr0 * (ty * j20) + (-tx) * (-nr1 * j20)
        dy /= det

        # dz: replace col 2 of J with -r
        # [[1, 0, nr0], [0, 1, nr1], [j20, j21, nr2]]
        dz = 1.0 * (1.0 * nr2 - nr1 * j21) - 0.0 + nr0 * (0.0 * j21 - 1.0 * j20)
        dz /= det

        x += dx
        y += dy
        z += dz

        if abs(dx) + abs(dy) + abs(dz) < tol:
            return np.array([x, y, z])

    return None  # did not converge


# ---------------------------------------------------------------------------
# Dual-surface ray-tracing (full OPD Total)
# ---------------------------------------------------------------------------


def _opd_raytrace_dual_surface(
    ant_map, post_map, fitting_radius, offset_x=0.0, offset_y=0.0, _cache=None
):
    """
    Full dual-surface ray-trace matching CSO's FromRayTracing (Total mode).

    Traces collimated rays through both corneal surfaces:
    air -> anterior refraction -> corneal propagation (Newton iteration
    to find posterior intersection) -> posterior refraction -> aqueous.

    Parameters
    ----------
    ant_map : ndarray, shape (n_rows, 256)
        Anterior elevation map in mm, -1000 for missing.
    post_map : ndarray, shape (n_rows, 256)
        Posterior elevation map in mm, -1000 for missing.
    fitting_radius : float
        Half the analysis diameter (mm).
    offset_x, offset_y : float
        Coordinate shift for centering.

    Returns
    -------
    coeffs_opd_um : ndarray (36,) or None
        OPD Zernike coefficients in um (piston and defocus zeroed).
    focal_z : float or None
        Focal distance in mm.
    """
    if ant_map is None or post_map is None:
        return None, None

    max_data_radius = (ant_map.shape[0] - 1) * _R_STEP_MM
    if fitting_radius > max_data_radius:
        return None, None

    # 1. Create ray grid — CSO uses adaptive radial sampling for dual-surface:
    #    nRadial = int(PupilRadius / 0.1) + 1 (formulas Section 2)
    n_radial = int(fitting_radius / 0.1) + 1
    n_meridional = 50
    ray_xy = create_ray_grid(fitting_radius, n_radial=n_radial, n_meridional=n_meridional)

    # 2. Evaluate both surfaces at map-frame positions
    query_xy = np.column_stack((ray_xy[:, 0] - offset_x, ray_xy[:, 1] - offset_y))
    ant_z, ant_dzdx, ant_dzdy, post_coeffs, valid = _biquad_eval_dual_batch(
        ant_map, post_map, query_xy, _cache=_cache
    )

    # Filter to valid rays
    n_valid = int(np.sum(valid))
    if n_valid < _ZERNIKE_MIN_POINTS:
        return None, None

    mask = valid
    # Analysis-centered positions for Zernike fitting
    x_v = ray_xy[mask, 0]
    y_v = ray_xy[mask, 1]
    # Query (map-frame) positions for physical ray geometry
    xq_v = query_xy[mask, 0]
    yq_v = query_xy[mask, 1]
    ant_z_v = ant_z[mask]
    ant_dzdx_v = ant_dzdx[mask]
    ant_dzdy_v = ant_dzdy[mask]
    post_coeffs_v = post_coeffs[mask]
    nv = n_valid

    # 3. Build anterior 3D positions using QUERY coordinates
    ant_points = np.column_stack((xq_v, yq_v, ant_z_v))
    entry_points = np.column_stack((xq_v, yq_v, np.zeros(nv)))

    # 4. Anterior surface normals (outward, toward incoming light)
    ant_normals = surface_normals_from_gradients(ant_dzdx_v, ant_dzdy_v)

    # 5. First Snell's law: air -> cornea at anterior surface
    incident_dir = np.array([0.0, 0.0, 1.0])
    incident_dirs = np.broadcast_to(incident_dir, (nv, 3)).copy()
    try:
        refracted_ant = snells_law_vector_batch(incident_dirs, ant_normals, N_AIR, N_CORNEA)
    except ValueError:
        return None, None

    # 6. Newton iteration: find posterior surface intersection for each ray
    post_points = np.full((nv, 3), np.nan)
    post_normals = np.full((nv, 3), np.nan)
    ray_valid = np.ones(nv, dtype=bool)

    for i in range(nv):
        result = _newton_posterior_intersection(post_coeffs_v[i], refracted_ant[i], ant_points[i])
        if result is None:
            ray_valid[i] = False
            continue
        post_points[i] = result

        # Posterior surface normal from biquadratic coefficients
        # Our gradient: dz/dx = 2*a*x + c + e*y for z = a*x^2 + ...
        pc = post_coeffs_v[i]
        px, py = result[0], result[1]
        post_dzdx = 2.0 * pc[0] * px + pc[2] + pc[4] * py
        post_dzdy = 2.0 * pc[1] * py + pc[3] + pc[4] * px
        # Normal in our convention: (dz/dx, dz/dy, -1) / norm
        nn = np.array([post_dzdx, post_dzdy, -1.0])
        nn /= np.linalg.norm(nn)
        post_normals[i] = nn

    # Filter out failed Newton iterations
    if np.sum(ray_valid) < _ZERNIKE_MIN_POINTS:
        return None, None

    # Apply the valid mask
    rv = ray_valid
    x_v2 = x_v[rv]
    y_v2 = y_v[rv]
    entry_points2 = entry_points[rv]
    ant_points2 = ant_points[rv]
    refracted_ant2 = refracted_ant[rv]
    post_points2 = post_points[rv]
    post_normals2 = post_normals[rv]

    # 7. Second Snell's law: cornea -> aqueous at posterior surface
    try:
        refracted_post = snells_law_vector_batch(refracted_ant2, post_normals2, N_CORNEA, N_AQUEOUS)
    except ValueError:
        return None, None

    # 8. 3D least-squares focal point from POSTERIOR refracted rays
    try:
        focal_x, focal_y, focal_z = estimate_focal_point(post_points2, refracted_post)
    except ValueError:
        return None, None

    # 9. Outlier removal: rays whose focal-plane intersection is >2mm from axis
    focal_point = np.array([focal_x, focal_y, focal_z])
    dt = (focal_z - post_points2[:, 2]) / refracted_post[:, 2]
    fp_xy = post_points2[:, :2] + dt[:, np.newaxis] * refracted_post[:, :2]
    fp_dist = np.sqrt(fp_xy[:, 0] ** 2 + fp_xy[:, 1] ** 2)
    ray_ok = fp_dist <= 2.0

    if np.sum(ray_ok) < _ZERNIKE_MIN_POINTS:
        return None, None

    x_v3 = x_v2[ray_ok]
    y_v3 = y_v2[ray_ok]
    entry_points3 = entry_points2[ray_ok]
    ant_points3 = ant_points2[ray_ok]
    post_points3 = post_points2[ray_ok]
    refracted_post3 = refracted_post[ray_ok]

    # Re-estimate focal point after outlier removal
    try:
        focal_x, focal_y, focal_z = estimate_focal_point(post_points3, refracted_post3)
    except ValueError:
        return None, None
    focal_point = np.array([focal_x, focal_y, focal_z])

    # 10. OPL computation (3-segment: air + cornea + aqueous)
    # Per-ray reference plane perpendicular to refracted ray through focal point
    d_air = np.linalg.norm(ant_points3 - entry_points3, axis=1)
    d_cornea = np.linalg.norm(post_points3 - ant_points3, axis=1)
    fp_vec = focal_point[np.newaxis, :] - post_points3
    d_aqueous = np.abs(np.sum(fp_vec * refracted_post3, axis=1))

    opl = N_AIR * d_air + N_CORNEA * d_cornea + N_AQUEOUS * d_aqueous

    # 11. OPD = OPL - mean(OPL)
    opd = opl - np.mean(opl)

    # 12. OPL outlier removal: remove rays with |OPD| > 1.0 mm
    opd_ok = np.abs(opd) <= 1.0
    if np.sum(opd_ok) < _ZERNIKE_MIN_POINTS:
        return None, None

    x_final = x_v3[opd_ok]
    y_final = y_v3[opd_ok]
    opd_final = opd[opd_ok]

    # Re-center after removing outliers
    opd_final = opd_final - np.mean(opd_final)

    # 13. Fit Zernike to OPD (mm) — 36 terms (order 7)
    coeffs_um, _ = _fit_zernike_coefficients(
        x_final,
        y_final,
        opd_final,
        fitting_radius,
        max_order=7,
    )
    if coeffs_um is None:
        return None, None

    # 14. Zero piston (j=0) AND defocus (j=4) — CSO always does this for Total
    coeffs_um[0] = 0.0
    coeffs_um[4] = 0.0

    return coeffs_um, focal_z


def _opd_emit_one_block(
    prefix, coeffs_um, bfs_radius, delta_n, fitting_radius, diameter, corneal_power=None
):
    """
    Produce the 44 output keys for one surface/diameter/centering.

    Parameters
    ----------
    prefix : str
        Key prefix, e.g. 'opd_front_4mm'.
    coeffs_um : np.ndarray or None
        Zernike coefficients in um. For geometric pipeline, these are
        elevation coefficients that get multiplied by delta_n. For
        ray-tracing pipeline, these are already OPD coefficients
        (pass delta_n=1.0).
    bfs_radius : float or None
        BFS radius in mm (for Radius column).
    delta_n : float
        Refractive index difference for OPD conversion (1.0 for
        ray-traced coefficients that are already OPD).
    fitting_radius : float
        Fitting radius in mm (= diameter / 2).
    diameter : float
        Nominal analysis diameter in mm.
    corneal_power : float or None
        If provided, override the BFS-derived corneal power with this
        ray-traced value.

    Returns
    -------
    dict
        44 key-value pairs.
    """
    out = {}

    if coeffs_um is None:
        # Return all 44 keys as None
        out[f"{prefix}_radius"] = None
        out[f"{prefix}_diameter"] = None
        for j in range(_OPD_N_EXPORT_COEFS):
            out[f"{prefix}_wfcoeff_{j}"] = None
        out[f"{prefix}_cyl"] = None
        out[f"{prefix}_cylax"] = None
        out[f"{prefix}_lsa"] = None
        out[f"{prefix}_lca"] = None
        out[f"{prefix}_hoa_rms"] = None
        out[f"{prefix}_cornealpower"] = None
        return out

    # Convert geometric coefficients to OPD domain
    coeffs_opd = coeffs_um[:_OPD_N_EXPORT_COEFS].copy() * delta_n

    # Zero piston (matches CSO: coeffs[0] = 0)
    coeffs_opd[0] = 0.0

    # Radius and Diameter
    out[f"{prefix}_radius"] = float(bfs_radius) if bfs_radius is not None else None
    out[f"{prefix}_diameter"] = float(diameter)

    # 36 Zernike coefficients
    for j in range(_OPD_N_EXPORT_COEFS):
        out[f"{prefix}_wfcoeff_{j}"] = float(coeffs_opd[j])

    # Derived metrics
    derived = _opd_derived_metrics(
        coeffs_opd, bfs_radius, fitting_radius, corneal_power=corneal_power
    )
    out[f"{prefix}_cyl"] = derived["cyl"]
    out[f"{prefix}_cylax"] = derived["cylax"]
    out[f"{prefix}_lsa"] = derived["lsa"]
    out[f"{prefix}_lca"] = derived["lca"]
    out[f"{prefix}_hoa_rms"] = derived["hoa_rms"]
    out[f"{prefix}_cornealpower"] = derived["cornealpower"]

    return out


def compute_opd_wavefront(raw_segments, metadata):
    """
    Compute the full 1,936-column OPD wavefront block from elevation maps.

    Fits Zernike polynomials at 11 diameters (2.0 to 7.0 mm) on both
    anterior and posterior elevation surfaces, with both pupil-centered
    and corneal-vertex-centered coordinate frames. Converts geometric
    Zernike coefficients to OPD using physical refractive indices, then
    derives cylinder, axis, LSA, LCA, HOA RMS, and corneal power.

    This reproduces the 1,936 columns the MS-39 exports under the
    "OPD Wavefront" category in the global CSV (columns 17--1952).

    Layout (4 blocks x 11 diameters x 44 params = 1,936):
      - opd_front_{d}mm_{param}:    anterior, pupil-centered
      - opd_total_{d}mm_{param}:    anterior+posterior, pupil-centered
      - opd_cv_front_{d}mm_{param}: anterior, corneal-vertex centered
      - opd_cv_total_{d}mm_{param}: anterior+posterior, CV centered

    Per diameter (44 values):
      Radius, Diameter, WFCoeff[0..35], Cyl, CylAx, LSA, LCA,
      HOA RMS, CornealPower

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of core.parse_csv(). Must contain 'elevation_anterior'
        and/or 'elevation_posterior' with shape (n_rows, 256).
        Values are in mm with -1000 as missing sentinel.
    metadata : dict
        Must contain 'PupilCX' and 'PupilCY' (pupil center offset in
        mm from corneal vertex) for CV centering. If absent, CV blocks
        will use (0, 0) offset (i.e., same as pupil-centered).

    Returns
    -------
    dict
        ~1,936 key-value pairs. Values are float or None where data
        is insufficient for fitting at a given diameter.
    """
    out = {}

    elev_ant = raw_segments.get("elevation_anterior")
    elev_post = raw_segments.get("elevation_posterior")

    # Pupil center offset for CV centering.
    # The raw polar maps are approximately pupil-centered. For CV-centered
    # analysis, shift by (-PupilCX, -PupilCY).
    pupil_cx = float(metadata.get("PupilCX", 0.0) or 0.0)
    pupil_cy = float(metadata.get("PupilCY", 0.0) or 0.0)

    # Build KD-trees once, reuse across all 11 diameters x 4 blocks
    _biquad_cache = {}

    for d in _OPD_DIAMETERS:
        d_label = _opd_diameter_label(d)
        fitting_radius = d / 2.0

        # =============================================================
        # PUPIL-CENTERED: ray-traced front + dual-surface total
        # Maps are VERTEX-CENTERED. For pupil-centered analysis,
        # shift the ray grid from vertex to pupil: offset=(-CX,-CY).
        # =============================================================
        # Front: single-surface ray-trace (air -> cornea)
        coeffs_front_pc, fz_front_pc = _opd_raytrace_one_diameter(
            elev_ant,
            fitting_radius,
            N_AIR,
            N_CORNEA,
            offset_x=-pupil_cx,
            offset_y=-pupil_cy,
            _cache=_biquad_cache,
        )
        # Front corneal power from ray-traced focal distance
        # CSO formula: P = n * 1000 / (-Fz).  Fz is positive in our
        # convention (light travels +z), so -Fz gives a negative power
        # for a converging surface.
        cp_front_pc = None
        if fz_front_pc is not None and abs(fz_front_pc) > 1e-6:
            cp_front_pc = N_CORNEA * 1000.0 / (-fz_front_pc)

        out.update(
            _opd_emit_one_block(
                f"opd_front_{d_label}",
                coeffs_front_pc,
                fitting_radius,  # Radius column = fitting radius
                1.0,  # delta_n=1.0: coeffs are already OPD
                fitting_radius,
                d,
                corneal_power=cp_front_pc,
            )
        )

        # Total = full dual-surface ray-trace (air -> cornea -> aqueous)
        coeffs_total_pc, fz_total_pc = _opd_raytrace_dual_surface(
            elev_ant,
            elev_post,
            fitting_radius,
            offset_x=-pupil_cx,
            offset_y=-pupil_cy,
            _cache=_biquad_cache,
        )
        # Total corneal power: P = n_aqueous * 1000 / Fz
        # CSO uses P = 1000 * n_aqueous / (-Fz_cso) with their Fz negative.
        # In our convention Fz is positive, so no negation needed.
        cp_total_pc = None
        if fz_total_pc is not None and abs(fz_total_pc) > 1e-6:
            cp_total_pc = N_AQUEOUS * 1000.0 / fz_total_pc

        out.update(
            _opd_emit_one_block(
                f"opd_total_{d_label}",
                coeffs_total_pc,
                fitting_radius,
                1.0,
                fitting_radius,
                d,
                corneal_power=cp_total_pc,
            )
        )

        # =============================================================
        # CORNEAL-VERTEX CENTERED: ray-traced front + dual-surface total
        # Maps are VERTEX-CENTERED. CV analysis: no offset needed.
        # =============================================================
        coeffs_front_cv, fz_front_cv = _opd_raytrace_one_diameter(
            elev_ant,
            fitting_radius,
            N_AIR,
            N_CORNEA,
            offset_x=0.0,
            offset_y=0.0,
            _cache=_biquad_cache,
        )

        cp_front_cv = None
        if fz_front_cv is not None and abs(fz_front_cv) > 1e-6:
            cp_front_cv = N_CORNEA * 1000.0 / (-fz_front_cv)

        out.update(
            _opd_emit_one_block(
                f"opd_cv_front_{d_label}",
                coeffs_front_cv,
                fitting_radius,
                1.0,
                fitting_radius,
                d,
                corneal_power=cp_front_cv,
            )
        )

        # CV Total = full dual-surface ray-trace (air -> cornea -> aqueous)
        coeffs_total_cv, fz_total_cv = _opd_raytrace_dual_surface(
            elev_ant,
            elev_post,
            fitting_radius,
            offset_x=0.0,
            offset_y=0.0,
            _cache=_biquad_cache,
        )
        cp_total_cv = None
        if fz_total_cv is not None and abs(fz_total_cv) > 1e-6:
            cp_total_cv = N_AQUEOUS * 1000.0 / fz_total_cv

        out.update(
            _opd_emit_one_block(
                f"opd_cv_total_{d_label}",
                coeffs_total_cv,
                fitting_radius,
                1.0,
                fitting_radius,
                d,
                corneal_power=cp_total_cv,
            )
        )

    return out


# ---------------------------------------------------------------------------
# PUBLISHED SCREENING INDICES (peer-reviewed, non-proprietary)
# ---------------------------------------------------------------------------
#
# These indices come from the published ophthalmology literature and can be
# computed from the same polar-map data that the MS-39 exports.  They are
# independent of CSO's proprietary screening indices above.
# ---------------------------------------------------------------------------


def _compute_is_value(sag_ant_map):
    """
    Rabinowitz I-S value: inferior minus superior mean keratometric
    power at the 3 mm zone.

    Reference: Rabinowitz YS, Survey of Ophthalmology 1998;42:297-319.
    Thresholds: >1.4 D suspect, >1.9 D keratoconus.

    Parameters
    ----------
    sag_ant_map : np.ndarray, shape (n_rows, 256)
        Sagittal anterior curvature polar map. Values are radii of
        curvature in mm; -1000 marks missing data.

    Returns
    -------
    float or None
        I-S value in diopters, or None if insufficient data.
    """
    clean = sag_ant_map.astype(np.float64)
    clean[clean == _MISSING] = np.nan
    n_rows = clean.shape[0]

    # Use rings at r ~ 1.5 mm (3 mm diameter).  Preferred rows 7-8
    # (1.4-1.6 mm), but severe KC eyes with poor fixation may have
    # these rows missing.  Fall back to wider band [6..10] (1.2-2.0 mm)
    # to avoid returning None for the most pathological eyes.
    preferred_rows = [7, 8]
    fallback_rows = [r for r in range(6, 11) if r < n_rows]

    # Inferior sector: 225-315 deg TABO (columns 160-224 for 256 meridians)
    inf_cols = list(range(160, 224))
    # Superior sector: 45-135 deg TABO (columns 32-96)
    sup_cols = list(range(32, 96))

    for rows in (preferred_rows, fallback_rows):
        inf_vals = []
        sup_vals = []
        for row in rows:
            if row >= n_rows:
                continue
            for j in inf_cols:
                v = clean[row, j]
                if not np.isnan(v) and v > 0:
                    inf_vals.append(337.5 / v)  # radius to diopters
            for j in sup_cols:
                v = clean[row, j]
                if not np.isnan(v) and v > 0:
                    sup_vals.append(337.5 / v)

        if len(inf_vals) >= 20 and len(sup_vals) >= 20:
            return float(np.mean(inf_vals) - np.mean(sup_vals))

    return None


def _compute_artmax(thk_map, thk_min_x, thk_min_y):
    """
    ARTmax = ThkMin / max(PPI).

    PPI = pachymetric progression index: rate of thickness increase
    from the thinnest point outward, per ring.

    Reference: Ambrósio R Jr et al., Am J Ophthalmol 2006;141:474-482.
    Threshold: <339 subclinical KC, <250 clinical KC.
    """
    clean = thk_map.astype(np.float64)
    clean[clean == _MISSING] = np.nan

    thk_min_val = np.nanmin(clean)
    if np.isnan(thk_min_val) or thk_min_val <= 0:
        return None, None  # artmax, max_ppi

    # Get CTSP centered on ThkMin
    ctsp = _compute_ctsp(clean, thk_min_x, thk_min_y)
    if ctsp is None:
        return None, None

    # Use ctsp[0] as the baseline.  If ctsp[0] is NaN (holes at the
    # cone center in severe KC), use thk_min_val as a close approximation
    # -- the CTSP center IS the thinnest point, so this is the best
    # substitute we have.
    baseline = ctsp[0] if not np.isnan(ctsp[0]) else thk_min_val

    # Compute PPI for each ring
    max_ppi = 0.0
    for i in range(1, len(ctsp)):
        r_mm = i * _R_STEP_MM
        if r_mm > 3.0:
            break
        if np.isnan(ctsp[i]):
            continue  # CSO skips NaN rings
        ppi = (ctsp[i] - baseline) / r_mm  # µm per mm
        if ppi > max_ppi:
            max_ppi = ppi

    if max_ppi <= 0:
        return None, None

    artmax = thk_min_val / max_ppi
    return float(artmax), float(max_ppi)


def _compute_epithelial_donut(epi_map):
    """
    Epithelial Donut Index: ratio of minimum epithelial thickness
    to the maximum of the surrounding annular mean.

    In keratoconus, epithelium thins over the cone (center of donut)
    and thickens around it (donut ring). A low ratio indicates
    epithelial compensation for stromal ectasia.

    Reference concept: Reinstein DZ et al., J Refract Surg 2009.

    Returns
    -------
    donut_ratio : float or None
        EpiThkMin / max(annular_mean). Normal ~0.85-0.95, KC <0.80.
    donut_delta : float or None
        max(annular_mean) - EpiThkMin in microns. Normal <8, KC >10.
    """
    clean = epi_map.astype(np.float64)
    clean[clean == _MISSING] = np.nan

    # Find epithelial minimum
    epi_min_val, epi_min_x, epi_min_y = _find_extremum(epi_map, mode="min")
    if epi_min_val is None or epi_min_val <= 0:
        return None, None

    n_rows, n_cols = clean.shape

    # Compute mean epithelial thickness in concentric annuli
    # centered on the epithelial minimum
    annular_means = []
    for ring_r in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ring_width = 0.5  # mm
        r_inner = ring_r - ring_width / 2
        r_outer = ring_r + ring_width / 2

        vals = []
        for i in range(n_rows):
            r_mm = i * _R_STEP_MM
            for j in range(n_cols):
                if np.isnan(clean[i, j]):
                    continue
                x = r_mm * _COS_THETA[j]
                y = r_mm * _SIN_THETA[j]
                dist = np.sqrt((x - epi_min_x) ** 2 + (y - epi_min_y) ** 2)
                if r_inner <= dist < r_outer:
                    vals.append(clean[i, j])

        if len(vals) >= 20:
            annular_means.append(float(np.mean(vals)))

    if len(annular_means) == 0:
        return None, None

    max_annular = max(annular_means)

    donut_ratio = epi_min_val / max_annular
    donut_delta = max_annular - epi_min_val

    return float(donut_ratio), float(donut_delta)


def _compute_srax(sag_ant_map):
    """
    Skewed Radial Axes (SRAX) index.

    Finds the steepest radial axis in the superior hemicircle and the
    steepest radial axis in the inferior hemicircle of the anterior
    sagittal curvature map at the ~3 mm zone (same ring as SimK).
    For a normal cornea, these axes are roughly 180 degrees apart.
    SRAX measures the deviation from that expected alignment.

    Reference: Rabinowitz YS & McDonnell PJ, Invest Ophthalmol Vis
    Sci 1989;30:2033-2038 (as part of the KISA% index).

    Parameters
    ----------
    sag_ant_map : np.ndarray, shape (n_rows, 256)
        Sagittal anterior curvature polar map.  Values are radii of
        curvature in mm; -1000 marks missing data.  Smaller radius =
        steeper curvature.

    Returns
    -------
    float or None
        SRAX in degrees, or None if insufficient data.
    """
    clean = sag_ant_map.astype(np.float64)
    clean[clean == _MISSING] = np.nan
    n_rows_map = clean.shape[0]
    n_cols = clean.shape[1]

    # Use rings at r ~ 1.5 mm (3 mm diameter).  Preferred rows 7-8
    # (1.4-1.6 mm), fall back to [6..10] for severe KC with missing rings.
    preferred_rows = [7, 8]
    fallback_rows = [r for r in range(6, 11) if r < n_rows_map]

    # Superior hemicircle: 45-135 deg TABO = columns 32-96
    sup_cols = np.arange(32, 96)
    # Inferior hemicircle: 225-315 deg TABO = columns 160-224
    inf_cols = np.arange(160, 224)

    for rows in (preferred_rows, fallback_rows):
        # Build a meridional curvature profile in diopters at the 3 mm zone
        profile = np.full(n_cols, np.nan, dtype=np.float64)
        for j in range(n_cols):
            vals = []
            for row in rows:
                if row < n_rows_map:
                    v = clean[row, j]
                    if not np.isnan(v) and v > 0:
                        vals.append(337.5 / v)  # radius to diopters
            if vals:
                profile[j] = np.mean(vals)

        sup_valid = profile[sup_cols]
        inf_valid = profile[inf_cols]

        if np.sum(~np.isnan(sup_valid)) >= 10 and np.sum(~np.isnan(inf_valid)) >= 10:
            break
    else:
        return None

    # Find steepest meridian in each hemicircle (highest diopter value)
    sup_best_local = int(np.nanargmax(sup_valid))
    inf_best_local = int(np.nanargmax(inf_valid))

    sup_best_col = sup_cols[sup_best_local]
    inf_best_col = inf_cols[inf_best_local]

    # Convert column indices to degrees
    sup_axis_deg = sup_best_col * 360.0 / n_cols
    inf_axis_deg = inf_best_col * 360.0 / n_cols

    # Angular difference between the two steepest axes
    angle_diff = abs(sup_axis_deg - inf_axis_deg)

    # SRAX = deviation from 180 degrees
    srax = abs(angle_diff - 180.0)

    return round(float(srax), 2)


def _compute_kisa_pct(sag_ant_map):
    """
    KISA% composite keratoconus screening index.

    Combines four topographic parameters into a single discriminant
    (CSO formula)::

        srax_factor = SRAX if AST > 1 else 1
        KISA% = max(1, K-47.2) * max(1, |I-S|) * max(1, AST) * max(1, srax_factor) / 3

    where K is central keratometric power (D), I-S is the
    inferior-superior value (D), AST is corneal astigmatism (D),
    and SRAX is the skewed radial axes index (degrees).

    Reference: Rabinowitz YS & McDonnell PJ, Invest Ophthalmol Vis
    Sci 1989;30:2033-2038.
    Thresholds: >60% suspect, >100% keratoconus.

    Parameters
    ----------
    sag_ant_map : np.ndarray, shape (n_rows, 256)
        Sagittal anterior curvature polar map.  Values are radii of
        curvature in mm; -1000 marks missing data.

    Returns
    -------
    kisa_pct : float or None
        KISA% value, or None if any component cannot be computed.
    srax : float or None
        SRAX component value in degrees, or None.
    """
    # Compute the individual components
    is_val = _compute_is_value(sag_ant_map)
    srax = _compute_srax(sag_ant_map)

    if is_val is None or srax is None:
        return None, srax

    # Compute K (central keratometric power) and AST from the same
    # 3 mm zone profile used by SimK.
    clean = sag_ant_map.astype(np.float64)
    clean[clean == _MISSING] = np.nan
    n_rows, n_cols = clean.shape

    # Build the SimK radial band profile.  Preferred band [1.25, 2.05] mm,
    # fallback to wider [1.0, 2.5] mm for severe KC with missing rings.
    valid = None
    for r_lo, r_hi in ((1.25, 2.05), (1.0, 2.5)):
        profile_sum = np.zeros(n_cols, dtype=np.float64)
        profile_cnt = np.zeros(n_cols, dtype=np.int64)
        for i in range(1, n_rows):
            r = i * _R_STEP_MM
            if r < r_lo or r > r_hi:
                continue
            for j in range(n_cols):
                val = clean[i, j]
                if not np.isnan(val):
                    profile_sum[j] += val
                    profile_cnt[j] += 1

        valid = profile_cnt > 0
        if np.sum(valid) >= 128:  # need at least half the meridians
            break
    else:
        return None, srax

    profile = np.full(n_cols, np.nan, dtype=np.float64)
    profile[valid] = profile_sum[valid] / profile_cnt[valid]

    # Convert profile from radius (mm) to diopters for K computation
    profile_d = np.full(n_cols, np.nan, dtype=np.float64)
    pos = valid & (profile > 0)
    profile_d[pos] = 337.5 / profile[pos]

    if np.sum(~np.isnan(profile_d)) < 128:
        return None, srax

    # K = mean keratometric power (KAvg equivalent)
    k_avg = float(np.nanmean(profile_d))

    # AST = |Ksteep - Kflat| = range of keratometric power
    k_max = float(np.nanmax(profile_d))
    k_min = float(np.nanmin(profile_d))
    ast = abs(k_max - k_min)

    # KISA% (CSO formula): each component is clamped to min 1.
    # SRAX is only included when AST > 1 D; otherwise treated as 1.
    srax_factor = srax if ast > 1.0 else 1.0
    kisa = (
        max(1.0, k_avg - 47.2) * max(1.0, abs(is_val)) * max(1.0, ast) * max(1.0, srax_factor)
    ) / 3.0

    return round(float(kisa), 4), srax


def _elevation_bfs_deviation(elev_map, sample_r_mm, sample_theta_rad, fitting_radius=4.0):
    """
    BFS elevation deviation at a specific polar location.

    1. Remove the apex offset (z at h=0) so the BFS formula
       z = R - sqrt(R^2 - h^2) applies correctly.  The anterior
       map has z=0 at apex; the posterior map has z = corneal
       thickness at vertex (~0.55-0.60 mm), which must be subtracted.
    2. Convert the elevation polar map to Cartesian (x, y, z).
    3. Fit a 1-parameter axis-centered BFS.
    4. Return the deviation (raw - BFS) at the sample point, in um.

    Parameters
    ----------
    elev_map : np.ndarray, shape (n_rows, 256)
        Elevation polar map in mm, with -1000 for missing.
    sample_r_mm : float
        Radial distance of the sampling point (mm).
    sample_theta_rad : float
        Angular position of the sampling point (radians).
    fitting_radius : float
        Radius (mm) within which to fit the BFS.  Default 4.0 mm
        (8 mm diameter), matching the BAD-D enhanced BFS zone.

    Returns
    -------
    float or None
        Elevation deviation in micrometers, or None on failure.
    """
    clean = _clean_polar_map(elev_map)

    # Determine apex offset: mean elevation at row 0 (h = 0)
    apex_vals = clean[0, :]
    apex_vals = apex_vals[~np.isnan(apex_vals)]
    if len(apex_vals) == 0:
        return None
    z0 = float(np.mean(apex_vals))

    # Convert to Cartesian within fitting zone
    x, y, z = _polar_elevation_to_cartesian(elev_map, fitting_radius)
    if len(x) < _ZERNIKE_MIN_POINTS:
        return None

    # Subtract apex offset so z=0 at vertex (required for 1-param BFS)
    z_centered = z - z0

    # Fit 1-parameter axis-centered BFS: R = sum((h^2+z^2)*z) / (2*sum(z^2))
    h2 = x**2 + y**2
    z2 = z_centered**2
    denom = 2.0 * np.sum(z2)
    if denom < 1e-30:
        return None
    bfs_R = float(np.sum((h2 + z2) * z_centered) / denom)

    # Sanity: R should be in physiological range.  Normal cornea is
    # 7-9 mm; severe KC can have BFS R as low as ~3 mm (Kmax 112D);
    # very flat post-refractive corneas can reach ~12 mm.  Use a wide
    # range to avoid rejecting pathological but valid eyes.
    if bfs_R < 2.0 or bfs_R > 25.0:
        return None

    # Sample raw elevation at the target location
    raw_elev = _bilinear_interp_polar(clean, sample_r_mm, sample_theta_rad)
    if np.isnan(raw_elev):
        return None
    raw_elev_centered = raw_elev - z0

    # BFS height at the sample point
    h2_sample = sample_r_mm**2
    disc = bfs_R**2 - h2_sample
    if disc <= 0:
        return None
    bfs_z = bfs_R - math.sqrt(disc)

    # Deviation in micrometers (raw - BFS)
    deviation_um = (raw_elev_centered - bfs_z) * 1000.0
    return deviation_um


def _compute_bad_d_components(raw_segments, metadata):
    """
    Extract the 5 BAD-D components for scoring.

    BAD-D (Belin/Ambrosio Enhanced Ectasia Display - Discriminant) adapted
    for the CSO MS-39 AS-OCT.  The five core components mirror the Pentacam
    BAD-D structure (Ambrosio et al., Am J Ophthalmol 2011) but are extracted
    from MS-39 polar maps.

    Components (all computed at the thinnest corneal point):
      Df: anterior elevation BFS deviation (um)
      Db: posterior elevation BFS deviation (um)
      Dp: pachymetric progression (max PPI, um/mm)
      Dt: thinnest pachymetry (um)
      Da: anterior radius of curvature (mm, from ABCD parameter A)

    Returns raw component values as a dict.  Z-score normalization and
    discriminant weights require the normative database (Phase B).
    """
    result = {"df": None, "db": None, "dp": None, "dt": None, "da": None}

    # ---- Locate thinnest point from corneal_thickness map ----
    thk_map = raw_segments.get("corneal_thickness")
    if thk_map is None:
        return result
    thk_min_val, thk_min_x, thk_min_y = _find_extremum(thk_map, mode="min")
    if thk_min_val is None:
        return result

    # ---- Dt: thinnest pachymetry (um) ----
    result["dt"] = float(thk_min_val)

    # ---- Convert ThkMin location to polar coordinates ----
    thk_min_r = math.sqrt(thk_min_x**2 + thk_min_y**2)
    thk_min_theta = math.atan2(thk_min_y, thk_min_x)
    if thk_min_theta < 0:
        thk_min_theta += 2 * math.pi

    # ---- Df: anterior elevation BFS deviation at ThkMin (um) ----
    elev_ant = raw_segments.get("elevation_anterior")
    if elev_ant is not None:
        df = _elevation_bfs_deviation(elev_ant, thk_min_r, thk_min_theta)
        if df is not None:
            result["df"] = float(df)

    # ---- Db: posterior elevation BFS deviation at ThkMin (um) ----
    elev_post = raw_segments.get("elevation_posterior")
    if elev_post is not None:
        db = _elevation_bfs_deviation(elev_post, thk_min_r, thk_min_theta)
        if db is not None:
            result["db"] = float(db)

    # ---- Dp: max PPI (um/mm) -- reuse CTSP from ARTmax computation ----
    clean_thk = _clean_polar_map(thk_map)
    ctsp = _compute_ctsp(clean_thk, thk_min_x, thk_min_y)
    if ctsp is not None:
        # Use ctsp[0] if valid, otherwise thk_min_val (same fallback as ARTmax)
        baseline = ctsp[0] if not np.isnan(ctsp[0]) else float(thk_min_val)
        max_ppi = 0.0
        for i in range(1, len(ctsp)):
            r_mm = i * _R_STEP_MM
            if r_mm > 3.0:
                break
            if np.isnan(ctsp[i]):
                continue  # CSO skips NaN rings
            ppi = (ctsp[i] - baseline) / r_mm  # um per mm
            if ppi > max_ppi:
                max_ppi = ppi
        if max_ppi > 0:
            result["dp"] = float(max_ppi)

    # ---- Da: anterior radius of curvature at ThkMin (mm) ----
    # ABCD parameter A = trimmed-mean sagittal radius in a 1.5mm disc
    # centered on ThkMin (GetCurvAvg).
    sag_ant = raw_segments.get("sagittal_anterior")
    if sag_ant is not None:
        da = _trimmed_mean_in_disc(
            sag_ant, thk_min_x, thk_min_y, radius=1.5, alpha=0.05, min_count=100
        )
        if not np.isnan(da):
            result["da"] = float(da)

    return result


def _bad_d_score(components, normatives=None):
    """
    BAD-D discriminant score.  Requires normative database.

    Without normatives (Phase A): returns raw components only.
    With normatives (Phase B): returns Z-scores and final BAD-D value.

    Parameters
    ----------
    components : dict
        Raw BAD-D component values from _compute_bad_d_components().
    normatives : dict or None
        Dict with keys 'mu' and 'sigma', each a dict mapping component
        names ('df', 'db', 'dp', 'dt', 'da') to float values.
        Also 'weights' mapping component names to LDA coefficients,
        and 'constant' for the LDA intercept.

    Returns
    -------
    float or None
        BAD-D discriminant score, or None if normatives are unavailable
        or any required component is missing.
    """
    if normatives is None:
        return None  # Cannot compute without normatives
    # Future: Z-score each component, apply discriminant weights
    # z_i = (x_i - mu_i) / sigma_i
    # BAD-D = sum(w_i * z_i) + constant
    return None


def compute_published_indices(raw_segments: dict, metadata: dict) -> dict:
    """
    Published, peer-reviewed screening indices (not CSO-proprietary).

    Parameters
    ----------
    raw_segments : dict[str, np.ndarray]
        Output of core.parse_csv().
    metadata : dict
        Exam metadata (currently unused, reserved for future indices).

    Returns
    -------
    dict
        Keys:
          published_is_value          -- Rabinowitz I-S value (D), or None
          published_artmax            -- ARTmax (µm), or None
          published_max_ppi           -- max PPI (µm/mm), or None
          published_simax_front       -- SImax anterior (D), or None
          published_simax_front_angle -- angle of max asymmetry (deg), or None
          published_epi_donut_ratio   -- Epithelial Donut ratio, or None
          published_epi_donut_delta   -- Epithelial Donut delta (µm), or None
          published_srax              -- SRAX index (degrees), or None
          published_kisa_pct          -- KISA% composite index, or None
          published_pti_population_adapted -- Population-adapted PTI, or None
          published_bad_d_df           -- BAD-D Df: anterior BFS deviation (um)
          published_bad_d_db           -- BAD-D Db: posterior BFS deviation (um)
          published_bad_d_dp           -- BAD-D Dp: max PPI (um/mm)
          published_bad_d_dt           -- BAD-D Dt: thinnest pachymetry (um)
          published_bad_d_da           -- BAD-D Da: anterior ARC at ThkMin (mm)
          published_bad_d_score        -- BAD-D score (None until Phase B)
    """
    out = {}
    sag_ant = raw_segments.get("sagittal_anterior")
    if sag_ant is not None:
        out["published_is_value"] = _compute_is_value(sag_ant)
        kisa, srax = _compute_kisa_pct(sag_ant)
        out["published_srax"] = srax
        out["published_kisa_pct"] = kisa
    else:
        out["published_is_value"] = None
        out["published_srax"] = None
        out["published_kisa_pct"] = None

    # ARTmax (Ambrósio Relational Thickness Maximum)
    thk_map = raw_segments.get("corneal_thickness")
    if thk_map is not None:
        thk_val, thk_x, thk_y = _find_extremum(thk_map, mode="min")
        if thk_val is not None:
            artmax, max_ppi = _compute_artmax(thk_map, thk_x, thk_y)
            out["published_artmax"] = artmax
            out["published_max_ppi"] = max_ppi
        else:
            out["published_artmax"] = None
            out["published_max_ppi"] = None
    else:
        out["published_artmax"] = None
        out["published_max_ppi"] = None

    # SImax (Maximum Symmetry Index) -- location-agnostic SI
    if sag_ant is not None:
        simax_val, simax_angle = _compute_simax(sag_ant, _DELTA_N_ANTERIOR)
        out["published_simax_front"] = simax_val
        out["published_simax_front_angle"] = simax_angle
    else:
        out["published_simax_front"] = None
        out["published_simax_front_angle"] = None

    # Epithelial Donut Index (Reinstein et al. 2009 concept)
    epi_map = raw_segments.get("epithelial_thickness")
    if epi_map is not None:
        donut_ratio, donut_delta = _compute_epithelial_donut(epi_map)
        out["published_epi_donut_ratio"] = donut_ratio
        out["published_epi_donut_delta"] = donut_delta
    else:
        out["published_epi_donut_ratio"] = None
        out["published_epi_donut_delta"] = None

    # Population-adapted PTI (requires normative database from 200k exams)
    out["published_pti_population_adapted"] = None  # Requires normative database

    # BAD-D (Belin/Ambrosio Enhanced Ectasia Display - Discriminant)
    bad_d = _compute_bad_d_components(raw_segments, metadata)
    out["published_bad_d_df"] = bad_d.get("df")
    out["published_bad_d_db"] = bad_d.get("db")
    out["published_bad_d_dp"] = bad_d.get("dp")
    out["published_bad_d_dt"] = bad_d.get("dt")
    out["published_bad_d_da"] = bad_d.get("da")
    out["published_bad_d_score"] = _bad_d_score(bad_d)  # None until Phase B

    return out
