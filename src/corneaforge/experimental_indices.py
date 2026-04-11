"""
Experimental corneal indices — ML features only.

WARNING: All outputs from this module are EXPERIMENTAL.  They are
100% ML-driven features intended for XGBoost/CatBoost model selection.
They are NOT validated clinical indices and MUST NOT be used for
clinical decision-making.  All CSV column names end with
``_experimental`` to make this unambiguous.

Pipeline:
  1. Algebraic conoid fit (Langenbucher quadric, 9 params, SVD)
     → geometric features: CRx, CRy, CQx, CQy, eigenvalue pattern
  2. Zernike decomposition of conoid residuals (order 8, 45 terms)
     → irregularity features (NOT the same as CSO clinical Zernike)
  3. Conoid-referenced OPD ray tracing
     → optical features: measured/conoid/residual wavefront

Reference
---------
Langenbucher A, Sauer T, Seitz B.
"Conoidal fitting of corneal topography height data."
J Refract Surg. 1999;15(S2):S240-S242.
"""

import logging
import math

import numpy as np
from scipy.optimize import least_squares

from corneaforge.computed_indices import (
    _R_STEP_MM,
    _ZERNIKE_MAX_ORDER,
    _ZERNIKE_MIN_POINTS,
    N_AIR,
    N_AQUEOUS,
    N_CORNEA,
    _biquad_eval_batch,
    _fit_zernike_coefficients,
    _polar_elevation_to_cartesian,
    create_ray_grid,
    estimate_focal_point,
    snells_law_vector_batch,
    surface_normals_from_gradients,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_quadric_surface(x, y, z):
    """
    Fit a general quadric surface with a33=1 normalization.

    General quadric: a11*X^2 + a22*Y^2 + a33*Z^2 + a12*XY + a13*XZ
                   + a23*YZ + b1*X + b2*Y + b3*Z + c = 0

    With a33 = 1, rearranged so that the RHS is -Z^2.

    Parameters
    ----------
    x, y, z : np.ndarray
        Cartesian coordinates of data points (mm).

    Returns
    -------
    coeffs : np.ndarray (9,) or None
        [a11, a22, a12, a13, a23, b1, b2, b3, c], or None on failure.
    """
    if len(x) < _ZERNIKE_MIN_POINTS:
        return None

    # Design matrix: [X^2, Y^2, XY, XZ, YZ, X, Y, Z, 1]
    design = np.column_stack(
        [
            x * x,
            y * y,
            x * y,
            x * z,
            y * z,
            x,
            y,
            z,
            np.ones(len(x)),
        ]
    )

    rhs = -(z * z)

    try:
        coeffs, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(coeffs)):
        return None

    return coeffs


def _extract_conoid_params(quadric_coeffs):
    """
    Eigendecompose quadric matrix to extract conoid geometry.

    Parameters
    ----------
    quadric_coeffs : np.ndarray (9,)
        [a11, a22, a12, a13, a23, b1, b2, b3, c]

    Returns
    -------
    params : dict or None
        semi_axes (sa1 >= sa2, sa3=z-like), center, apex, rotation_matrix,
        curvatures, asphericity, Euler angles.  None on failure.
    """
    a11, a22, a12, a13, a23, b1, b2, b3, c = quadric_coeffs

    # Assemble 3x3 symmetric matrix (a33 = 1)
    A = np.array(
        [
            [a11, a12 / 2, a13 / 2],
            [a12 / 2, a22, a23 / 2],
            [a13 / 2, a23 / 2, 1.0],
        ]
    )

    b_vec = np.array([b1, b2, b3])

    # Eigendecompose (eigh for real symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # All eigenvalues must be positive for an ellipsoid
    if np.any(eigenvalues <= 0):
        return None

    # Center of quadric: X0 = -A^{-1} b / 2
    try:
        X0 = -np.linalg.solve(A, b_vec) / 2.0
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(X0)):
        return None

    # Constant after centering
    d = X0 @ A @ X0 + b_vec @ X0 + c

    # For valid ellipsoid: d < 0
    if d >= 0:
        return None

    # Semi-axes
    sa = np.sqrt(-d / eigenvalues)  # (3,)

    # Identify the Z-like axis: eigenvector most aligned with [0,0,1]
    z_hat = np.array([0.0, 0.0, 1.0])
    alignment = np.abs(eigenvectors.T @ z_hat)  # dot product per eigenvector
    z_idx = int(np.argmax(alignment))

    # Build sorted index: transverse axes first, then Z-like
    other_idx = [i for i in range(3) if i != z_idx]
    # Convention: sa1 >= sa2 (sa1 = flat meridian, sa2 = steep meridian)
    if sa[other_idx[0]] < sa[other_idx[1]]:
        other_idx = [other_idx[1], other_idx[0]]

    sorted_idx = [other_idx[0], other_idx[1], z_idx]
    sa_sorted = sa[sorted_idx]
    ev_sorted = eigenvectors[:, sorted_idx]  # columns are sorted eigenvectors

    sa1, sa2, sa3 = sa_sorted

    # Ensure the Z-like eigenvector points in +Z direction
    # (the corneal surface convention: Z increases into the eye)
    if ev_sorted[2, 2] < 0:
        ev_sorted[:, 2] = -ev_sorted[:, 2]

    # Ensure right-handed coordinate system
    cross = np.cross(ev_sorted[:, 0], ev_sorted[:, 1])
    if np.dot(cross, ev_sorted[:, 2]) < 0:
        ev_sorted[:, 1] = -ev_sorted[:, 1]

    # Radii of curvature (mm)
    CRx = sa1**2 / sa3
    CRy = sa2**2 / sa3

    # Asphericity (conic constant Q)
    CQx = sa1**2 / sa3**2 - 1.0
    CQy = sa2**2 / sa3**2 - 1.0

    # Apex position: closest point on ellipsoid to observer (min Z).
    # In principal coords, apex_principal = [0, 0, -sa3].
    # In original coords: apex = X0 + U @ apex_principal
    apex_principal = np.array([0.0, 0.0, -sa3])
    apex = X0 + ev_sorted @ apex_principal

    # Euler angles from rotation matrix U (columns = principal axes)
    e3 = ev_sorted[:, 2]  # Z-like eigenvector
    e1 = ev_sorted[:, 0]  # flat-meridian eigenvector

    alpha_deg = math.degrees(math.asin(np.clip(-e3[1], -1.0, 1.0)))  # tilt around X
    beta_deg = math.degrees(math.asin(np.clip(e3[0], -1.0, 1.0)))  # tilt around Y
    gamma_deg = math.degrees(math.atan2(e1[1], e1[0])) % 180.0  # astigmatic axis

    return {
        "semi_axes": sa_sorted,  # (sa1, sa2, sa3) in mm
        "center": X0,  # (3,) mm
        "apex": apex,  # (3,) mm
        "rotation_matrix": ev_sorted,  # (3,3) columns = principal axes
        "eigenvalues": eigenvalues[sorted_idx],
        "d": d,
        "CRx": CRx,
        "CRy": CRy,
        "CQx": CQx,
        "CQy": CQy,
        "tilt_alpha": alpha_deg,
        "tilt_beta": beta_deg,
        "axis_gamma": gamma_deg,
    }


def _transform_to_canonical(x, y, z, conoid_params):
    """
    Transform data points to conoid-canonical coordinates.

    Parameters
    ----------
    x, y, z : np.ndarray
        Original Cartesian coordinates (mm).
    conoid_params : dict
        Output of _extract_conoid_params.

    Returns
    -------
    Xc, Yc, Zc : np.ndarray
        Canonical coordinates (mm).
    """
    U = conoid_params["rotation_matrix"]
    X0 = conoid_params["center"]

    pts = np.column_stack([x, y, z])  # (N, 3)
    canonical = (pts - X0) @ U  # (N, 3)

    return canonical[:, 0], canonical[:, 1], canonical[:, 2]


def _conoid_residuals(Xc, Yc, Zc, conoid_params):
    """
    Reconstruct conoid surface and compute residuals.

    Parameters
    ----------
    Xc, Yc, Zc : np.ndarray
        Canonical coordinates (mm).
    conoid_params : dict
        Output of _extract_conoid_params.

    Returns
    -------
    delta_z : np.ndarray
        Residuals Zc - Zc_recon (mm).
    """
    sa1, sa2, sa3 = conoid_params["semi_axes"]

    # Argument of sqrt; clip to >= 0 for points near the edge
    arg = 1.0 - (Xc / sa1) ** 2 - (Yc / sa2) ** 2
    arg = np.clip(arg, 0.0, None)

    # Negative sign: corneal surface is in the -Z hemisphere of the ellipsoid
    Zc_recon = -sa3 * np.sqrt(arg)

    return Zc - Zc_recon


def _conoid_fit_one_surface(polar_map, fitting_radius, surface_type):
    """
    Full Langenbucher conoid pipeline for one surface.

    Parameters
    ----------
    polar_map : np.ndarray (n_rows, 256)
        Polar elevation map (mm), -1000 for missing.
    fitting_radius : float
        Analysis zone radius (mm).
    surface_type : str
        'anterior' or 'posterior'.

    Returns
    -------
    conoid_params : dict or None
        Conoid geometry parameters.
    zernike_coeffs_um : np.ndarray (45,) or None
        Zernike coefficients on conoid residuals (um).
    diagnostics : dict
        Residual RMS, sanity checks, etc.
    """
    diagnostics = {}

    if polar_map is None:
        return None, None, diagnostics

    # Step 1: polar -> Cartesian
    x, y, z = _polar_elevation_to_cartesian(polar_map, fitting_radius)

    if len(x) < _ZERNIKE_MIN_POINTS:
        return None, None, diagnostics

    # Step 2: algebraic quadric fit
    quadric_coeffs = _fit_quadric_surface(x, y, z)
    if quadric_coeffs is None:
        return None, None, diagnostics

    # Step 3-4: eigendecompose and extract conoid parameters
    conoid_params = _extract_conoid_params(quadric_coeffs)
    if conoid_params is None:
        return None, None, diagnostics

    # Store quadric coefficients in params for downstream use
    conoid_params["quadric_coeffs"] = quadric_coeffs

    # Step 5-6: compute conoid residuals in the ORIGINAL instrument frame.
    #
    # The canonical-frame approach (transform to eigenvector coordinates,
    # reconstruct conoid height there) is theoretically cleaner but breaks
    # on real data: corneal elevation maps cover only ~15-30° of arc, so
    # the off-diagonal quadric terms (a13, a23) are poorly constrained.
    # This makes the eigenvectors unreliable (false 30°+ tilts), which
    # poisons the canonical transform and thus the Zernike decomposition.
    #
    # Instead, we evaluate the conoid surface directly at the original
    # (x, y) positions using the quadric equation (same function the ray
    # tracer uses).  Residuals = z_measured - z_conoid, Zernike in the
    # instrument frame.  The conoid parameters (CRx, CRy, CQ, apex, tilt)
    # are still extracted from the eigendecomposition — they depend on
    # eigenvalues (well-conditioned) more than eigenvectors.
    query_xy = np.column_stack([x, y])
    # Always select the NEAR root (smaller Z) for residual computation.
    # Both anterior and posterior measured data sit on the observer-facing
    # side of their respective ellipsoids.  The "posterior" root (far side)
    # is only meaningful for ray tracing (rays exiting through the back).
    z_conoid, _, _, conoid_valid = _conoid_eval_surface(
        quadric_coeffs,
        query_xy,
        "anterior",
    )

    # Keep only points where the conoid evaluation succeeded
    valid = conoid_valid & np.isfinite(z_conoid)
    if np.sum(valid) < _ZERNIKE_MIN_POINTS:
        return None, None, diagnostics

    x_v, y_v, z_v = x[valid], y[valid], z[valid]
    z_con_v = z_conoid[valid]
    delta_z = z_v - z_con_v

    # Residual RMS in microns
    residual_rms_um = float(np.sqrt(np.mean(delta_z**2))) * 1000.0
    diagnostics["residual_rms_um"] = residual_rms_um

    # Steps 7-9: Zernike on residuals in the instrument frame
    zernike_coeffs_um, rms_fit = _fit_zernike_coefficients(
        x_v, y_v, delta_z, fitting_radius, max_order=_ZERNIKE_MAX_ORDER
    )

    if rms_fit is not None:
        diagnostics["fit_rms_um"] = rms_fit * 1000.0  # mm -> um
    else:
        diagnostics["fit_rms_um"] = None

    # Sanity checks from Zernike coefficients
    if zernike_coeffs_um is not None:
        c = zernike_coeffs_um
        # Tilt check: sqrt(c1^2 + c2^2) — j=1,2 are tilts
        diagnostics["tilt_check_um"] = float(np.sqrt(c[1] ** 2 + c[2] ** 2))
        # Defocus check: |c4| — j=4 is defocus (Z_2^0)
        diagnostics["defocus_check_um"] = float(abs(c[4]))
        # Astigmatism check: sqrt(c3^2 + c5^2) — j=3,5 are astigmatism
        diagnostics["astig_check_um"] = float(np.sqrt(c[3] ** 2 + c[5] ** 2))
    else:
        diagnostics["tilt_check_um"] = None
        diagnostics["defocus_check_um"] = None
        diagnostics["astig_check_um"] = None

    return conoid_params, zernike_coeffs_um, diagnostics


# ---------------------------------------------------------------------------
# Tilted biconic fit (8 parameters, Levenberg-Marquardt)
# ---------------------------------------------------------------------------
#
# Per Consejo 2021: posterior asphericity is significantly corrupted by
# tilt at all aperture radii (p<0.01).  Per Langenbucher 2023: conoid
# CQx/CQy are coupled through shared sa3 and have 96µm P95 uncertainty
# on the posterior surface.  The tilted biconic fixes both issues:
# independent BQx/BQy and explicit tilt parameters.
#
# 8 parameters: BRx, BRy, BQx, BQy, Bz0, gamma, alpha_x, alpha_y
# Fitted via scipy.optimize.least_squares (LM) with bounds.
# Initialized from the algebraic conoid when available.

_logger = logging.getLogger(__name__)

# Bounds for LM
_BIC_R_MIN, _BIC_R_MAX = 3.0, 15.0
_BIC_Q_MIN, _BIC_Q_MAX = -2.0, 2.0
_BIC_TILT_MAX = math.radians(15.0)  # ±15° max tilt
_BIC_Z0_MARGIN = 0.5  # mm around apex
_BIC_COND_WARN = 1e8  # Jacobian condition number threshold


def _biconic_surface(params, x, y):
    """
    Evaluate tilted biconic surface height at (x, y) points.

    Parameters: [BRx, BRy, BQx, BQy, Bz0, gamma, alpha_x, alpha_y]

    The tilt (alpha_x, alpha_y) rotates the coordinate system before
    evaluating the biconic.  gamma rotates around Z to align with
    the astigmatic axis.
    """
    BRx, BRy, BQx, BQy, Bz0, gamma, ax, ay = params

    # 1. Rotate by astigmatic axis (gamma) around Z
    cg, sg = np.cos(gamma), np.sin(gamma)
    xr = x * cg + y * sg
    yr = -x * sg + y * cg

    # 2. Apply tilt correction (small-angle: rotate the surface model,
    #    which is equivalent to counter-rotating the data).
    #    For small angles, the height correction is approximately:
    #    z_tilt ≈ alpha_x * yr + alpha_y * xr  (first-order tilt)
    #    This is subtracted from the biconic height.
    tilt_correction = ax * yr + ay * xr

    # 3. Evaluate biconic in the rotated frame
    term_x = xr * xr / BRx
    term_y = yr * yr / BRy
    numerator = term_x + term_y

    disc = 1.0 - (1.0 + BQx) * xr * xr / (BRx * BRx) - (1.0 + BQy) * yr * yr / (BRy * BRy)
    disc = np.maximum(disc, 1e-12)
    denominator = 1.0 + np.sqrt(disc)

    z_bic = Bz0 + numerator / denominator - tilt_correction
    return z_bic


def _biconic_residuals(params, x, y, z):
    """Residual vector for least_squares."""
    return z - _biconic_surface(params, x, y)


def _fit_biconic(x, y, z, conoid_params=None, apex_z=0.0):
    """
    Fit an 8-parameter tilted biconic surface via Levenberg-Marquardt.

    Parameters
    ----------
    x, y, z : np.ndarray
        Cartesian coordinates of data points (mm).
    conoid_params : dict or None
        Output of _extract_conoid_params, used for initialization.
    apex_z : float
        Elevation at the surface apex (mm).

    Returns
    -------
    result : dict or None
        BRx, BRy, BQx, BQy, Bz0, gamma, alpha_x, alpha_y,
        residual_rms_um, jacobian_cond.
    """
    if len(x) < _ZERNIKE_MIN_POINTS:
        return None

    # --- Initialization ---
    if conoid_params is not None:
        BRx0 = conoid_params["CRx"]
        BRy0 = conoid_params["CRy"]
        BQx0 = np.clip(conoid_params["CQx"], _BIC_Q_MIN, _BIC_Q_MAX)
        BQy0 = np.clip(conoid_params["CQy"], _BIC_Q_MIN, _BIC_Q_MAX)
        gamma0 = math.radians(conoid_params["axis_gamma"])
    else:
        # Fallback: crude sphere estimate
        h2 = x * x + y * y
        z_rc = z - apex_z
        z2 = z_rc * z_rc
        denom = 2.0 * np.sum(z2)
        R0 = float(np.sum((h2 + z2) * z_rc) / denom) if denom > 1e-30 else 7.8
        R0 = np.clip(R0, _BIC_R_MIN, _BIC_R_MAX)
        BRx0 = R0
        BRy0 = R0
        BQx0 = -0.2
        BQy0 = -0.2
        gamma0 = 0.0

    p0 = np.array([BRx0, BRy0, BQx0, BQy0, apex_z, gamma0, 0.0, 0.0])

    # --- Bounds ---
    lower = np.array(
        [
            _BIC_R_MIN,
            _BIC_R_MIN,
            _BIC_Q_MIN,
            _BIC_Q_MIN,
            apex_z - _BIC_Z0_MARGIN,
            -math.pi,
            -_BIC_TILT_MAX,
            -_BIC_TILT_MAX,
        ]
    )
    upper = np.array(
        [
            _BIC_R_MAX,
            _BIC_R_MAX,
            _BIC_Q_MAX,
            _BIC_Q_MAX,
            apex_z + _BIC_Z0_MARGIN,
            math.pi,
            _BIC_TILT_MAX,
            _BIC_TILT_MAX,
        ]
    )

    # Clip initial guess to bounds
    p0 = np.clip(p0, lower + 1e-10, upper - 1e-10)

    # --- Levenberg-Marquardt ---
    try:
        sol = least_squares(
            _biconic_residuals,
            p0,
            args=(x, y, z),
            method="trf",  # trust-region reflective (supports bounds)
            bounds=(lower, upper),
            max_nfev=500,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
    except Exception:
        return None

    if not sol.success and sol.status not in (1, 2, 3, 4):
        return None

    BRx, BRy, BQx, BQy, Bz0, gamma, ax, ay = sol.x

    # --- Residual RMS ---
    rms_mm = float(np.sqrt(np.mean(sol.fun**2)))

    # --- Jacobian condition number check ---
    jac_cond = None
    if sol.jac is not None:
        try:
            sv = np.linalg.svd(sol.jac, compute_uv=False)
            if sv[-1] > 0:
                jac_cond = float(sv[0] / sv[-1])
                if jac_cond > _BIC_COND_WARN:
                    _logger.warning(
                        "Biconic Jacobian condition number %.2e exceeds "
                        "threshold %.2e — fit may be unstable",
                        jac_cond,
                        _BIC_COND_WARN,
                    )
        except np.linalg.LinAlgError:
            pass

    # Normalize gamma to [0, 180) degrees
    gamma_deg = math.degrees(gamma) % 180.0

    return {
        "BRx": float(BRx),
        "BRy": float(BRy),
        "BQx": float(BQx),
        "BQy": float(BQy),
        "Bz0": float(Bz0),
        "gamma": gamma_deg,
        "alpha_x": math.degrees(ax),
        "alpha_y": math.degrees(ay),
        "residual_rms_um": rms_mm * 1000.0,
        "jacobian_cond": jac_cond,
    }


def _biconic_fit_one_surface(polar_map, fitting_radius, conoid_params=None):
    """
    Fit tilted biconic to one corneal surface and compute Zernike residuals.

    Returns (biconic_params, zernike_coeffs_um, diagnostics) or (None, ...).
    """
    diagnostics = {}
    if polar_map is None:
        return None, None, diagnostics

    x, y, z = _polar_elevation_to_cartesian(polar_map, fitting_radius)
    if len(x) < _ZERNIKE_MIN_POINTS:
        return None, None, diagnostics

    # Apex z from row 0
    row0 = polar_map[0].astype(np.float64)
    row0_valid = row0[row0 != -1000]
    apex_z = float(np.mean(row0_valid)) if len(row0_valid) > 0 else 0.0

    # Fit biconic
    bic = _fit_biconic(x, y, z, conoid_params=conoid_params, apex_z=apex_z)
    if bic is None:
        return None, None, diagnostics

    diagnostics["residual_rms_um"] = bic["residual_rms_um"]
    diagnostics["jacobian_cond"] = bic["jacobian_cond"]

    # Compute residuals in instrument frame
    params_vec = np.array(
        [
            bic["BRx"],
            bic["BRy"],
            bic["BQx"],
            bic["BQy"],
            bic["Bz0"],
            math.radians(bic["gamma"]),
            math.radians(bic["alpha_x"]),
            math.radians(bic["alpha_y"]),
        ]
    )
    z_bic = _biconic_surface(params_vec, x, y)
    delta_z = z - z_bic

    # Zernike on biconic residuals
    zernike_coeffs_um, rms_fit = _fit_zernike_coefficients(
        x, y, delta_z, fitting_radius, max_order=_ZERNIKE_MAX_ORDER
    )
    if rms_fit is not None:
        diagnostics["fit_rms_um"] = rms_fit * 1000.0

    # Sanity checks
    if zernike_coeffs_um is not None:
        c = zernike_coeffs_um
        diagnostics["tilt_check_um"] = float(np.sqrt(c[1] ** 2 + c[2] ** 2))
        diagnostics["defocus_check_um"] = float(abs(c[4]))
        diagnostics["astig_check_um"] = float(np.sqrt(c[3] ** 2 + c[5] ** 2))

    return bic, zernike_coeffs_um, diagnostics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_conoid_analysis(raw_segments, metadata, fitting_radius=4.0):
    """
    Langenbucher conoid analysis for both corneal surfaces.

    Parameters
    ----------
    raw_segments : dict
        Keys include 'elevation_anterior', 'elevation_posterior' (polar maps).
    metadata : dict
        Exam metadata (not used directly but kept for interface consistency).
    fitting_radius : float
        Analysis zone radius in mm (default 4.0).

    Returns
    -------
    dict
        Conoid parameters, Zernike coefficients on residuals, diagnostics.
        Keys follow the pattern conoid_{ant|post}_{param}.
        Internal conoid_params dicts stored as _conoid_{ant|post}_params.
    """
    result = {}
    _X = "_experimental"  # suffix for all CSV column names

    surface_configs = [
        ("ant", "anterior", "elevation_anterior"),
        ("post", "posterior", "elevation_posterior"),
    ]

    for prefix, surface_type, map_key in surface_configs:
        polar_map = raw_segments.get(map_key)

        conoid_params, zernike_coeffs_um, diagnostics = _conoid_fit_one_surface(
            polar_map, fitting_radius, surface_type
        )

        # Internal dicts (no suffix — not CSV columns, used by ray tracing)
        result[f"_conoid_{prefix}_params"] = conoid_params

        # --- Eigenvalue features (reuse quadric from conoid fit) ---
        qc = conoid_params["quadric_coeffs"] if conoid_params is not None else None
        if qc is None and polar_map is not None:
            # Conoid extraction failed (e.g. hyperboloid) but quadric fit
            # itself may have succeeded — compute eigenvalues from raw fit.
            x, y, z = _polar_elevation_to_cartesian(polar_map, fitting_radius)
            qc = _fit_quadric_surface(x, y, z) if len(x) >= _ZERNIKE_MIN_POINTS else None

        if qc is not None:
            A = np.array(
                [
                    [qc[0], qc[2] / 2, qc[3] / 2],
                    [qc[2] / 2, qc[1], qc[4] / 2],
                    [qc[3] / 2, qc[4] / 2, 1.0],
                ]
            )
            evals = np.sort(np.linalg.eigvalsh(A))
            n_pos = int(np.sum(evals > 0))
            result[f"conoid_{prefix}_n_positive_eigenvalues{_X}"] = n_pos
            result[f"conoid_{prefix}_eigenvalue_min{_X}"] = float(evals[0])
            result[f"conoid_{prefix}_eigenvalue_max{_X}"] = float(evals[2])
            result[f"conoid_{prefix}_eigenvalue_ratio{_X}"] = (
                float(evals[0] / evals[2]) if abs(evals[2]) > 1e-15 else None
            )
            result[f"conoid_{prefix}_quadric_type{_X}"] = (
                "ellipsoid" if n_pos == 3 else "hyperboloid"
            )
        else:
            for k in [
                "n_positive_eigenvalues",
                "eigenvalue_min",
                "eigenvalue_max",
                "eigenvalue_ratio",
                "quadric_type",
            ]:
                result[f"conoid_{prefix}_{k}{_X}"] = None

        if conoid_params is not None:
            # --- Layer 1: reliable geometric parameters (from eigenvalues) ---
            result[f"conoid_{prefix}_CRx{_X}"] = conoid_params["CRx"]
            result[f"conoid_{prefix}_CRy{_X}"] = conoid_params["CRy"]
            result[f"conoid_{prefix}_CQx{_X}"] = conoid_params["CQx"]
            result[f"conoid_{prefix}_CQy{_X}"] = conoid_params["CQy"]

            # --- Derived features ---
            CRx, CRy = conoid_params["CRx"], conoid_params["CRy"]
            CQx, CQy = conoid_params["CQx"], conoid_params["CQy"]
            result[f"conoid_{prefix}_K_flat_apical{_X}"] = 337.5 / CRx
            result[f"conoid_{prefix}_K_steep_apical{_X}"] = 337.5 / CRy
            result[f"conoid_{prefix}_cyl_apical{_X}"] = 337.5 / CRy - 337.5 / CRx
            result[f"conoid_{prefix}_mean_Q{_X}"] = (CQx + CQy) / 2.0
            result[f"conoid_{prefix}_delta_Q{_X}"] = abs(CQx - CQy)

            # Internal (for ray tracing, not CSV)
            result[f"conoid_{prefix}_quadric_coeffs"] = list(conoid_params["quadric_coeffs"])

            # --- Unreliable parameters (eigenvector-dependent, ~15° arc coverage) ---
            # Kept for research but NOT recommended as ML features.
            # result[f"conoid_{prefix}_CAx{_X}"] = float(conoid_params["apex"][0])
            # result[f"conoid_{prefix}_CAy{_X}"] = float(conoid_params["apex"][1])
            # result[f"conoid_{prefix}_tilt_alpha{_X}"] = conoid_params["tilt_alpha"]
            # result[f"conoid_{prefix}_tilt_beta{_X}"] = conoid_params["tilt_beta"]
            # result[f"conoid_{prefix}_axis_gamma{_X}"] = conoid_params["axis_gamma"]
        else:
            for key in [
                "CRx",
                "CRy",
                "CQx",
                "CQy",
                "K_flat_apical",
                "K_steep_apical",
                "cyl_apical",
                "mean_Q",
                "delta_Q",
            ]:
                result[f"conoid_{prefix}_{key}{_X}"] = None
            result[f"conoid_{prefix}_quadric_coeffs"] = None

        # --- Layer 2: Zernike on conoid residuals (ML features, NOT clinical Zernike) ---
        if zernike_coeffs_um is not None:
            for j in range(len(zernike_coeffs_um)):
                result[f"conoid_{prefix}_z{j}_um{_X}"] = float(zernike_coeffs_um[j])
            c = zernike_coeffs_um
            result[f"conoid_{prefix}_irregular_astig_um{_X}"] = float(
                np.sqrt(c[3] ** 2 + c[5] ** 2)
            )
            result[f"conoid_{prefix}_residual_coma_um{_X}"] = float(np.sqrt(c[7] ** 2 + c[8] ** 2))
            result[f"conoid_{prefix}_residual_sa_um{_X}"] = float(abs(c[12]))
            result[f"conoid_{prefix}_residual_hoa_rms_um{_X}"] = float(np.sqrt(np.sum(c[6:] ** 2)))
        else:
            for j in range(45):
                result[f"conoid_{prefix}_z{j}_um{_X}"] = None
            for k in [
                "irregular_astig_um",
                "residual_coma_um",
                "residual_sa_um",
                "residual_hoa_rms_um",
            ]:
                result[f"conoid_{prefix}_{k}{_X}"] = None

        # --- Diagnostics ---
        for diag_key in ["residual_rms_um", "fit_rms_um"]:
            result[f"conoid_{prefix}_{diag_key}{_X}"] = diagnostics.get(diag_key)

        # =============================================================
        # Layer 3: Tilted biconic refit (8 params, LM)
        # =============================================================
        # Independent BQx/BQy (not coupled through sa3 like CQx/CQy).
        # Explicit tilt correction (Consejo 2021: posterior Q corrupted
        # without it at all apertures, p<0.01).
        bic_params, bic_zernike_um, bic_diag = _biconic_fit_one_surface(
            polar_map, fitting_radius, conoid_params=conoid_params
        )

        if bic_params is not None:
            result[f"biconic_{prefix}_BRx{_X}"] = bic_params["BRx"]
            result[f"biconic_{prefix}_BRy{_X}"] = bic_params["BRy"]
            result[f"biconic_{prefix}_BQx{_X}"] = bic_params["BQx"]
            result[f"biconic_{prefix}_BQy{_X}"] = bic_params["BQy"]
            result[f"biconic_{prefix}_K_flat{_X}"] = 337.5 / bic_params["BRx"]
            result[f"biconic_{prefix}_K_steep{_X}"] = 337.5 / bic_params["BRy"]
            result[f"biconic_{prefix}_cyl{_X}"] = (
                337.5 / bic_params["BRy"] - 337.5 / bic_params["BRx"]
            )
            result[f"biconic_{prefix}_mean_Q{_X}"] = (bic_params["BQx"] + bic_params["BQy"]) / 2.0
            result[f"biconic_{prefix}_delta_Q{_X}"] = abs(bic_params["BQx"] - bic_params["BQy"])
            result[f"biconic_{prefix}_axis{_X}"] = bic_params["gamma"]
            result[f"biconic_{prefix}_tilt_x{_X}"] = bic_params["alpha_x"]
            result[f"biconic_{prefix}_tilt_y{_X}"] = bic_params["alpha_y"]
            result[f"biconic_{prefix}_residual_rms_um{_X}"] = bic_params["residual_rms_um"]
            result[f"biconic_{prefix}_jacobian_cond{_X}"] = bic_params["jacobian_cond"]
        else:
            for k in [
                "BRx",
                "BRy",
                "BQx",
                "BQy",
                "K_flat",
                "K_steep",
                "cyl",
                "mean_Q",
                "delta_Q",
                "axis",
                "tilt_x",
                "tilt_y",
                "residual_rms_um",
                "jacobian_cond",
            ]:
                result[f"biconic_{prefix}_{k}{_X}"] = None

        # Biconic Zernike residuals
        if bic_zernike_um is not None:
            for j in range(len(bic_zernike_um)):
                result[f"biconic_{prefix}_z{j}_um{_X}"] = float(bic_zernike_um[j])
            c = bic_zernike_um
            result[f"biconic_{prefix}_irregular_astig_um{_X}"] = float(
                np.sqrt(c[3] ** 2 + c[5] ** 2)
            )
            result[f"biconic_{prefix}_residual_coma_um{_X}"] = float(np.sqrt(c[7] ** 2 + c[8] ** 2))
            result[f"biconic_{prefix}_residual_sa_um{_X}"] = float(abs(c[12]))
            result[f"biconic_{prefix}_residual_hoa_rms_um{_X}"] = float(np.sqrt(np.sum(c[6:] ** 2)))
        else:
            for j in range(45):
                result[f"biconic_{prefix}_z{j}_um{_X}"] = None
            for k in [
                "irregular_astig_um",
                "residual_coma_um",
                "residual_sa_um",
                "residual_hoa_rms_um",
            ]:
                result[f"biconic_{prefix}_{k}{_X}"] = None

        for dk in ["fit_rms_um", "tilt_check_um", "defocus_check_um", "astig_check_um"]:
            result[f"biconic_{prefix}_{dk}{_X}"] = bic_diag.get(dk)

    # --- Cross-surface features (anterior/posterior ratio) ---
    crx_a = result.get(f"conoid_ant_CRx{_X}")
    crx_p = result.get(f"conoid_post_CRx{_X}")
    result[f"conoid_post_ant_CRx_ratio{_X}"] = (
        crx_p / crx_a if crx_a and crx_p and crx_a > 0 else None
    )
    cry_a = result.get(f"conoid_ant_CRy{_X}")
    cry_p = result.get(f"conoid_post_CRy{_X}")
    result[f"conoid_post_ant_CRy_ratio{_X}"] = (
        cry_p / cry_a if cry_a and cry_p and cry_a > 0 else None
    )
    qa = result.get(f"conoid_ant_mean_Q{_X}")
    qp = result.get(f"conoid_post_mean_Q{_X}")
    result[f"conoid_post_ant_Q_ratio{_X}"] = qp / qa if qa and qp and abs(qa) > 1e-6 else None

    # Biconic cross-surface ratios
    bry_a = result.get(f"biconic_ant_BRy{_X}")
    bry_p = result.get(f"biconic_post_BRy{_X}")
    result[f"biconic_post_ant_BRy_ratio{_X}"] = (
        bry_p / bry_a if bry_a and bry_p and bry_a > 0 else None
    )
    bqa = result.get(f"biconic_ant_mean_Q{_X}")
    bqp = result.get(f"biconic_post_mean_Q{_X}")
    result[f"biconic_post_ant_Q_ratio{_X}"] = bqp / bqa if bqa and bqp and abs(bqa) > 1e-6 else None

    return result


# ===========================================================================
# Conoid-referenced ray tracing
# ===========================================================================
#
# Standard OPD ray tracing mixes regular optics (defocus, astigmatism, conic
# asphericity) with genuinely irregular higher-order aberrations.  By tracing
# rays through both the measured corneal surface AND the best-fit analytical
# conoid, we decompose the wavefront into:
#
#   OPD_measured  = full wavefront from actual cornea
#   OPD_conoid    = wavefront from the regular conoid model
#   OPD_residual  = OPD_measured - OPD_conoid  (genuine irregular optics)
#
# The conoid captures defocus, regular astigmatism, and the asphericity
# contribution analytically.  The residual captures coma, trefoil, spherical
# aberration beyond the conoid, and all truly irregular optics.
# ===========================================================================

_CONOID_ZERNIKE_ORDER = 8
_CONOID_ZERNIKE_NTERMS = 45  # (8+1)*(8+2)/2
_CONOID_DIAMETERS = [3.0, 4.0, 5.0, 6.0]
_DISC_EPS = 1e-12


def _conoid_eval_surface(quadric_coeffs, query_xy, surface="anterior"):
    """
    Evaluate the analytical conoid (general quadric) surface at query points.

    Solves Z^2 + p*Z + q = 0 at each (X, Y), selects the anterior (smaller Z)
    or posterior (larger Z) root, and computes gradients via implicit
    differentiation.

    Parameters
    ----------
    quadric_coeffs : array-like, shape (9,)
        [a11, a22, a12, a13, a23, b1, b2, b3, c].
    query_xy : ndarray, shape (N, 2)
        X, Y positions in the instrument frame.
    surface : {'anterior', 'posterior'}
        Which root of the quadratic to select.

    Returns
    -------
    z, dz_dx, dz_dy : ndarray (N,)
    valid : ndarray (N,) of bool
    """
    qc = np.asarray(quadric_coeffs, dtype=np.float64)
    a11, a22, a12, a13, a23, b1, b2, b3, c = qc

    X = query_xy[:, 0]
    Y = query_xy[:, 1]
    N = len(X)

    # Quadratic in Z:  Z^2 + p*Z + q = 0
    p = a13 * X + a23 * Y + b3
    q = a11 * X * X + a22 * Y * Y + a12 * X * Y + b1 * X + b2 * Y + c
    disc = p * p - 4.0 * q

    valid = disc >= _DISC_EPS
    sqrt_disc = np.zeros(N, dtype=np.float64)
    sqrt_disc[valid] = np.sqrt(disc[valid])

    if surface == "anterior":
        z = (-p - sqrt_disc) / 2.0
    elif surface == "posterior":
        z = (-p + sqrt_disc) / 2.0
    else:
        raise ValueError(f"surface must be 'anterior' or 'posterior', got {surface!r}")

    # Gradients via implicit differentiation of F(X, Y, Z) = 0
    dF_dX = 2.0 * a11 * X + a12 * Y + a13 * z + b1
    dF_dY = a12 * X + 2.0 * a22 * Y + a23 * z + b2
    dF_dZ = a13 * X + a23 * Y + 2.0 * z + b3

    dF_dZ_safe = np.where(np.abs(dF_dZ) < 1e-15, 1e-15, dF_dZ)
    dz_dx = -dF_dX / dF_dZ_safe
    dz_dy = -dF_dY / dF_dZ_safe

    valid = valid & (np.abs(dF_dZ) >= 1e-12)
    z[~valid] = np.nan
    dz_dx[~valid] = np.nan
    dz_dy[~valid] = np.nan

    return z, dz_dx, dz_dy, valid


def _conoid_raytrace_single(
    quadric_coeffs,
    fitting_radius,
    n1,
    n2,
    surface="anterior",
    offset_x=0.0,
    offset_y=0.0,
):
    """
    Trace collimated rays through the analytical conoid surface.

    Returns
    -------
    coeffs_opd_um : ndarray (45,) or None
    focal_z : float or None
    """
    ray_xy = create_ray_grid(fitting_radius)
    query_xy = np.column_stack((ray_xy[:, 0] - offset_x, ray_xy[:, 1] - offset_y))

    z, dz_dx, dz_dy, valid = _conoid_eval_surface(quadric_coeffs, query_xy, surface)
    n_valid = int(np.sum(valid))
    if n_valid < _ZERNIKE_MIN_POINTS:
        return None, None

    mask = valid
    x_v, y_v = ray_xy[mask, 0], ray_xy[mask, 1]
    xq_v, yq_v = query_xy[mask, 0], query_xy[mask, 1]
    z_v = z[mask]
    nv = n_valid

    surface_points = np.column_stack((xq_v, yq_v, z_v))
    entry_points = np.column_stack((xq_v, yq_v, np.zeros(nv)))
    normals = surface_normals_from_gradients(dz_dx[mask], dz_dy[mask])

    incident_dirs = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (nv, 3)).copy()
    try:
        refracted_dirs = snells_law_vector_batch(incident_dirs, normals, n1, n2)
    except ValueError:
        return None, None

    try:
        focal_x, focal_y, focal_z = estimate_focal_point(surface_points, refracted_dirs)
    except ValueError:
        return None, None

    d1 = np.linalg.norm(surface_points - entry_points, axis=1)
    focal_point = np.array([focal_x, focal_y, focal_z])
    fp_vec = focal_point[np.newaxis, :] - surface_points
    d2 = np.abs(np.sum(fp_vec * refracted_dirs, axis=1))
    opl = n1 * d1 + n2 * d2
    opd = opl - np.mean(opl)

    coeffs_um, _ = _fit_zernike_coefficients(
        x_v,
        y_v,
        opd,
        fitting_radius,
        max_order=_CONOID_ZERNIKE_ORDER,
    )
    if coeffs_um is None:
        return None, None
    coeffs_um[0] = 0.0
    return coeffs_um, focal_z


def _conoid_raytrace_residual_single(
    polar_map,
    quadric_coeffs,
    fitting_radius,
    n1,
    n2,
    surface="anterior",
    offset_x=0.0,
    offset_y=0.0,
    _cache=None,
):
    """
    Compute conoid-referenced OPD for a single corneal surface.

    Traces rays through both the measured surface (biquadratic) and the
    analytical conoid, decomposes wavefront into measured, conoid, residual.

    Returns dict with coeffs_measured_um, coeffs_conoid_um, coeffs_residual_um,
    focal_z_measured, focal_z_conoid.  Or None on failure.
    """
    if polar_map is None:
        return None

    ray_xy = create_ray_grid(fitting_radius)
    query_xy = np.column_stack((ray_xy[:, 0] - offset_x, ray_xy[:, 1] - offset_y))

    # Evaluate MEASURED surface (biquadratic interpolation)
    meas_z, meas_dz_dx, meas_dz_dy, meas_valid = _biquad_eval_batch(
        polar_map,
        query_xy,
        _cache=_cache,
    )
    # Evaluate CONOID surface (analytical)
    con_z, con_dz_dx, con_dz_dy, con_valid = _conoid_eval_surface(
        quadric_coeffs,
        query_xy,
        surface,
    )

    both_valid = meas_valid & con_valid
    n_valid = int(np.sum(both_valid))
    if n_valid < _ZERNIKE_MIN_POINTS:
        return None

    mask = both_valid
    x_v, y_v = ray_xy[mask, 0], ray_xy[mask, 1]
    xq_v, yq_v = query_xy[mask, 0], query_xy[mask, 1]
    nv = n_valid

    incident_dir = np.array([0.0, 0.0, 1.0])
    entry_points = np.column_stack((xq_v, yq_v, np.zeros(nv)))

    # --- MEASURED surface ray trace ---
    meas_surface_pts = np.column_stack((xq_v, yq_v, meas_z[mask]))
    meas_normals = surface_normals_from_gradients(meas_dz_dx[mask], meas_dz_dy[mask])
    incident_dirs = np.broadcast_to(incident_dir, (nv, 3)).copy()
    try:
        meas_refracted = snells_law_vector_batch(incident_dirs, meas_normals, n1, n2)
    except ValueError:
        return None
    try:
        meas_fx, meas_fy, meas_fz = estimate_focal_point(meas_surface_pts, meas_refracted)
    except ValueError:
        return None

    # --- CONOID surface ray trace (compute first to get reference focal point) ---
    con_surface_pts = np.column_stack((xq_v, yq_v, con_z[mask]))
    con_normals = surface_normals_from_gradients(con_dz_dx[mask], con_dz_dy[mask])
    incident_dirs_con = np.broadcast_to(incident_dir, (nv, 3)).copy()
    try:
        con_refracted = snells_law_vector_batch(incident_dirs_con, con_normals, n1, n2)
    except ValueError:
        return None
    try:
        con_fx, con_fy, con_fz = estimate_focal_point(con_surface_pts, con_refracted)
    except ValueError:
        return None

    # Use the conoid focal point as the shared reference for BOTH OPDs.
    # This prevents spurious defocus in the residual from focal length
    # mismatch between measured and conoid surfaces.
    shared_focal = np.array([con_fx, con_fy, con_fz])

    # --- MEASURED OPL (referenced to conoid focal point) ---
    meas_d1 = np.linalg.norm(meas_surface_pts - entry_points, axis=1)
    meas_fp_vec = shared_focal[np.newaxis, :] - meas_surface_pts
    meas_d2 = np.abs(np.sum(meas_fp_vec * meas_refracted, axis=1))
    meas_opl = n1 * meas_d1 + n2 * meas_d2
    meas_opd = meas_opl - np.mean(meas_opl)

    # --- CONOID OPL (same shared focal point) ---
    con_d1 = np.linalg.norm(con_surface_pts - entry_points, axis=1)
    con_fp_vec = shared_focal[np.newaxis, :] - con_surface_pts
    con_d2 = np.abs(np.sum(con_fp_vec * con_refracted, axis=1))
    con_opl = n1 * con_d1 + n2 * con_d2
    con_opd = con_opl - np.mean(con_opl)

    # --- Residual OPD (pure shape irregularity, no spurious defocus) ---
    residual_opd = meas_opd - con_opd

    # --- Fit Zernike to all three ---
    coeffs_meas_um, _ = _fit_zernike_coefficients(
        x_v,
        y_v,
        meas_opd,
        fitting_radius,
        max_order=_CONOID_ZERNIKE_ORDER,
    )
    coeffs_con_um, _ = _fit_zernike_coefficients(
        x_v,
        y_v,
        con_opd,
        fitting_radius,
        max_order=_CONOID_ZERNIKE_ORDER,
    )
    coeffs_res_um, _ = _fit_zernike_coefficients(
        x_v,
        y_v,
        residual_opd,
        fitting_radius,
        max_order=_CONOID_ZERNIKE_ORDER,
    )

    if coeffs_meas_um is None or coeffs_con_um is None or coeffs_res_um is None:
        return None

    coeffs_meas_um[0] = 0.0
    coeffs_con_um[0] = 0.0
    coeffs_res_um[0] = 0.0

    return {
        "coeffs_measured_um": coeffs_meas_um,
        "coeffs_conoid_um": coeffs_con_um,
        "coeffs_residual_um": coeffs_res_um,
        "focal_z_measured": meas_fz,
        "focal_z_conoid": con_fz,
    }


def _conoid_opd_diameter_label(d):
    """Convert diameter float to key label: 4.0 -> '4mm', 3.5 -> '3p5mm'."""
    if d == int(d):
        return f"{int(d)}mm"
    return f"{str(d).replace('.', 'p')}mm"


def _emit_conoid_opd_values(out, surf, dlabel, result, prefix_extra=""):
    """Emit output keys for a successful conoid OPD computation."""
    _X = "_experimental"
    base = f"conoid_opd_{surf}{prefix_extra}_{dlabel}"
    for component, coeff_key in [
        ("measured", "coeffs_measured_um"),
        ("conoid", "coeffs_conoid_um"),
        ("residual", "coeffs_residual_um"),
    ]:
        coeffs = result[coeff_key]
        for j in range(_CONOID_ZERNIKE_NTERMS):
            out[f"{base}_{component}_z{j}_um{_X}"] = float(coeffs[j])

    out[f"{base}_focal_measured_mm{_X}"] = float(result["focal_z_measured"])
    out[f"{base}_focal_conoid_mm{_X}"] = float(result["focal_z_conoid"])

    coeffs_meas = result["coeffs_measured_um"]
    coeffs_res = result["coeffs_residual_um"]
    hoa_meas = float(np.sqrt(np.sum(coeffs_meas[6:] ** 2)))
    hoa_res = float(np.sqrt(np.sum(coeffs_res[6:] ** 2)))
    out[f"{base}_hoa_rms_measured_um{_X}"] = hoa_meas
    out[f"{base}_hoa_rms_residual_um{_X}"] = hoa_res
    # MS-39 HOA RMS repeatability is ~0.3 µm; below that, ratio is noise.
    out[f"{base}_hoa_absorption{_X}"] = 1.0 - hoa_res / hoa_meas if hoa_meas > 0.3 else None


def _emit_conoid_opd_none(out, surf, dlabel, prefix_extra=""):
    """Emit None for all output keys when computation fails."""
    _X = "_experimental"
    base = f"conoid_opd_{surf}{prefix_extra}_{dlabel}"
    for component in ("measured", "conoid", "residual"):
        for j in range(_CONOID_ZERNIKE_NTERMS):
            out[f"{base}_{component}_z{j}_um{_X}"] = None
    out[f"{base}_focal_measured_mm{_X}"] = None
    out[f"{base}_focal_conoid_mm{_X}"] = None
    out[f"{base}_hoa_rms_measured_um{_X}"] = None
    out[f"{base}_hoa_rms_residual_um{_X}"] = None
    out[f"{base}_hoa_absorption{_X}"] = None


def compute_conoid_opd(raw_segments, metadata, conoid_result, fitting_radius=4.0):
    """
    Conoid-referenced OPD wavefront for anterior and posterior surfaces.

    For each surface and diameter, traces rays through both the measured
    cornea and the analytical conoid, decomposing the wavefront into
    measured, conoid, and residual components.

    Parameters
    ----------
    raw_segments : dict
        Must contain 'elevation_anterior' and/or 'elevation_posterior'.
    metadata : dict
        Must contain 'PupilCX', 'PupilCY' for CV-centered analysis.
    conoid_result : dict
        Output of compute_conoid_analysis.
    fitting_radius : float
        Default fitting radius (overridden per-diameter in the loop).

    Returns
    -------
    dict
        Keys: conoid_opd_{surf}_{diam}_{component}_z{j}_um, etc.
    """
    out = {}
    _cache = {}

    pupil_cx = float(metadata.get("PupilCX", 0.0) or 0.0)
    pupil_cy = float(metadata.get("PupilCY", 0.0) or 0.0)

    # Both surfaces use "anterior" root — the measured corneal surface
    # is always on the observer-facing (near) side of its own ellipsoid.
    # The "posterior" root gives the far hemisphere, which is physically
    # meaningless for corneal topography.
    surface_configs = [
        ("ant", "elevation_anterior", "conoid_ant_quadric_coeffs", N_AIR, N_CORNEA, "anterior"),
        (
            "post",
            "elevation_posterior",
            "conoid_post_quadric_coeffs",
            N_CORNEA,
            N_AQUEOUS,
            "anterior",
        ),
    ]

    for surf_label, map_key, qc_key, n1, n2, surf_type in surface_configs:
        polar_map = raw_segments.get(map_key)
        quadric_coeffs = conoid_result.get(qc_key)

        if polar_map is None or quadric_coeffs is None:
            for d in _CONOID_DIAMETERS:
                dlabel = _conoid_opd_diameter_label(d)
                _emit_conoid_opd_none(out, surf_label, dlabel)
            continue

        max_data_radius = (polar_map.shape[0] - 1) * _R_STEP_MM

        for d in _CONOID_DIAMETERS:
            dlabel = _conoid_opd_diameter_label(d)
            fr = d / 2.0

            if fr > max_data_radius:
                _emit_conoid_opd_none(out, surf_label, dlabel)
                continue

            # Pupil-centered (offset = 0)
            result = _conoid_raytrace_residual_single(
                polar_map,
                quadric_coeffs,
                fr,
                n1,
                n2,
                surface=surf_type,
                _cache=_cache,
            )
            if result is None:
                _emit_conoid_opd_none(out, surf_label, dlabel)
                continue
            _emit_conoid_opd_values(out, surf_label, dlabel, result)

            # CV-centered (offset from metadata)
            if abs(pupil_cx) > 0.001 or abs(pupil_cy) > 0.001:
                result_cv = _conoid_raytrace_residual_single(
                    polar_map,
                    quadric_coeffs,
                    fr,
                    n1,
                    n2,
                    surface=surf_type,
                    offset_x=-pupil_cx,
                    offset_y=-pupil_cy,
                    _cache=_cache,
                )
                if result_cv is None:
                    _emit_conoid_opd_none(out, surf_label, dlabel, prefix_extra="_cv")
                else:
                    _emit_conoid_opd_values(out, surf_label, dlabel, result_cv, prefix_extra="_cv")

    return out
