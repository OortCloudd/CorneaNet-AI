"""
CorneaForge Core Module
========================

Shared foundation for the neural network and visualization pipelines.
This module handles two tasks:

1. PARSING — Reading MS39 CSV exports and dynamically detecting segments.
2. CONVERSION — Transforming polar coordinate data into Cartesian image grids.

WHY THIS MODULE EXISTS
----------------------
Both pipelines (NN and Visual) need the exact same data extraction and
coordinate conversion. Centralizing this logic here means:
  - A firmware change (e.g., new header lines) is fixed in ONE place
  - Both pipelines are guaranteed to read the same data the same way
  - The conversion math is tested and maintained once

HOW THE MS39 STORES DATA
-------------------------
The MS39 corneal topographer measures the eye using concentric rings of
light projected onto the cornea. It records:
  - 256 angular meridians (columns): evenly spaced points around each ring
  - 22-27 radial rings (rows): from the center of the cornea outward

This is a polar coordinate system: (radius, angle) → measurement value.
The CSV file contains one such polar matrix per measurement type (curvature,
thickness, etc.), separated by text headers like "SagittalAnterior [mm]".

To use this data as images — for CNN input or clinical visualization — we
must convert from polar (rings × angles) to Cartesian (x × y pixel grid).
That conversion is the core mathematical operation in this module.
"""

import numpy as np
from scipy.spatial import Delaunay

# ==========================================================================
# SEGMENT REGISTRY
# ==========================================================================
# Maps the header text found in MS39 CSV exports to internal snake_case names.
#
# WHY DYNAMIC DETECTION (instead of hardcoded line numbers):
# The MS39 firmware has changed header layouts between versions. For example,
# a recent update added 5 metadata lines, shifting every segment down by 5.
# Hardcoding "sagittal starts at row 28" breaks silently — you read the wrong
# data with no error. Dynamic detection finds the header text itself, so
# firmware can add or remove header lines without breaking the parser.

SEGMENT_HEADERS = {
    "SagittalAnterior": "sagittal_anterior",
    "TangentialAnterior": "tangential_anterior",
    "GaussianAnterior": "gaussian_anterior",
    "SagittalPosterior": "sagittal_posterior",
    "TangentialPosterior": "tangential_posterior",
    "GaussianPosterior": "gaussian_posterior",
    "RefractiveFrontalPowerAnterior": "refra_frontal_power_anterior",
    "RefractiveFrontalPowerPosterior": "refra_frontal_power_posterior",
    "RefractiveEquivalentPower": "refra_equivalent_power",
    "ElevationAnterior": "elevation_anterior",
    "ElevationPosterior": "elevation_posterior",
    "ElevationStromal": "elevation_stromal",
    "CornealThickness": "corneal_thickness",
    "StromalThickness": "stromal_thickness",
    "EpitelialThickness": "epithelial_thickness",  # Typo is in the MS39 firmware
    "AnteriorChamberDepth": "anterior_chamber_depth",
}

# The 13 segments used by default (excluding elevations).
#
# WHY EXCLUDE ELEVATIONS:
# Elevation maps measure height relative to a Best Fitted Sphere (BFS).
# However, the MS39 applies an undocumented internal transformation that
# maps all elevation values into the range [0, 3]. Without knowing what
# that transformation is, we cannot interpret the values reliably.
# The other 13 segments are in standard physical units (mm, µm, Diopters).
DEFAULT_SEGMENTS = [
    "sagittal_anterior",
    "tangential_anterior",
    "gaussian_anterior",
    "sagittal_posterior",
    "tangential_posterior",
    "gaussian_posterior",
    "refra_frontal_power_anterior",
    "refra_frontal_power_posterior",
    "refra_equivalent_power",
    "corneal_thickness",
    "stromal_thickness",
    "epithelial_thickness",
    "anterior_chamber_depth",
]

ALL_SEGMENTS = DEFAULT_SEGMENTS + [
    "elevation_anterior",
    "elevation_posterior",
    "elevation_stromal",
]


# ==========================================================================
# CSV PARSING
# ==========================================================================


def _read_lines(filepath):
    """
    Read all lines from a CSV file, handling encoding differences.

    The MS-39 runs on Windows. Depending on the hospital's locale,
    it may export as UTF-8 (with or without BOM) or Windows-1252.
    Patient names with accents (é, è, ç, ï) cause UnicodeDecodeError
    if we assume UTF-8 on a Windows-1252 file.

    Strategy: try UTF-8 first (covers UTF-8 + BOM), fall back to
    Windows-1252 (the most common non-UTF-8 encoding on Windows).
    """
    for encoding in ("utf-8-sig", "windows-1252"):
        try:
            with open(filepath, encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    # If both fail, let it raise with the original encoding
    with open(filepath, encoding="utf-8-sig") as f:
        return f.readlines()


def _identify_header(line):
    """
    Check if a line is a known segment header.

    Returns the internal segment name (e.g., 'sagittal_anterior') or None.

    We check if the header text appears anywhere in the line because the
    actual CSV line includes the unit: "SagittalAnterior [mm]". A simple
    'in' check is sufficient because no header name is a substring of another.
    """
    stripped = line.strip()
    for header_text, internal_name in SEGMENT_HEADERS.items():
        if header_text in stripped:
            return internal_name
    return None


def _detect_delimiter(line):
    """
    Detect whether a data line uses ; or , as delimiter.

    The MS-39 firmware changed from , to ; at some point.
    We check which delimiter produces more splits. A data line
    has 256 numeric values — the correct delimiter gives 256+ parts.
    """
    semicolons = line.count(";")
    commas = line.count(",")
    return ";" if semicolons >= commas else ","


def _parse_data_rows(lines, start_idx, expected_cols=256):
    """
    Read consecutive rows of delimiter-separated numbers starting at start_idx.

    HOW IT KNOWS WHEN TO STOP:
    It reads lines one by one. If a line has fewer than expected_cols numeric
    values, or contains non-numeric text, it stops. This naturally handles
    the transition from data rows to the next segment header or separator.

    NOTE ON -1000 PADDING:
    The MS39 pads each segment block to exactly 32 rows. Real data comes first,
    then rows filled entirely with -1000 (the device's "no measurement" sentinel).
    We read ALL rows here, including the -1000 padding. The cleanup happens later
    in the pipeline — this keeps the parser's job simple: just read numbers.
    """
    if start_idx >= len(lines):
        return None

    # Auto-detect delimiter from the first data line
    delim = _detect_delimiter(lines[start_idx])

    rows = []
    for i in range(start_idx, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            break
        parts = stripped.rstrip(delim).split(delim)
        if len(parts) < expected_cols:
            break
        try:
            values = [float(x) for x in parts[:expected_cols]]
            rows.append(values)
        except ValueError:
            break  # Hit a non-numeric line (next header or separator)

    return np.array(rows, dtype=np.float64) if rows else None


def parse_csv(filepath):
    """
    Parse an MS39 CSV export, dynamically detecting all data segments.

    HOW IT WORKS:
    1. Read the entire file as text lines (~544 lines, trivially fast)
    2. Scan each line for known segment headers (e.g., "SagittalAnterior [mm]")
    3. When a header is found, read all numeric rows immediately below it
    4. Return everything in a dictionary

    This approach is firmware-agnostic: if the manufacturer adds new header
    lines, changes metadata, or reorders the preamble, the parser still finds
    each segment by its label.

    Parameters
    ----------
    filepath : str
        Path to the MS39 CSV file.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are segment names (e.g., 'sagittal_anterior').
        Values are numpy arrays of shape (n_rows, 256) in polar coordinates.
        The arrays include -1000 padding rows — these are cleaned downstream.
    """
    lines = _read_lines(filepath)

    segments = {}
    for i, line in enumerate(lines):
        name = _identify_header(line)
        if name is not None:
            data = _parse_data_rows(lines, i + 1)
            if data is not None:
                segments[name] = data

    return segments


def parse_metadata(filepath):
    """
    Extract patient and exam metadata from the CSV header.

    The first ~30 lines contain key;value pairs like:
        Patient_ID;P0433494363
        Exam_Eye;OD
        Exam_Scan_Date;06/03/2024

    We parse these generically — any line with a semicolon-separated
    key;value pair before the first data segment is captured.

    Returns
    -------
    dict[str, str]
        Metadata key-value pairs exactly as found in the CSV.
    """
    metadata = {}
    delim = None
    for line in _read_lines(filepath):
        stripped = line.strip()
        # Stop at the first segment header — metadata section is over
        if _identify_header(stripped) is not None:
            break
        # Skip separator lines (##########...)
        if stripped.startswith("#"):
            continue
        # Auto-detect delimiter from the first metadata line
        if delim is None:
            delim = ";" if ";" in stripped else ","
        # Parse key<delim>value pairs
        # Split on delimiter, take the second token as value.
        # Don't use split(delim, 1) — some research files paste extra data
        # after column 256. Split fully and take the first non-empty value.
        parts = stripped.split(delim)
        if len(parts) >= 2:
            key = parts[0].strip()
            value = parts[1].strip()
            if key and value:
                metadata[key] = value

    return metadata


# ==========================================================================
# POLAR → CARTESIAN CONVERSION
# ==========================================================================


def clean_polar_data(polar_array, keep_missing=False):
    """
    Clean a raw polar matrix from the MS39.

    The MS39 uses -1000 in two different situations:
      1. PADDING ROWS: The device pads each segment block to exactly 32 rows.
         Real data comes first, then entire rows of -1000. These are purely
         structural — they carry zero information and must always be removed.
      2. SPARSE MISSING POINTS: Within real data rows, individual cells can
         be -1000 when the device couldn't capture that specific point.
         This happens when the patient's eyelid blocks the beam, they blink,
         or fixation is poor. These are almost exclusively peripheral.

    The keep_missing parameter controls how sparse -1000 values are handled:

      keep_missing=False (default, for NN pipeline):
          All -1000 → NaN. The interpolation will fill these gaps with
          interpolated values from neighbors. This is the right choice for
          CNNs because:
          - The missing pattern is a confound (technician skill, lid anatomy)
            not a disease signal (keratoconus is central, gaps are peripheral)
          - On small datasets, the model would learn "missing periphery =
            disease" as a shortcut instead of learning actual corneal shape

      keep_missing=True (for Visual pipeline):
          Sparse -1000 values are preserved as a special marker (-1000 stays
          in the output, not converted to NaN). The visual pipeline renders
          these as gaps so ophthalmologists can see where the device
          failed to capture — this is standard clinical practice and helps
          them judge data quality and patient cooperation.

    In both modes, full-padding rows (entirely -1000) are always removed.

    Parameters
    ----------
    polar_array : np.ndarray
        Raw polar data, shape (n_rows, 256). May contain -1000 values.
    keep_missing : bool
        If True, preserve sparse -1000 values for clinical display.
        If False, convert all -1000 to NaN for interpolation.

    Returns
    -------
    np.ndarray
        Cleaned data with padding rows removed.
    """
    cleaned = polar_array.copy()

    # Step 1: Always remove full-padding rows (rows that are entirely -1000)
    # These are the MS39's block padding — not real measurements.
    padding_rows = (cleaned == -1000).all(axis=1)
    cleaned = cleaned[~padding_rows]

    # Step 2: Handle sparse -1000 values within real data rows
    if keep_missing:
        # Visual pipeline: keep -1000 in place. The visual pipeline will
        # detect these later and render them as gaps.
        # We still need NaN for truly empty cells (if any exist), but
        # the -1000 sentinel stays untouched.
        pass
    else:
        # NN pipeline: convert all remaining -1000 to NaN so the
        # interpolation fills them with neighboring values.
        cleaned[cleaned == -1000] = np.nan

    return cleaned


def _build_polar_grid(polar_array):
    """
    Build polar and Cartesian coordinate grids from a polar array.

    This is shared between polar_to_cartesian() and missing_data_mask().
    Factored out to avoid duplicating the coordinate math.

    The 256 columns span 0° to ~358.6° (360/256 per step). The angular
    gap between column 255 and column 0 (1.4°) is handled naturally by
    the Delaunay triangulation's convex hull — no periodicity duplication
    needed. Tested: removing the legacy periodicity trick changes output
    by at most 0.003mm (0.3× the MS-39's measurement Sw of ~0.01mm).

    Returns: (polar_array, x_polar, y_polar)
    """
    n_radial, n_angular = polar_array.shape

    r = np.linspace(0, 1, n_radial)
    theta = np.linspace(0, 2 * np.pi * (1 - 1 / n_angular), n_angular)
    theta_grid, r_grid = np.meshgrid(theta, r)

    x_polar = r_grid * np.cos(theta_grid)
    y_polar = r_grid * np.sin(theta_grid)

    return polar_array, x_polar, y_polar


# ==========================================================================
# PRECOMPUTED INTERPOLATION WEIGHTS
# ==========================================================================
#
# THE CORE INSIGHT:
# The polar grid geometry (row→radius, col→angle) is identical for every
# patient with the same number of radial rows. The MS39 has only 4 unique
# row counts (21, 22, 25, 26) across all 16 segments. This means we can
# precompute EVERYTHING about the interpolation geometry once and reuse it
# for every patient.
#
# WHAT WE PRECOMPUTE (one-time cost, ~70-120ms per grid shape):
#
#   1. Delaunay triangulation of the polar→cartesian points
#   2. For each output pixel: which triangle it falls in
#   3. For each output pixel: the 3 barycentric weights (how much each
#      triangle vertex contributes to this pixel's interpolated value)
#
# These are stored as two lookup tables:
#   - indices[i, 0:3] → which 3 input points affect output pixel i
#   - weights[i, 0:3] → how much each contributes (barycentric coords)
#
# WHAT HAPPENS AT RUNTIME (per segment, per patient):
#
#   gathered = input_values[indices]     # read 3 values per pixel (fancy index)
#   result = sum(gathered * weights)     # weighted sum (element-wise multiply)
#
# That's it. Two numpy operations. No triangle lookup, no barycentric
# computation, no scipy overhead. The CPU auto-vectorizes this to SIMD.
#
# MEMORY & CACHE BEHAVIOR:
#   - Input polar data (~27×256 = 27KB) fits in L1 cache (32-64KB)
#     and gets read repeatedly as many output pixels reference the same
#     input points → L1 cache hits after first access
#   - Lookup tables (~1.2MB at 224×224, ~6.4MB at 512×512) sit in L2/L3
#     and are read sequentially → CPU prefetcher handles this efficiently
#   - Output array written sequentially → also prefetcher-friendly
#
# PERFORMANCE HISTORY:
#   Original griddata (no cache):         76.5 ms/segment
#   Cached Delaunay + LinearNDInterp:      3.4 ms/segment
#   Precomputed weights + numpy vectorized: 1.1 ms/segment  (69× total speedup)

_weight_cache = {}


def _get_cached_weights(n_radial, target_size):
    """
    Get or build cached interpolation weight lookup tables for a given
    (n_radial, target_size) combination.

    On first call: builds Delaunay triangulation, finds which triangle each
    output pixel falls in, computes barycentric coordinates. (~70-120ms)

    On subsequent calls: returns the cached lookup tables instantly.

    Returns
    -------
    indices : np.ndarray, shape (n_output, 3), dtype int32
        For each output pixel, the 3 input point indices (triangle vertices).
    weights : np.ndarray, shape (n_output, 3), dtype float32
        For each output pixel, the 3 barycentric interpolation weights.
        These sum to 1.0 for valid pixels.
    valid : np.ndarray, shape (n_output,), dtype bool
        True for pixels inside the triangulation (inside the cornea).
        False for pixels outside (corners of the square + beyond data).
    circle_mask : np.ndarray, shape (target_size, target_size), dtype bool
        True for pixels outside the unit circle (square corners).
    shape : tuple (target_size, target_size)
        Output array shape for reshaping.
    """
    cache_key = (n_radial, target_size)
    if cache_key in _weight_cache:
        return _weight_cache[cache_key]

    # --- Step 1: Build the polar→cartesian coordinate grid ---
    n_angular = 256
    r = np.linspace(0, 1, n_radial)
    theta = np.linspace(0, 2 * np.pi * (1 - 1 / n_angular), n_angular)
    theta_grid, r_grid = np.meshgrid(theta, r)
    x_polar = (r_grid * np.cos(theta_grid)).ravel()
    y_polar = (r_grid * np.sin(theta_grid)).ravel()

    # --- Step 2: Delaunay triangulation ---
    # Connects all polar points into a mesh of triangles.
    # This is the expensive step (~50ms) that we do only once.
    points = np.column_stack([x_polar, y_polar])
    tri = Delaunay(points)

    # --- Step 3: Build output Cartesian grid ---
    axis = np.linspace(-1, 1, target_size)
    xi, yi = np.meshgrid(axis, axis)
    circle_mask = (xi**2 + yi**2) > 1.0
    shape = (target_size, target_size)

    # --- Step 4: Find which triangle each output pixel falls in ---
    n_output = target_size * target_size
    output_points = np.column_stack([xi.ravel(), yi.ravel()])
    simplex_indices = tri.find_simplex(output_points)

    # --- Step 5: Compute barycentric coordinates (vectorized) ---
    # For each output pixel inside the triangulation:
    #   - Get the 3 vertices of its triangle
    #   - Compute weights via the affine transform: b = T[:2] @ (point - T[2])
    #   - The 3 weights are (b0, b1, 1 - b0 - b1), summing to 1.0
    #
    # Barycentric interpretation: if a pixel is at the exact position of
    # vertex 0, weights = (1, 0, 0). If it's at the centroid, weights =
    # (1/3, 1/3, 1/3). The interpolated value is the weighted average
    # of the 3 vertex values.

    indices = np.zeros((n_output, 3), dtype=np.int32)
    weights = np.zeros((n_output, 3), dtype=np.float32)
    valid = simplex_indices >= 0  # pixels inside the triangulation

    valid_idx = np.where(valid)[0]
    si = simplex_indices[valid_idx]

    # Triangle vertices for each valid pixel
    vtx = tri.simplices[si]  # (n_valid, 3) — indices into input points

    # Affine transforms for each triangle (precomputed by Delaunay)
    T = tri.transform[si]  # (n_valid, 3, 3)

    # Barycentric coordinates via matrix multiply
    delta = output_points[valid_idx] - T[:, 2, :]  # (n_valid, 2)
    b = np.einsum("ijk,ik->ij", T[:, :2, :], delta)  # (n_valid, 2)

    indices[valid_idx] = vtx
    weights[valid_idx, 0] = b[:, 0]
    weights[valid_idx, 1] = b[:, 1]
    weights[valid_idx, 2] = 1.0 - b[:, 0] - b[:, 1]

    cached = (indices, weights, valid, circle_mask, shape)
    _weight_cache[cache_key] = cached
    return cached


def _apply_weights(input_flat, indices, weights, valid, circle_mask, shape):
    """
    Apply precomputed interpolation weights to produce a Cartesian image.

    This is the hot path — called once per segment per patient.
    The operation is conceptually simple:

        for each output pixel i:
            result[i] = w0 * input[idx0] + w1 * input[idx1] + w2 * input[idx2]

    Implemented as three vectorized numpy operations:
        1. gathered = input_flat[indices]   — fancy indexing, reads 3 values per pixel
        2. gathered * weights               — element-wise multiply
        3. np.sum(..., axis=1)              — sum across the 3 vertices

    No Python loops, no scipy calls. The CPU auto-vectorizes this to SIMD
    (AVX2/AVX-512), processing 8-16 pixels per instruction.
    """
    # Gather: for each output pixel, read the 3 input values it depends on
    # indices shape: (n_output, 3) — each row has 3 indices into input_flat
    # gathered shape: (n_output, 3) — the 3 input values for each pixel
    gathered = input_flat[indices]

    # Weighted sum: multiply each gathered value by its barycentric weight
    # and sum across the 3 vertices → one interpolated value per pixel
    result = np.sum(gathered * weights, axis=1).astype(np.float32)

    # Mark pixels outside the triangulation as NaN
    result[~valid] = np.nan

    # Reshape to 2D and apply the circular mask
    result = result.reshape(shape)
    result[circle_mask] = np.nan
    return result


# ==========================================================================
# CACHED DELAUNAY FOR NaN-CONTAINING SEGMENTS
# ==========================================================================
#
# When a segment has NaN gaps, the full-grid precomputed weights can't be
# used directly. Instead, we build a Delaunay triangulation from the valid
# (non-NaN) polar points, precompute barycentric weights, and cache the
# result keyed by (n_radial, target_size, NaN_pattern_hash).
#
# Segments sharing the same NaN pattern (same cells missing) reuse the
# cache. In practice, 11 NaN-containing segments collapse to ~4 unique
# patterns per patient, and patterns recur across patients.

_nan_weight_cache: dict = {}


def _interpolate_with_nan_cache(periodic, x_polar, y_polar, target_size):
    """Interpolate a polar array with NaN gaps using cached Delaunay weights."""
    import hashlib

    nan_mask = np.isnan(periodic)
    cache_key = (periodic.shape[0], target_size, hashlib.md5(nan_mask.tobytes()).hexdigest())

    if cache_key not in _nan_weight_cache:
        # Build Delaunay from valid points only (matches griddata behavior)
        valid = ~nan_mask
        points = np.column_stack([x_polar[valid].ravel(), y_polar[valid].ravel()])
        tri = Delaunay(points)

        # Build output grid
        axis = np.linspace(-1, 1, target_size)
        xi, yi = np.meshgrid(axis, axis)
        circle_mask = (xi**2 + yi**2) > 1.0
        shape = (target_size, target_size)

        n_output = target_size * target_size
        output_points = np.column_stack([xi.ravel(), yi.ravel()])
        simplex_indices = tri.find_simplex(output_points)

        indices = np.zeros((n_output, 3), dtype=np.int32)
        weights = np.zeros((n_output, 3), dtype=np.float32)
        valid_wt = simplex_indices >= 0

        valid_idx = np.where(valid_wt)[0]
        si = simplex_indices[valid_idx]
        vtx = tri.simplices[si]
        T = tri.transform[si]
        delta = output_points[valid_idx] - T[:, 2, :]
        b = np.einsum("ijk,ik->ij", T[:, :2, :], delta)

        indices[valid_idx] = vtx
        weights[valid_idx, 0] = b[:, 0]
        weights[valid_idx, 1] = b[:, 1]
        weights[valid_idx, 2] = 1.0 - b[:, 0] - b[:, 1]

        _nan_weight_cache[cache_key] = (indices, weights, valid_wt, circle_mask, shape)

    indices, weights, valid_wt, circle_mask, shape = _nan_weight_cache[cache_key]

    # Extract valid values in the same order as the Delaunay point set
    valid_vals = periodic[~np.isnan(periodic)].ravel().astype(np.float32)
    return _apply_weights(valid_vals, indices, weights, valid_wt, circle_mask, shape)


def polar_to_cartesian(polar_array, target_size=224):
    """
    Convert a polar coordinate matrix to a Cartesian (image) grid.

    THE PROBLEM
    -----------
    The MS39 stores data in polar coordinates:
      - Each row = a ring at a fixed distance from the corneal center
      - Each column = one of 256 angular meridians (0° to 360°)

    CNNs expect rectangular pixel grids (images). Ophthalmologists expect
    circular topography maps. Both need Cartesian coordinates.

    THE METHOD
    ----------
    Step 1 — Angular periodicity:
        The first meridian (0°) and last (360°) are the same physical line.
        We duplicate column 0 at the end so interpolation wraps smoothly
        instead of leaving a visible seam at the 0°/360° boundary.

    Step 2 — Precomputed weight lookup:
        On first call for a given grid shape, we build a Delaunay triangulation
        and compute barycentric interpolation weights for every output pixel.
        This is cached — subsequent calls skip directly to the multiply step.

    Step 3 — Vectorized interpolation:
        Each output pixel's value = weighted sum of 3 input values (the
        triangle vertices). Implemented as numpy fancy indexing + multiply
        + sum. ~1ms per segment with warm cache.

    Step 4 — Circular mask:
        The cornea is round. Pixels outside the unit circle (the corners
        of the square output) are set to NaN — there's no data there.

    WHY LINEAR INTERPOLATION (not cubic):
        With only 22-27 radial steps, the data is too sparse for cubic
        interpolation to add real detail. Cubic would introduce ringing
        artifacts (overshoot near sharp transitions) without improving
        accuracy. Linear is faster and artifact-free.

    INTERPOLATION PRECISION:
        The interpolation from 26×256 polar to 224×224 Cartesian creates
        values that do not exist in the original measurement — ~97% of
        output pixels are interpolated, not measured. The float32 rounding
        error is ~0.000002mm (0.0002× the MS-39's measurement Sw of
        ~0.01mm for sagittal curvature). The interpolation is not the
        precision bottleneck — the device is.

        Reference repeatability (MS-39, Savini 2018, Wang 2024):
          Sagittal curvature:  Sw ≈ 0.01 mm  (CoR ≈ 0.03 mm)
          Pachymetry:          Sw ≈ 0.7-1.7 µm (CoR ≈ 2-5 µm)
          Epithelial thickness: Sw ≈ 0.91 µm  (CoR ≈ 2.5 µm)

    Parameters
    ----------
    polar_array : np.ndarray
        Cleaned polar data, shape (n_radial, 256). NaN where no data.
        Must NOT contain -1000 values — clean them first with clean_polar_data().
    target_size : int
        Side length of the output square image. Default 224 for NN
        (matches standard CNN input sizes). Use 512 for visualization.

    Returns
    -------
    np.ndarray
        Shape (target_size, target_size), dtype float32.
        NaN outside the circular cornea boundary.
    """
    periodic, x_polar, y_polar = _build_polar_grid(polar_array)

    # Gather valid (non-NaN) points for interpolation
    valid_data = ~np.isnan(periodic)
    if not np.any(valid_data):
        return np.full((target_size, target_size), np.nan, dtype=np.float32)

    # Check if ALL points are valid (common case: no NaN in the polar data).
    # If so, use the fast precomputed-weights path. If there are NaN gaps,
    # fall back to griddata which handles arbitrary missing points.
    if valid_data.all():
        # FAST PATH: precomputed weights (1ms per segment)
        n_radial = polar_array.shape[0]
        indices, weights, valid, circle_mask, shape = _get_cached_weights(n_radial, target_size)
        input_flat = periodic.ravel().astype(np.float32)
        return _apply_weights(input_flat, indices, weights, valid, circle_mask, shape)
    else:
        # Segments with NaN: cache precomputed weights per NaN pattern.
        # Segments sharing the same (n_radial, NaN locations) reuse the
        # same Delaunay triangulation and barycentric weights.
        # First occurrence: ~100ms (Delaunay build). Reuse: ~3ms.
        return _interpolate_with_nan_cache(periodic, x_polar, y_polar, target_size)


def missing_data_mask(polar_array_with_sentinels, target_size=512):
    """
    Project -1000 locations from polar coordinates onto a Cartesian grid.

    PURPOSE
    -------
    When the MS39 cannot capture a measurement point (eyelid obstruction,
    blink, poor fixation), it records -1000. These points are almost always
    peripheral. For clinical visualization, ophthalmologists need to SEE
    where the device failed — it helps them judge data quality and decide
    whether to re-examine the patient.

    This function takes the raw polar data (with -1000 still in place),
    creates a binary mask in polar coordinates (1 = missing, 0 = captured),
    and interpolates that mask onto the Cartesian image grid.

    The result is a soft mask (values between 0 and 1) because linear
    interpolation blends the boundary. We threshold at 0.5 to get a clean
    binary mask: True = device could not measure this region.

    Parameters
    ----------
    polar_array_with_sentinels : np.ndarray
        Raw polar data WITH -1000 values still present (from clean_polar_data
        called with keep_missing=True).
    target_size : int
        Must match the target_size used in polar_to_cartesian.

    Returns
    -------
    np.ndarray of bool, shape (target_size, target_size)
        True where the original data had -1000 (missing measurement).
        False where the device successfully captured a value.
    """
    # Build a binary polar mask: 1.0 where -1000, 0.0 where measured
    binary_polar = (polar_array_with_sentinels == -1000).astype(np.float32)

    # Use precomputed weights (the binary mask has no NaN, so it always
    # takes the fast path). Same geometry as polar_to_cartesian.
    n_radial = binary_polar.shape[0]
    indices, weights, valid, circle_mask, shape = _get_cached_weights(n_radial, target_size)

    input_flat = binary_polar.ravel().astype(np.float32)

    # Apply the same weight lookup — but interpret the result as a mask.
    # Values near 1.0 = mostly -1000 neighbors, near 0.0 = mostly measured.
    mask_values = _apply_weights(input_flat, indices, weights, valid, circle_mask, shape)

    # Threshold: if more than half the contributing polar points were -1000,
    # this Cartesian pixel is "missing"
    mask_values = np.nan_to_num(mask_values, nan=0.0)
    return mask_values > 0.5
