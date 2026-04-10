"""
CorneaForge Input Validator
============================

Validates MS39 CSV files before processing. Returns a result with errors
(reject the file) and warnings (process but inform the clinician).

PHILOSOPHY
----------
The validator checks STRUCTURE, not clinical normality. A corneal
thickness of 1200µm is severe edema — the validator passes it silently.
A corneal thickness of 50,000µm is a device error — the validator warns.

Only two things are errors (prediction blocked):
  1. File is unreadable (binary, encoding failure)
  2. No MS39 segments found (not an MS39 export)

Everything else is a WARNING. The prediction runs, the result is shown
to the ophthalmologist WITH the warnings, and THEY decide whether to
trust it. The model — not the validator — judges data quality through
its confidence score.

Usage:
    from corneaforge.validate import validate_csv

    result = validate_csv("/path/to/patient.csv")
    if result.errors:
        # Cannot process — file is unreadable or not MS39
    else:
        # Process and show result + any warnings to the clinician
"""

from dataclasses import dataclass, field

import numpy as np

from corneaforge.core import DEFAULT_SEGMENTS, parse_csv, parse_metadata

# Device-error ranges per segment.
# These catch values that CANNOT come from a human eye regardless of disease.
# A corneal radius of 2mm is extreme pathology but possible.
# A corneal radius of -5mm is a device error.
#
# The validator does NOT judge clinical normality — the model does that.
# These ranges only catch corrupt data, not sick patients.
VALID_RANGES = {
    "sagittal_anterior": (0.5, 100.0),  # mm — no eye has R<0.5 or R>100
    "tangential_anterior": (0.5, 100.0),  # mm — tangential can spike at periphery
    "gaussian_anterior": (0.5, 100.0),  # mm
    "sagittal_posterior": (0.5, 100.0),  # mm
    "tangential_posterior": (0.5, 100.0),  # mm
    "gaussian_posterior": (0.5, 100.0),  # mm
    "refra_frontal_power_anterior": (0.0, 200.0),  # D
    "refra_frontal_power_posterior": (-50.0, 5.0),  # D (normally negative)
    "refra_equivalent_power": (-10.0, 200.0),  # D
    "corneal_thickness": (10.0, 5000.0),  # µm — edema can reach 1200+
    "stromal_thickness": (-100.0, 5000.0),  # µm — allow negative (device artifact)
    "epithelial_thickness": (-100.0, 500.0),  # µm — allow negative (device artifact)
    "anterior_chamber_depth": (-1.0, 10.0),  # mm
}

# Expected row counts per segment (after padding removal).
# If a segment has a different count, the firmware changed or the file is corrupt.
EXPECTED_ROWS = {21, 22, 25, 26}

# Maximum acceptable -1000 percentage before warning.
# Above this, too much data is missing for reliable analysis.
MISSING_WARN_THRESHOLD = 0.40  # 40%


@dataclass
class ValidationResult:
    """
    Result of validating an MS39 CSV file.

    errors: list of strings — file should be rejected, cannot process safely
    warnings: list of strings — file can be processed but clinician should know
    segments_found: list of segment names successfully detected
    eye: str — OD or OS (empty if not found)
    """

    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    segments_found: list = field(default_factory=list)
    eye: str = ""

    @property
    def valid(self):
        return len(self.errors) == 0


def validate_csv(filepath):
    """
    Validate an MS39 CSV file for processing readiness.

    Convenience wrapper that parses the file then validates.
    For batch processing, use validate_parsed() instead to avoid
    reading the file twice (once for validation, once for processing).

    Returns a ValidationResult with errors and warnings.
    """
    result = ValidationResult()

    # Can we read the file at all?
    try:
        metadata = parse_metadata(filepath)
    except UnicodeDecodeError:
        result.errors.append("File is not valid text (binary or wrong encoding)")
        return result
    except Exception as e:
        result.errors.append(f"Cannot read file: {e}")
        return result

    try:
        raw_segments = parse_csv(filepath)
    except Exception as e:
        result.errors.append(f"CSV parsing failed: {e}")
        return result

    return validate_parsed(metadata, raw_segments)


def validate_parsed(metadata, raw_segments):
    """
    Validate pre-parsed MS39 data.

    Use this in pipelines to avoid redundant file I/O:
        metadata = parse_metadata(filepath)
        raw_segments = parse_csv(filepath)
        result = validate_parsed(metadata, raw_segments)
        if result.valid:
            # process raw_segments directly — no second parse needed

    Parameters
    ----------
    metadata : dict
        Output of parse_metadata().
    raw_segments : dict
        Output of parse_csv(). Keys are segment names, values are numpy arrays.

    Returns a ValidationResult with errors and warnings.
    """
    result = ValidationResult()

    # --- 1. Does it have basic MS39 metadata? ---
    eye = metadata.get("Exam_Eye", "").strip().upper()
    if eye in ("OD", "OS"):
        result.eye = eye
    else:
        # Default to OD with warning — don't block the prediction.
        # The ophthalmologist can see the warning and correct if needed.
        result.eye = "OD"
        result.warnings.append(
            f"Exam_Eye is '{metadata.get('Exam_Eye', '')}' — defaulting to OD. "
            "Nasal/temporal features may be swapped if this is actually OS."
        )

    if not metadata.get("Patient_ID"):
        result.warnings.append("Patient_ID is missing — file may be an incomplete export")

    # --- 2. Do we have segments? ---
    if not raw_segments:
        result.errors.append("No data segments found — not an MS39 export")
        return result

    result.segments_found = list(raw_segments.keys())

    # --- 3. Are the expected segments present? ---
    missing = [s for s in DEFAULT_SEGMENTS if s not in raw_segments]
    if missing:
        result.warnings.append(f"{len(missing)} segments missing: {', '.join(missing)}")

    # --- 5. Per-segment checks ---
    for name, data in raw_segments.items():
        if name not in DEFAULT_SEGMENTS:
            continue

        # Row count check
        # Remove full-padding rows first (same logic as clean_polar_data)
        padding_mask = (data == -1000).all(axis=1)
        real_rows = data[~padding_mask]
        n_rows = real_rows.shape[0]

        if n_rows == 0:
            result.warnings.append(f"{name}: entirely empty (all -1000)")
            continue

        if n_rows not in EXPECTED_ROWS:
            result.warnings.append(
                f"{name}: {n_rows} data rows (expected one of {sorted(EXPECTED_ROWS)}) "
                "— possible firmware change"
            )

        # Missing data percentage
        n_sentinel = (real_rows == -1000).sum()
        n_total = real_rows.size
        missing_pct = n_sentinel / n_total if n_total > 0 else 0

        if missing_pct > MISSING_WARN_THRESHOLD:
            result.warnings.append(
                f"{name}: {missing_pct:.0%} missing data (-1000) — poor acquisition"
            )

        # Value range check (on non-sentinel values only)
        values = real_rows[real_rows != -1000]
        if len(values) == 0:
            continue

        expected_range = VALID_RANGES.get(name)
        if expected_range is None:
            continue

        lo, hi = expected_range
        out_of_range = ((values < lo) | (values > hi)).sum()
        if out_of_range > 0:
            pct = out_of_range / len(values) * 100
            actual_min, actual_max = values.min(), values.max()
            result.warnings.append(
                f"{name}: {out_of_range} values ({pct:.1f}%) outside "
                f"[{lo}, {hi}] (actual range: [{actual_min:.1f}, {actual_max:.1f}])"
            )

        # Negative values where they shouldn't exist
        if name in ("corneal_thickness", "stromal_thickness") and np.any(values < 0):
            n_neg = (values < 0).sum()
            result.warnings.append(f"{name}: {n_neg} negative values (device artifact)")

    return result
