"""
Tests for corneaforge.validate

Tests both validate_csv (file-based) and validate_parsed (pre-parsed data).
Each error and warning path in the validator is tested explicitly.
"""

import numpy as np

from corneaforge.core import DEFAULT_SEGMENTS
from corneaforge.validate import (
    EXPECTED_ROWS,
    MISSING_WARN_THRESHOLD,
    VALID_RANGES,
    validate_csv,
    validate_parsed,
)

_REALISTIC_VALUES = {
    "sagittal_anterior": 7.8,
    "tangential_anterior": 7.8,
    "gaussian_anterior": 7.8,
    "sagittal_posterior": 6.5,
    "tangential_posterior": 6.5,
    "gaussian_posterior": 6.5,
    "refra_frontal_power_anterior": 44.0,
    "refra_frontal_power_posterior": -6.0,
    "refra_equivalent_power": 42.0,
    "corneal_thickness": 540.0,
    "stromal_thickness": 480.0,
    "epithelial_thickness": 53.0,
    "anterior_chamber_depth": 3.2,
}


def _make_segments(names=None, n_rows=26, n_cols=256, value=None):
    """Build a dict of clean synthetic segments with realistic values."""
    if names is None:
        names = DEFAULT_SEGMENTS
    segments = {}
    for name in names:
        v = value if value is not None else _REALISTIC_VALUES.get(name, 7.8)
        segments[name] = np.full((n_rows, n_cols), v, dtype=np.float64)
    return segments


def _make_metadata(eye="OD", patient_id="P001"):
    """Build a minimal valid metadata dict."""
    meta = {
        "Patient_Last_Name": "TEST",
        "Patient_First_Name": "SYNTH",
        "Exam_Eye": eye,
    }
    if patient_id:
        meta["Patient_ID"] = patient_id
    return meta


# ==========================================================================
# ERRORS — file should be rejected
# ==========================================================================


class TestErrors:
    """Only two things should block processing: unreadable file or no segments."""

    def test_binary_file_returns_error(self, tmp_path):
        """Binary garbage should produce an error, not crash."""
        f = tmp_path / "garbage.csv"
        f.write_bytes(b"\x80\x81\x82\x00\xff\xfe")
        result = validate_csv(str(f))
        assert not result.valid
        assert any("text" in e.lower() or "binary" in e.lower() for e in result.errors)

    def test_empty_segments_is_error(self):
        """No segments at all → not an MS39 file."""
        result = validate_parsed(_make_metadata(), {})
        assert not result.valid
        assert any("No data segments" in e for e in result.errors)


class TestExamEyeIsWarningNotError:
    """Missing/invalid Exam_Eye defaults to OD with warning — doesn't block."""

    def test_missing_exam_eye_defaults_to_od(self):
        meta = _make_metadata()
        del meta["Exam_Eye"]
        result = validate_parsed(meta, _make_segments())
        assert result.valid
        assert result.eye == "OD"
        assert any("Exam_Eye" in w for w in result.warnings)

    def test_invalid_exam_eye_defaults_to_od(self):
        meta = _make_metadata(eye="XX")
        result = validate_parsed(meta, _make_segments())
        assert result.valid
        assert result.eye == "OD"
        assert any("defaulting to OD" in w for w in result.warnings)

    def test_no_default_segments_is_warning(self):
        """Segments present but none from DEFAULT_SEGMENTS — warn, don't block."""
        segments = {"some_unknown_segment": np.ones((26, 256))}
        result = validate_parsed(_make_metadata(), segments)
        assert result.valid
        assert any("missing" in w.lower() for w in result.warnings)


# ==========================================================================
# WARNINGS — file can be processed but clinician should know
# ==========================================================================


class TestWarnings:
    def test_missing_patient_id_warns(self):
        """Missing Patient_ID is a warning, not an error."""
        meta = _make_metadata(patient_id=None)
        result = validate_parsed(meta, _make_segments())
        assert result.valid
        assert any("Patient_ID" in w for w in result.warnings)

    def test_missing_segments_warns(self):
        """Having 10 of 13 segments warns about the 3 missing ones."""
        present = DEFAULT_SEGMENTS[:10]
        result = validate_parsed(_make_metadata(), _make_segments(names=present))
        assert result.valid
        assert any("3 segments missing" in w for w in result.warnings)

    def test_unexpected_row_count_warns(self):
        """Row count not in {21, 22, 25, 26} warns about firmware change."""
        segments = _make_segments(names=["sagittal_anterior"], n_rows=30)
        # Add padding rows so clean removes them to get 30 real rows
        # Actually 30 is not in EXPECTED_ROWS so it should warn
        result = validate_parsed(_make_metadata(), segments)
        assert result.valid
        assert any("firmware" in w.lower() or "rows" in w.lower() for w in result.warnings)

    def test_high_missing_percentage_warns(self):
        """More than 40% -1000 values should warn about poor acquisition."""
        segments = _make_segments(names=["sagittal_anterior"])
        # Set 50% of values to -1000
        segments["sagittal_anterior"][:13, :] = -1000
        result = validate_parsed(_make_metadata(), segments)
        assert result.valid
        assert any("missing" in w.lower() or "-1000" in w for w in result.warnings)

    def test_out_of_range_values_warns(self):
        """Values outside physiological range should warn."""
        segments = _make_segments(names=["sagittal_anterior"])
        # Sagittal anterior device-error range is (0.5, 100.0) — set to 999
        segments["sagittal_anterior"][0, :10] = 999.0
        result = validate_parsed(_make_metadata(), segments)
        assert result.valid
        assert any("outside" in w for w in result.warnings)

    def test_negative_thickness_warns(self):
        """Negative stromal thickness is a device artifact, should warn."""
        segments = _make_segments(names=["stromal_thickness"], value=500.0)
        segments["stromal_thickness"][0, :5] = -10.0
        result = validate_parsed(_make_metadata(), segments)
        assert result.valid
        assert any("negative" in w.lower() for w in result.warnings)


# ==========================================================================
# CLEAN FILES — no errors, no warnings
# ==========================================================================


class TestCleanFiles:
    def test_all_segments_normal_values(self):
        """Perfect file with all 13 segments and normal values → no issues."""
        result = validate_parsed(_make_metadata(), _make_segments())
        assert result.valid
        assert len(result.warnings) == 0
        assert len(result.errors) == 0

    def test_both_eyes_valid(self):
        """Both OD and OS are accepted."""
        for eye in ("OD", "OS"):
            result = validate_parsed(_make_metadata(eye=eye), _make_segments())
            assert result.valid
            assert result.eye == eye

    def test_segments_found_populated(self):
        """segments_found should list all detected segments."""
        result = validate_parsed(_make_metadata(), _make_segments())
        assert len(result.segments_found) == 13


# ==========================================================================
# VALIDATE_PARSED vs VALIDATE_CSV consistency
# ==========================================================================


class TestConsistency:
    def test_validate_csv_on_empty_file(self, tmp_path):
        """validate_csv wraps validate_parsed — empty file should error."""
        f = tmp_path / "empty.csv"
        f.write_text("")
        result = validate_csv(str(f))
        assert not result.valid

    def test_validate_csv_on_valid_minimal(self, tmp_path):
        """validate_csv on a minimal valid file should work."""
        content = "Exam_Eye;OD\n"
        content += "Patient_ID;P001\n"
        content += "SagittalAnterior [mm]\n"
        content += ";".join(["7.8"] * 256) + "\n"
        f = tmp_path / "valid.csv"
        f.write_text(content)
        result = validate_csv(str(f))
        # May have warnings (missing segments) but should parse without error
        assert result.eye == "OD"
        assert "sagittal_anterior" in result.segments_found


# ==========================================================================
# EDGE CASES
# ==========================================================================


class TestEdgeCases:
    def test_entirely_sentinel_segment_warns(self):
        """A segment that is 100% -1000 should warn."""
        segments = _make_segments(names=["sagittal_anterior"])
        segments["sagittal_anterior"][:] = -1000
        result = validate_parsed(_make_metadata(), segments)
        assert result.valid
        assert any("empty" in w.lower() or "missing" in w.lower() for w in result.warnings)

    def test_valid_result_properties(self):
        """ValidationResult.valid is True when errors is empty."""
        result = validate_parsed(_make_metadata(), _make_segments())
        assert result.valid is True
        result.errors.append("something broke")
        assert result.valid is False

    def test_threshold_constants_are_sane(self):
        """Sanity check on the hardcoded thresholds."""
        assert 0 < MISSING_WARN_THRESHOLD < 1
        assert len(VALID_RANGES) > 0
        assert len(EXPECTED_ROWS) == 4
        for lo, hi in VALID_RANGES.values():
            assert lo < hi
