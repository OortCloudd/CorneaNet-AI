"""
Tests for corneaforge.descriptive_stats
"""

from unittest.mock import patch

import numpy as np

from corneaforge.descriptive_stats import SEGMENTS_ENABLED, _angular_sectors
from corneaforge.descriptive_stats import process_single_csv as stats_process


# ==========================================================================
# Helpers
# ==========================================================================

FAKE_METADATA = {
    "Patient_Last_Name": "Test",
    "Patient_First_Name": "Patient",
    "Patient_ID": "P000000001",
    "Patient_Date_of_Birth": "01/01/1990",
    "Patient_Gender": "M",
    "Exam_Eye": "OD",
    "Exam_Scan_Date": "01/01/2025",
}

EXPECTED_SECTORS = {"superior", "inferior", "nasal", "temporal"}
TOTAL_COLUMNS = 256
COLUMNS_PER_SECTOR = 64


def _make_polar(rows=26, cols=256):
    return np.random.uniform(6.0, 9.0, size=(rows, cols))


def _make_segment_dict(segment_names):
    return {name: _make_polar() for name in segment_names}


def _sector_columns(ranges):
    cols = set()
    for start, end in ranges:
        cols.update(range(start, end))
    return cols


def _count_columns(ranges):
    return sum(end - start for start, end in ranges)


def _enabled_segment_names():
    return [name for name, on in SEGMENTS_ENABLED.items() if on]


# ==========================================================================
# OD/OS angular sectors (formerly test_risk7_od_os.py)
# ==========================================================================


class TestSectorNames:
    def test_od_returns_four_sectors(self):
        assert set(_angular_sectors("OD").keys()) == EXPECTED_SECTORS

    def test_os_returns_four_sectors(self):
        assert set(_angular_sectors("OS").keys()) == EXPECTED_SECTORS


class TestSuperiorInferiorUnchanged:
    def test_superior_identical(self):
        assert _angular_sectors("OD")["superior"] == _angular_sectors("OS")["superior"]

    def test_inferior_identical(self):
        assert _angular_sectors("OD")["inferior"] == _angular_sectors("OS")["inferior"]


class TestNasalTemporalSwap:
    def test_od_nasal_equals_os_temporal(self):
        od = _angular_sectors("OD")
        os = _angular_sectors("OS")
        assert _sector_columns(od["nasal"]) == _sector_columns(os["temporal"])

    def test_od_temporal_equals_os_nasal(self):
        od = _angular_sectors("OD")
        os = _angular_sectors("OS")
        assert _sector_columns(od["temporal"]) == _sector_columns(os["nasal"])

    def test_nasal_and_temporal_are_different_within_same_eye(self):
        od = _angular_sectors("OD")
        assert _sector_columns(od["nasal"]) != _sector_columns(od["temporal"])
        os = _angular_sectors("OS")
        assert _sector_columns(os["nasal"]) != _sector_columns(os["temporal"])


class TestFullCoverageNoOverlap:
    def test_od_full_coverage(self):
        sectors = _angular_sectors("OD")
        all_cols = set()
        for ranges in sectors.values():
            cols = _sector_columns(ranges)
            assert all_cols.isdisjoint(cols), "OD sectors overlap"
            all_cols.update(cols)
        assert all_cols == set(range(TOTAL_COLUMNS))

    def test_os_full_coverage(self):
        sectors = _angular_sectors("OS")
        all_cols = set()
        for ranges in sectors.values():
            cols = _sector_columns(ranges)
            assert all_cols.isdisjoint(cols), "OS sectors overlap"
            all_cols.update(cols)
        assert all_cols == set(range(TOTAL_COLUMNS))


class TestSectorSize:
    def test_od_sector_sizes(self):
        for name, ranges in _angular_sectors("OD").items():
            assert _count_columns(ranges) == COLUMNS_PER_SECTOR

    def test_os_sector_sizes(self):
        for name, ranges in _angular_sectors("OS").items():
            assert _count_columns(ranges) == COLUMNS_PER_SECTOR


class TestInvalidEyeValues:
    def test_empty_string_treated_as_os(self):
        assert _angular_sectors("") == _angular_sectors("OS")

    def test_xx_treated_as_os(self):
        assert _angular_sectors("XX") == _angular_sectors("OS")

    def test_lowercase_od_treated_as_os(self):
        assert _angular_sectors("od") == _angular_sectors("OS")

    def test_none_treated_as_os(self):
        assert _angular_sectors(None) == _angular_sectors("OS")


# ==========================================================================
# Missing segments (formerly test_risk8_missing.py stats parts)
# ==========================================================================


class TestMissingSegments:
    @patch("corneaforge.descriptive_stats.parse_metadata")
    @patch("corneaforge.descriptive_stats.parse_csv")
    def test_3_of_enabled_segments(self, mock_parse, mock_meta):
        enabled = _enabled_segment_names()
        present = enabled[:3]
        mock_parse.return_value = _make_segment_dict(present)
        mock_meta.return_value = FAKE_METADATA.copy()
        row = stats_process("/fake/patient.csv")
        assert row["patient_id"] == "P000000001"
        assert row["exam_eye"] == "OD"
        for seg in present:
            assert any(k.startswith(seg) for k in row)
        for seg in enabled[3:]:
            assert not any(k.startswith(seg) for k in row)

    @patch("corneaforge.descriptive_stats.parse_metadata")
    @patch("corneaforge.descriptive_stats.parse_csv")
    def test_0_segments_returns_metadata_only(self, mock_parse, mock_meta):
        mock_parse.return_value = {}
        mock_meta.return_value = FAKE_METADATA.copy()
        row = stats_process("/fake/patient.csv")
        assert row["patient_id"] == "P000000001"
        for seg in _enabled_segment_names():
            assert not any(k.startswith(seg) for k in row)

    @patch("corneaforge.descriptive_stats.parse_metadata")
    @patch("corneaforge.descriptive_stats.parse_csv")
    def test_all_enabled_segments_present(self, mock_parse, mock_meta):
        enabled = _enabled_segment_names()
        mock_parse.return_value = _make_segment_dict(enabled)
        mock_meta.return_value = FAKE_METADATA.copy()
        row = stats_process("/fake/patient.csv")
        for seg in enabled:
            assert any(k.startswith(seg) for k in row)

    @patch("corneaforge.descriptive_stats.parse_metadata")
    @patch("corneaforge.descriptive_stats.parse_csv")
    def test_disabled_segments_ignored_even_if_present(self, mock_parse, mock_meta):
        disabled = [name for name, on in SEGMENTS_ENABLED.items() if not on]
        assert len(disabled) > 0
        mock_parse.return_value = _make_segment_dict(disabled)
        mock_meta.return_value = FAKE_METADATA.copy()
        row = stats_process("/fake/patient.csv")
        for seg in disabled:
            assert not any(k.startswith(seg) for k in row)


class TestNoException:
    @patch("corneaforge.descriptive_stats.parse_metadata")
    @patch("corneaforge.descriptive_stats.parse_csv")
    def test_empty_parse_does_not_raise(self, mock_parse, mock_meta):
        mock_parse.return_value = {}
        mock_meta.return_value = FAKE_METADATA.copy()
        row = stats_process("/fake/patient.csv")
        assert isinstance(row, dict)

    @patch("corneaforge.descriptive_stats.parse_metadata")
    @patch("corneaforge.descriptive_stats.parse_csv")
    def test_partial_segments_does_not_raise(self, mock_parse, mock_meta):
        enabled = _enabled_segment_names()
        mock_parse.return_value = _make_segment_dict(enabled[:1])
        mock_meta.return_value = FAKE_METADATA.copy()
        row = stats_process("/fake/patient.csv")
        assert isinstance(row, dict)

    @patch("corneaforge.descriptive_stats.parse_metadata")
    @patch("corneaforge.descriptive_stats.parse_csv")
    def test_os_eye_with_partial_segments(self, mock_parse, mock_meta):
        enabled = _enabled_segment_names()
        mock_parse.return_value = _make_segment_dict(enabled[:2])
        meta = FAKE_METADATA.copy()
        meta["Exam_Eye"] = "OS"
        mock_meta.return_value = meta
        row = stats_process("/fake/patient.csv")
        assert isinstance(row, dict)
        assert row["exam_eye"] == "OS"


class TestFeatureTypes:
    @patch("corneaforge.descriptive_stats.parse_metadata")
    @patch("corneaforge.descriptive_stats.parse_csv")
    def test_feature_values_are_numeric(self, mock_parse, mock_meta):
        enabled = _enabled_segment_names()
        mock_parse.return_value = _make_segment_dict(enabled[:3])
        mock_meta.return_value = FAKE_METADATA.copy()
        row = stats_process("/fake/patient.csv")
        metadata_keys = {
            "filename", "patient_last_name", "patient_first_name",
            "patient_id", "patient_dob", "patient_gender", "exam_eye", "exam_date",
        }
        for key, val in row.items():
            if key in metadata_keys:
                continue
            assert isinstance(val, (int, float, np.integer, np.floating)), (
                f"Feature '{key}' has non-numeric type {type(val)}"
            )
