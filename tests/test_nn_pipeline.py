"""
Tests for corneaforge.nn_pipeline
"""

from unittest.mock import patch

import numpy as np

from corneaforge.core import DEFAULT_SEGMENTS
from corneaforge.nn_pipeline import TARGET_SIZE
from corneaforge.nn_pipeline import process_single_csv as nn_process


def _make_polar(rows=26, cols=256):
    return np.random.uniform(6.0, 9.0, size=(rows, cols))


def _make_segment_dict(segment_names):
    return {name: _make_polar() for name in segment_names}


class TestMissingSegments:
    @patch("corneaforge.nn_pipeline.parse_csv")
    def test_5_of_13_segments_returns_correct_tensor(self, mock_parse):
        present = DEFAULT_SEGMENTS[:5]
        mock_parse.return_value = _make_segment_dict(present)
        tensor, found = nn_process("/fake/patient.csv")
        assert tensor is not None
        assert tensor.shape == (5, TARGET_SIZE, TARGET_SIZE)
        assert tensor.dtype == np.float32
        assert found == present

    @patch("corneaforge.nn_pipeline.parse_csv")
    def test_0_segments_returns_none(self, mock_parse):
        mock_parse.return_value = {}
        tensor, found = nn_process("/fake/patient.csv")
        assert tensor is None
        assert found == []

    @patch("corneaforge.nn_pipeline.parse_csv")
    def test_1_segment_returns_tensor(self, mock_parse):
        present = [DEFAULT_SEGMENTS[0]]
        mock_parse.return_value = _make_segment_dict(present)
        tensor, found = nn_process("/fake/patient.csv")
        assert tensor is not None
        assert tensor.shape == (1, TARGET_SIZE, TARGET_SIZE)
        assert found == present

    @patch("corneaforge.nn_pipeline.parse_csv")
    def test_all_13_segments_returns_full_tensor(self, mock_parse):
        mock_parse.return_value = _make_segment_dict(DEFAULT_SEGMENTS)
        tensor, found = nn_process("/fake/patient.csv")
        assert tensor is not None
        assert tensor.shape == (13, TARGET_SIZE, TARGET_SIZE)
        assert found == DEFAULT_SEGMENTS

    @patch("corneaforge.nn_pipeline.parse_csv")
    def test_custom_target_size(self, mock_parse):
        present = DEFAULT_SEGMENTS[:3]
        mock_parse.return_value = _make_segment_dict(present)
        tensor, found = nn_process("/fake/patient.csv", target_size=64)
        assert tensor.shape == (3, 64, 64)

    @patch("corneaforge.nn_pipeline.parse_csv")
    def test_no_nan_in_output(self, mock_parse):
        present = DEFAULT_SEGMENTS[:5]
        mock_parse.return_value = _make_segment_dict(present)
        tensor, _ = nn_process("/fake/patient.csv", target_size=64)
        assert not np.any(np.isnan(tensor))


class TestWarnings:
    @patch("corneaforge.nn_pipeline.parse_csv")
    def test_warns_for_each_missing_segment(self, mock_parse, caplog):
        present = DEFAULT_SEGMENTS[:5]
        missing = DEFAULT_SEGMENTS[5:]
        mock_parse.return_value = _make_segment_dict(present)
        with caplog.at_level("WARNING", logger="corneaforge.nn"):
            nn_process("/fake/patient.csv")
        for seg_name in missing:
            assert seg_name in caplog.text

    @patch("corneaforge.nn_pipeline.parse_csv")
    def test_no_warnings_when_all_present(self, mock_parse, caplog):
        mock_parse.return_value = _make_segment_dict(DEFAULT_SEGMENTS)
        with caplog.at_level("WARNING", logger="corneaforge.nn"):
            nn_process("/fake/patient.csv")
        assert caplog.text == ""

    @patch("corneaforge.nn_pipeline.parse_csv")
    def test_warnings_do_not_raise(self, mock_parse):
        mock_parse.return_value = {}
        tensor, found = nn_process("/fake/patient.csv")
        assert tensor is None
