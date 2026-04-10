"""
Tests for corneaforge.core

These tests use synthetic data (not real patient data) to verify that the
parsing and conversion logic works correctly. No CSV files are needed.
"""

import numpy as np
import pytest
from scipy.spatial import QhullError

from corneaforge.core import (
    ALL_SEGMENTS,
    DEFAULT_SEGMENTS,
    SEGMENT_HEADERS,
    _identify_header,
    _parse_data_rows,
    clean_polar_data,
    missing_data_mask,
    parse_csv,
    parse_metadata,
    polar_to_cartesian,
)

# ==========================================================================
# Header detection
# ==========================================================================


class TestIdentifyHeader:
    def test_recognizes_known_headers(self):
        assert _identify_header("SagittalAnterior [mm]") == "sagittal_anterior"
        assert _identify_header("CornealThickness [µm]") == "corneal_thickness"
        assert _identify_header("EpitelialThickness [µm]") == "epithelial_thickness"

    def test_ignores_non_headers(self):
        assert _identify_header("1.23;4.56;7.89") is None
        assert _identify_header("##########") is None
        assert _identify_header("Patient_ID;P001") is None
        assert _identify_header("") is None

    def test_all_registered_headers_are_detectable(self):
        for header_text, internal_name in SEGMENT_HEADERS.items():
            # Simulate the CSV format: "HeaderText [unit]"
            line = f"{header_text} [mm]"
            assert _identify_header(line) == internal_name


# ==========================================================================
# Data row parsing
# ==========================================================================


class TestParseDataRows:
    def test_reads_numeric_rows(self):
        lines = [
            ";".join(str(float(i)) for i in range(256)),
            ";".join(str(float(i + 1)) for i in range(256)),
            "SomeHeader [mm]",  # stops here
        ]
        result = _parse_data_rows(lines, 0)
        assert result is not None
        assert result.shape == (2, 256)
        assert result[0, 0] == 0.0
        assert result[1, 0] == 1.0

    def test_stops_at_non_numeric(self):
        lines = [
            ";".join("1.0" for _ in range(256)),
            "NotANumber;text;here",
        ]
        result = _parse_data_rows(lines, 0)
        assert result.shape == (1, 256)

    def test_stops_at_short_row(self):
        lines = [
            ";".join("1.0" for _ in range(256)),
            ";".join("1.0" for _ in range(10)),  # too few columns
        ]
        result = _parse_data_rows(lines, 0)
        assert result.shape == (1, 256)

    def test_returns_none_for_no_data(self):
        lines = ["NotNumeric"]
        assert _parse_data_rows(lines, 0) is None


# ==========================================================================
# Polar data cleaning
# ==========================================================================


class TestCleanPolarData:
    def _make_polar(self):
        """Create synthetic polar data: 27 real rows + 5 padding rows of -1000."""
        real_data = np.random.uniform(6.0, 9.0, size=(27, 256))
        # Add some sparse -1000 in the last 2 real rows (simulating peripheral gaps)
        real_data[25, 200:210] = -1000
        real_data[26, 195:215] = -1000
        padding = np.full((5, 256), -1000.0)
        return np.vstack([real_data, padding])

    def test_removes_padding_rows(self):
        raw = self._make_polar()
        assert raw.shape[0] == 32
        cleaned = clean_polar_data(raw, keep_missing=False)
        assert cleaned.shape[0] == 27  # only real rows remain

    def test_keep_missing_false_replaces_sentinels(self):
        raw = self._make_polar()
        cleaned = clean_polar_data(raw, keep_missing=False)
        assert not np.any(cleaned == -1000)
        # -1000 cells should now be NaN
        assert np.any(np.isnan(cleaned))

    def test_keep_missing_true_preserves_sentinels(self):
        raw = self._make_polar()
        cleaned = clean_polar_data(raw, keep_missing=True)
        assert cleaned.shape[0] == 27  # padding still removed
        assert np.any(cleaned == -1000)  # sparse -1000 preserved


# ==========================================================================
# Polar → Cartesian conversion
# ==========================================================================


class TestPolarToCartesian:
    def _make_clean_polar(self, n_rows=27):
        """Synthetic polar data, already cleaned (no -1000, no padding)."""
        return np.random.uniform(6.0, 9.0, size=(n_rows, 256)).astype(np.float64)

    def test_output_shape(self):
        polar = self._make_clean_polar()
        result = polar_to_cartesian(polar, target_size=64)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_circular_mask(self):
        """Corners of the square output should be NaN (outside the unit circle)."""
        polar = self._make_clean_polar()
        result = polar_to_cartesian(polar, target_size=100)
        # The four corners should definitely be NaN
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, -1])
        assert np.isnan(result[-1, 0])
        assert np.isnan(result[-1, -1])
        # The center should NOT be NaN
        assert not np.isnan(result[50, 50])

    def test_nan_fraction_matches_circle(self):
        """~21.5% of a square inscribing a unit circle is outside the circle."""
        polar = self._make_clean_polar()
        result = polar_to_cartesian(polar, target_size=224)
        nan_frac = np.isnan(result).sum() / result.size
        # pi/4 ≈ 0.785, so outside = 1 - pi/4 ≈ 0.215
        assert 0.18 < nan_frac < 0.25

    def test_all_nan_input(self):
        """If the polar data is entirely NaN, output should be all NaN."""
        polar = np.full((27, 256), np.nan)
        result = polar_to_cartesian(polar, target_size=64)
        assert np.all(np.isnan(result))


# ==========================================================================
# Missing data mask
# ==========================================================================


class TestMissingDataMask:
    def test_no_missing_data(self):
        polar = np.random.uniform(6.0, 9.0, size=(27, 256))
        mask = missing_data_mask(polar, target_size=64)
        assert mask.shape == (64, 64)
        assert not np.any(mask)  # no -1000 → no True values

    def test_with_missing_data(self):
        polar = np.random.uniform(6.0, 9.0, size=(27, 256))
        polar[26, :] = -1000  # entire outer ring missing
        mask = missing_data_mask(polar, target_size=64)
        assert np.any(mask)  # should have some True values
        assert mask.dtype == bool


# ==========================================================================
# Segment constants
# ==========================================================================


class TestConstants:
    def test_default_segments_count(self):
        assert len(DEFAULT_SEGMENTS) == 13

    def test_all_segments_includes_elevations(self):
        assert len(ALL_SEGMENTS) == 16
        assert "elevation_anterior" in ALL_SEGMENTS
        assert "elevation_anterior" not in DEFAULT_SEGMENTS

    def test_all_headers_map_to_segments(self):
        all_internal = set(ALL_SEGMENTS)
        for internal_name in SEGMENT_HEADERS.values():
            assert internal_name in all_internal


# ==========================================================================
# Edge cases (formerly test_risk_edges.py)
# ==========================================================================


class TestCleanPolarDataEdges:
    def test_no_padding_returns_unchanged(self):
        rng = np.random.default_rng(42)
        arr = rng.uniform(40.0, 50.0, size=(22, 256))
        result = clean_polar_data(arr)
        np.testing.assert_array_equal(result, arr)

    def test_keep_missing_removes_full_rows_but_keeps_sparse(self):
        rng = np.random.default_rng(99)
        real_data = rng.uniform(40.0, 50.0, size=(22, 256))
        real_data[0, 0] = -1000.0
        real_data[5, 128] = -1000.0
        real_data[21, 255] = -1000.0
        padding = np.full((10, 256), -1000.0)
        arr = np.vstack([real_data, padding])
        result = clean_polar_data(arr, keep_missing=True)
        assert result.shape == (22, 256)
        assert result[0, 0] == -1000.0
        assert result[5, 128] == -1000.0
        assert result[21, 255] == -1000.0


class TestPolarToCartesianEdges:
    def test_single_radial_row_is_degenerate(self):
        arr = np.full((1, 256), 42.0)
        with pytest.raises(QhullError):
            polar_to_cartesian(arr, target_size=32)

    def test_two_radial_rows(self):
        arr = np.full((2, 256), 42.0)
        result = polar_to_cartesian(arr, target_size=32)
        assert result.shape == (32, 32)
        assert result.dtype == np.float32
        assert np.isfinite(result[16, 16])

    def test_large_polar_array(self):
        rng = np.random.default_rng(123)
        arr = rng.uniform(40.0, 50.0, size=(50, 256))
        result = polar_to_cartesian(arr, target_size=64)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        assert np.isfinite(result[32, 32])


class TestParseDataRowsEdges:
    def test_start_index_beyond_list(self):
        lines = ["1;2;3"]
        result = _parse_data_rows(lines, start_idx=100)
        assert result is None


class TestMissingDataMaskEdges:
    def test_all_missing_data(self):
        arr = np.full((22, 256), -1000.0)
        mask = missing_data_mask(arr, target_size=64)
        assert mask.shape == (64, 64)
        assert mask.dtype == bool
        axis = np.linspace(-1, 1, 64)
        xi, yi = np.meshgrid(axis, axis)
        inside_circle = (xi**2 + yi**2) <= 1.0
        inside_and_masked = mask[inside_circle].sum()
        total_inside = inside_circle.sum()
        assert inside_and_masked / total_inside > 0.9


# ==========================================================================
# Interior NaN (formerly test_risk4_interior.py)
# ==========================================================================

MS39_RADIAL_COUNTS = [21, 22, 25, 26]
TARGET_SIZES = [224, 512]


def _interior_mask(size, r_max=0.99):
    axis = np.linspace(-1, 1, size)
    xi, yi = np.meshgrid(axis, axis)
    return (xi**2 + yi**2) < r_max**2


def _make_clean_polar(n_radial, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(6.0, 9.0, size=(n_radial, 256)).astype(np.float64)


def _make_sparse_nan_polar(n_radial, nan_fraction=0.02, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.uniform(6.0, 9.0, size=(n_radial, 256)).astype(np.float64)
    n_cells = n_radial * 256
    n_nan = int(n_cells * nan_fraction)
    nan_indices = rng.choice(n_cells, size=n_nan, replace=False)
    data.ravel()[nan_indices] = np.nan
    return data


class TestNoInteriorNaN:
    @pytest.mark.parametrize("n_radial", MS39_RADIAL_COUNTS)
    @pytest.mark.parametrize("target_size", TARGET_SIZES)
    def test_no_nan_inside_circle(self, n_radial, target_size):
        polar = _make_clean_polar(n_radial)
        result = polar_to_cartesian(polar, target_size=target_size)
        interior = _interior_mask(target_size)
        n_nan_inside = np.isnan(result[interior]).sum()
        assert n_nan_inside == 0, (
            f"Found {n_nan_inside} NaN pixels inside r<0.99 "
            f"(n_radial={n_radial}, target_size={target_size})"
        )

    @pytest.mark.parametrize("target_size", TARGET_SIZES)
    def test_center_pixel_is_finite(self, target_size):
        polar = _make_clean_polar(n_radial=22)
        result = polar_to_cartesian(polar, target_size=target_size)
        mid = target_size // 2
        assert np.isfinite(result[mid, mid])

    def test_constant_input_produces_constant_output(self):
        value = 7.5
        polar = np.full((22, 256), value, dtype=np.float64)
        result = polar_to_cartesian(polar, target_size=64)
        interior = _interior_mask(64)
        interior_values = result[interior]
        assert not np.any(np.isnan(interior_values))
        np.testing.assert_allclose(interior_values, value, atol=1e-4)


class TestSparseNaNContainment:
    @pytest.mark.parametrize("n_radial", MS39_RADIAL_COUNTS)
    @pytest.mark.parametrize("target_size", TARGET_SIZES)
    def test_sparse_nan_limited_spread(self, n_radial, target_size):
        polar = _make_sparse_nan_polar(n_radial, nan_fraction=0.02)
        result = polar_to_cartesian(polar, target_size=target_size)
        interior = _interior_mask(target_size)
        n_interior = interior.sum()
        n_nan_inside = np.isnan(result[interior]).sum()
        nan_ratio = n_nan_inside / n_interior
        assert nan_ratio < 0.15

    @pytest.mark.parametrize("n_radial", MS39_RADIAL_COUNTS)
    def test_sparse_nan_center_survives(self, n_radial):
        polar = _make_sparse_nan_polar(n_radial, nan_fraction=0.02)
        polar[0, :] = 7.5
        polar[1, :] = 7.5
        result = polar_to_cartesian(polar, target_size=224)
        mid = 224 // 2
        center_patch = result[mid - 2 : mid + 3, mid - 2 : mid + 3]
        assert np.all(np.isfinite(center_patch))

    def test_peripheral_nan_does_not_corrupt_center(self):
        polar = _make_clean_polar(n_radial=26)
        rng = np.random.default_rng(99)
        for row in [23, 24, 25]:
            mask = rng.random(256) < 0.5
            polar[row, mask] = np.nan
        result = polar_to_cartesian(polar, target_size=224)
        axis = np.linspace(-1, 1, 224)
        xi, yi = np.meshgrid(axis, axis)
        inner_region = (xi**2 + yi**2) < 0.7**2
        assert np.all(np.isfinite(result[inner_region]))


class TestInteriorEdgeCases:
    def test_single_nan_cell_does_not_propagate(self):
        polar = _make_clean_polar(n_radial=22)
        polar[10, 128] = np.nan
        result = polar_to_cartesian(polar, target_size=224)
        interior = _interior_mask(224)
        n_nan_inside = np.isnan(result[interior]).sum()
        assert n_nan_inside < 50

    def test_entire_row_nan_still_limited(self):
        polar = _make_clean_polar(n_radial=22)
        polar[15, :] = np.nan
        result = polar_to_cartesian(polar, target_size=224)
        interior = _interior_mask(224)
        n_interior = interior.sum()
        n_nan_inside = np.isnan(result[interior]).sum()
        nan_ratio = n_nan_inside / n_interior
        assert nan_ratio < 0.30

    @pytest.mark.parametrize("target_size", TARGET_SIZES)
    def test_minimum_radial_rows(self, target_size):
        polar = _make_clean_polar(n_radial=21)
        result = polar_to_cartesian(polar, target_size=target_size)
        interior = _interior_mask(target_size)
        n_nan_inside = np.isnan(result[interior]).sum()
        assert n_nan_inside == 0


# ==========================================================================
# Sentinel handling (formerly test_risk6_sentinel.py)
# ==========================================================================

SENTINEL = -1000.0
N_COLS = 256


def _make_polar_with_padding(n_real_rows=27, n_padding_rows=5):
    real_data = np.random.default_rng(42).uniform(6.0, 9.0, size=(n_real_rows, N_COLS))
    padding = np.full((n_padding_rows, N_COLS), SENTINEL)
    return np.vstack([real_data, padding])


def _inject_sentinels(polar, positions):
    for row, cols in positions:
        polar[row, cols] = SENTINEL
    return polar


def _full_nn_chain(polar_raw, target_size=64):
    cleaned = clean_polar_data(polar_raw, keep_missing=False)
    cartesian = polar_to_cartesian(cleaned, target_size=target_size)
    return np.nan_to_num(cartesian, nan=0.0)


class TestCleanRemovesAllSentinels:
    def test_removes_sparse_sentinels_in_periphery(self):
        raw = _make_polar_with_padding()
        _inject_sentinels(raw, [(25, slice(200, 230)), (26, slice(180, 256))])
        cleaned = clean_polar_data(raw, keep_missing=False)
        assert not np.any(cleaned == SENTINEL)

    def test_removes_sentinels_at_center(self):
        raw = _make_polar_with_padding()
        _inject_sentinels(raw, [(0, slice(0, 10)), (1, slice(120, 130))])
        cleaned = clean_polar_data(raw, keep_missing=False)
        assert not np.any(cleaned == SENTINEL)

    def test_removes_scattered_sentinels(self):
        raw = _make_polar_with_padding()
        rng = np.random.default_rng(99)
        scatter_rows = rng.integers(0, 27, size=50)
        scatter_cols = rng.integers(0, N_COLS, size=50)
        for r, c in zip(scatter_rows, scatter_cols):
            raw[r, c] = SENTINEL
        cleaned = clean_polar_data(raw, keep_missing=False)
        assert not np.any(cleaned == SENTINEL)


class TestCleanReplacesWithNaN:
    def test_sparse_sentinels_become_nan(self):
        raw = _make_polar_with_padding()
        _inject_sentinels(raw, [(25, slice(200, 210))])
        cleaned = clean_polar_data(raw, keep_missing=False)
        assert np.all(np.isnan(cleaned[25, 200:210]))

    def test_sparse_sentinels_are_not_zero(self):
        raw = _make_polar_with_padding()
        _inject_sentinels(raw, [(25, slice(200, 210))])
        cleaned = clean_polar_data(raw, keep_missing=False)
        for col in range(200, 210):
            assert np.isnan(cleaned[25, col])

    def test_non_sentinel_values_unchanged(self):
        raw = _make_polar_with_padding()
        original_value = raw[10, 128]
        _inject_sentinels(raw, [(25, slice(200, 210))])
        cleaned = clean_polar_data(raw, keep_missing=False)
        assert cleaned[10, 128] == pytest.approx(original_value)

    def test_center_sentinels_become_nan(self):
        raw = _make_polar_with_padding()
        _inject_sentinels(raw, [(0, 0)])
        cleaned = clean_polar_data(raw, keep_missing=False)
        assert np.isnan(cleaned[0, 0])


class TestFinalTensorClean:
    def test_basic_no_nan_no_sentinel(self):
        raw = _make_polar_with_padding()
        result = _full_nn_chain(raw)
        assert not np.any(np.isnan(result))
        assert not np.any(result == SENTINEL)

    def test_peripheral_gaps_no_nan_no_sentinel(self):
        raw = _make_polar_with_padding()
        _inject_sentinels(raw, [(25, slice(200, 230)), (26, slice(180, 256))])
        result = _full_nn_chain(raw)
        assert not np.any(np.isnan(result))
        assert not np.any(result == SENTINEL)

    def test_center_gaps_no_nan_no_sentinel(self):
        raw = _make_polar_with_padding()
        _inject_sentinels(raw, [(0, slice(0, 20)), (1, slice(100, 150))])
        result = _full_nn_chain(raw)
        assert not np.any(np.isnan(result))
        assert not np.any(result == SENTINEL)

    def test_scattered_gaps_no_nan_no_sentinel(self):
        raw = _make_polar_with_padding()
        rng = np.random.default_rng(55)
        scatter_rows = rng.integers(0, 27, size=40)
        scatter_cols = rng.integers(0, N_COLS, size=40)
        for r, c in zip(scatter_rows, scatter_cols):
            raw[r, c] = SENTINEL
        result = _full_nn_chain(raw)
        assert not np.any(np.isnan(result))
        assert not np.any(result == SENTINEL)

    def test_entire_row_sentinel_no_nan_no_sentinel(self):
        raw = _make_polar_with_padding()
        raw[26, :] = SENTINEL
        result = _full_nn_chain(raw)
        assert not np.any(np.isnan(result))
        assert not np.any(result == SENTINEL)

    def test_output_dtype_is_float32(self):
        raw = _make_polar_with_padding()
        result = _full_nn_chain(raw)
        assert result.dtype == np.float32


class TestSentinelPositionVariants:
    @pytest.mark.parametrize(
        "label,positions",
        [
            ("center_single_cell", [(0, 0)]),
            ("center_block", [(0, slice(0, 30)), (1, slice(0, 30))]),
            ("periphery_last_row_block", [(26, slice(200, 256))]),
            ("periphery_two_rows", [(25, slice(200, 256)), (26, slice(180, 256))]),
            ("mid_ring_block", [(13, slice(100, 120))]),
            ("first_and_last_column", [(10, 0), (10, 255)]),
        ],
        ids=lambda p: p if isinstance(p, str) else None,
    )
    def test_position_variant(self, label, positions):
        raw = _make_polar_with_padding()
        _inject_sentinels(raw, positions)
        result = _full_nn_chain(raw)
        assert not np.any(np.isnan(result))
        assert not np.any(result == SENTINEL)

    def test_entire_outer_ring_sentinel(self):
        raw = _make_polar_with_padding()
        raw[26, :] = SENTINEL
        result = _full_nn_chain(raw)
        assert not np.any(np.isnan(result))
        assert not np.any(result == SENTINEL)

    def test_multiple_entire_rows_sentinel(self):
        raw = _make_polar_with_padding()
        raw[24, :] = SENTINEL
        raw[25, :] = SENTINEL
        raw[26, :] = SENTINEL
        result = _full_nn_chain(raw)
        assert not np.any(np.isnan(result))
        assert not np.any(result == SENTINEL)

    def test_heavy_scatter_20_percent(self):
        raw = _make_polar_with_padding()
        rng = np.random.default_rng(123)
        n_cells = 27 * N_COLS
        n_sentinel = int(0.20 * n_cells)
        flat_indices = rng.choice(n_cells, size=n_sentinel, replace=False)
        rows, cols = np.unravel_index(flat_indices, (27, N_COLS))
        raw[rows, cols] = SENTINEL
        result = _full_nn_chain(raw)
        assert not np.any(np.isnan(result))
        assert not np.any(result == SENTINEL)


# ==========================================================================
# Malformed CSV input (formerly test_risk9_malformed.py)
# ==========================================================================


def _write_csv(tmp_path, content, name="test.csv"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _make_data_row(ncols=256, value=7.5):
    return ";".join(f"{value}" for _ in range(ncols))


class TestEmptyFile:
    def test_parse_csv_returns_empty_dict(self, tmp_path):
        path = _write_csv(tmp_path, "")
        result = parse_csv(path)
        assert result == {}

    def test_parse_metadata_returns_empty_dict(self, tmp_path):
        path = _write_csv(tmp_path, "")
        result = parse_metadata(path)
        assert result == {}


class TestMetadataOnly:
    CONTENT = "Patient_ID;P0433494363\nExam_Eye;OD\nExam_Scan_Date;06/03/2024\n"

    def test_parse_csv_returns_empty_dict(self, tmp_path):
        path = _write_csv(tmp_path, self.CONTENT)
        result = parse_csv(path)
        assert result == {}

    def test_parse_metadata_extracts_fields(self, tmp_path):
        path = _write_csv(tmp_path, self.CONTENT)
        meta = parse_metadata(path)
        assert meta["Patient_ID"] == "P0433494363"
        assert meta["Exam_Eye"] == "OD"


class TestHeaderWithoutDataRows:
    def test_empty_after_header(self, tmp_path):
        path = _write_csv(tmp_path, "SagittalAnterior [mm]\n")
        result = parse_csv(path)
        assert result == {}

    def test_non_numeric_after_header(self, tmp_path):
        path = _write_csv(tmp_path, "SagittalAnterior [mm]\nsome;random;text\n")
        result = parse_csv(path)
        assert result == {}

    def test_blank_line_after_header(self, tmp_path):
        content = "SagittalAnterior [mm]\n\nCornealThickness [µm]\n"
        path = _write_csv(tmp_path, content)
        result = parse_csv(path)
        assert result == {}


class TestBinaryGarbage:
    def test_parse_csv_raises_on_binary(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_bytes(b"\x80\x81\x82\xff\xfe\x00\x01" * 100)
        with pytest.raises((UnicodeDecodeError, ValueError)):
            parse_csv(str(path))

    def test_parse_metadata_raises_on_binary(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_bytes(b"\x80\x81\x82\xff\xfe\x00\x01" * 100)
        with pytest.raises((UnicodeDecodeError, ValueError)):
            parse_metadata(str(path))


class TestShortRows:
    def test_all_rows_too_short(self, tmp_path):
        rows = "\n".join(_make_data_row(ncols=10) for _ in range(5))
        content = f"SagittalAnterior [mm]\n{rows}\n"
        path = _write_csv(tmp_path, content)
        result = parse_csv(path)
        assert "sagittal_anterior" not in result

    def test_one_valid_row_then_short(self, tmp_path):
        good_row = _make_data_row(ncols=256)
        short_row = _make_data_row(ncols=10)
        content = f"SagittalAnterior [mm]\n{good_row}\n{short_row}\n"
        path = _write_csv(tmp_path, content)
        result = parse_csv(path)
        assert "sagittal_anterior" in result
        assert result["sagittal_anterior"].shape == (1, 256)


class TestMinimalValidInput:
    def test_single_segment_parses(self, tmp_path):
        n_rows = 5
        rows = "\n".join(_make_data_row(ncols=256, value=7.5) for _ in range(n_rows))
        content = f"SagittalAnterior [mm]\n{rows}\n"
        path = _write_csv(tmp_path, content)
        result = parse_csv(path)
        assert "sagittal_anterior" in result
        arr = result["sagittal_anterior"]
        assert arr.shape == (n_rows, 256)
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, 7.5)

    def test_two_segments_parse(self, tmp_path):
        rows_a = "\n".join(_make_data_row(ncols=256, value=1.0) for _ in range(3))
        rows_b = "\n".join(_make_data_row(ncols=256, value=2.0) for _ in range(4))
        content = f"SagittalAnterior [mm]\n{rows_a}\n\nCornealThickness [µm]\n{rows_b}\n"
        path = _write_csv(tmp_path, content)
        result = parse_csv(path)
        assert "sagittal_anterior" in result
        assert "corneal_thickness" in result
        assert result["sagittal_anterior"].shape == (3, 256)
        assert result["corneal_thickness"].shape == (4, 256)

    def test_metadata_before_segment(self, tmp_path):
        rows = "\n".join(_make_data_row(ncols=256, value=8.0) for _ in range(2))
        content = f"Patient_ID;P001\nExam_Eye;OS\nSagittalAnterior [mm]\n{rows}\n"
        path = _write_csv(tmp_path, content)
        result = parse_csv(path)
        assert "sagittal_anterior" in result
        meta = parse_metadata(str(tmp_path / "test.csv"))
        assert meta["Patient_ID"] == "P001"
        assert meta["Exam_Eye"] == "OS"
