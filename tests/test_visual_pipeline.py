"""
Tests for corneaforge.visual_pipeline
"""

import re

import numpy as np
import pytest

mcolors = pytest.importorskip("matplotlib.colors", exc_type=ImportError)

from corneaforge.visual_pipeline import (  # noqa: E402
    CLINICAL_BOUNDS,
    CLINICAL_COLORS,
    CMAP,
)

# Known increasing scales (bounds[0] < bounds[-1])
INCREASING_SEGMENTS = [
    "sagittal_anterior",
    "tangential_anterior",
    "gaussian_anterior",
    "sagittal_posterior",
    "tangential_posterior",
    "gaussian_posterior",
    "corneal_thickness",
    "stromal_thickness",
    "anterior_chamber_depth",
    "refra_frontal_power_posterior",
]

# Known decreasing scales (bounds[0] > bounds[-1])
DECREASING_SEGMENTS = [
    "epithelial_thickness",
    "refra_frontal_power_anterior",
    "refra_equivalent_power",
]


class TestBoundsLength:
    def test_all_segments_have_40_bounds(self):
        for segment, bounds in CLINICAL_BOUNDS.items():
            assert len(bounds) == 40


class TestScaleDirection:
    def test_increasing_scales_are_sorted_ascending(self):
        for seg in INCREASING_SEGMENTS:
            bounds = CLINICAL_BOUNDS[seg]
            assert bounds[0] < bounds[-1]

    def test_decreasing_scales_are_sorted_descending(self):
        for seg in DECREASING_SEGMENTS:
            bounds = CLINICAL_BOUNDS[seg]
            assert bounds[0] > bounds[-1]

    def test_increasing_scale_uses_cmap_not_reversed(self):
        bounds = CLINICAL_BOUNDS["sagittal_anterior"]
        is_decreasing = bounds[0] > bounds[-1]
        assert not is_decreasing
        chosen_cmap = CMAP.reversed() if is_decreasing else CMAP
        assert np.allclose(chosen_cmap(0.0), CMAP(0.0))

    def test_decreasing_scale_uses_cmap_reversed(self):
        bounds = CLINICAL_BOUNDS["epithelial_thickness"]
        is_decreasing = bounds[0] > bounds[-1]
        assert is_decreasing
        chosen_cmap = CMAP.reversed() if is_decreasing else CMAP
        reversed_cmap = CMAP.reversed()
        assert np.allclose(chosen_cmap(0.0), reversed_cmap(0.0))
        assert np.allclose(chosen_cmap(1.0), reversed_cmap(1.0))
        assert not np.allclose(CMAP(0.0), reversed_cmap(0.0))

    def test_all_segments_covered(self):
        covered = set(INCREASING_SEGMENTS) | set(DECREASING_SEGMENTS)
        for seg in CLINICAL_BOUNDS:
            assert seg in covered


class TestValueToColorMapping:
    def test_low_value_maps_to_first_color_increasing(self):
        bounds = CLINICAL_BOUNDS["sagittal_anterior"]
        levels = sorted(bounds)
        norm = mcolors.BoundaryNorm(levels, CMAP.N)
        assert norm(levels[0]) == 0

    def test_high_value_maps_to_last_color_increasing(self):
        bounds = CLINICAL_BOUNDS["sagittal_anterior"]
        levels = sorted(bounds)
        norm = mcolors.BoundaryNorm(levels, CMAP.N)
        assert norm(levels[-1]) >= CMAP.N - 1

    def test_low_value_maps_to_first_color_decreasing(self):
        bounds = CLINICAL_BOUNDS["epithelial_thickness"]
        levels = sorted(bounds)
        reversed_cmap = CMAP.reversed()
        norm = mcolors.BoundaryNorm(levels, reversed_cmap.N)
        assert norm(levels[0]) == 0

    def test_mid_value_maps_to_middle_color(self):
        bounds = CLINICAL_BOUNDS["sagittal_anterior"]
        levels = sorted(bounds)
        norm = mcolors.BoundaryNorm(levels, CMAP.N)
        mid_value = (levels[0] + levels[-1]) / 2.0
        idx = norm(mid_value)
        assert 5 < idx < CMAP.N - 5


class TestNaNRendering:
    def test_nan_maps_to_white(self):
        bad_color = CMAP(np.ma.masked)
        assert np.allclose(bad_color, [1.0, 1.0, 1.0, 1.0])

    def test_masked_array_nan_renders_white(self):
        bounds = CLINICAL_BOUNDS["sagittal_anterior"]
        levels = sorted(bounds)
        norm = mcolors.BoundaryNorm(levels, CMAP.N)
        data = np.array([[7.0, np.nan], [np.nan, 8.0]])
        normalized = norm(np.ma.masked_invalid(data))
        rgba = CMAP(normalized)
        assert np.allclose(rgba[0, 1], [1.0, 1.0, 1.0, 1.0])
        assert np.allclose(rgba[1, 0], [1.0, 1.0, 1.0, 1.0])


class TestColorCount:
    def test_exactly_40_colors(self):
        assert len(CLINICAL_COLORS) == 40

    def test_cmap_has_40_colors(self):
        assert CMAP.N == 40


class TestHexColorValidity:
    HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

    def test_all_colors_match_hex_pattern(self):
        for i, color in enumerate(CLINICAL_COLORS):
            assert self.HEX_PATTERN.match(color)

    def test_all_colors_parseable_by_matplotlib(self):
        for i, color in enumerate(CLINICAL_COLORS):
            rgba = mcolors.to_rgba(color)
            assert len(rgba) == 4
            assert all(0.0 <= c <= 1.0 for c in rgba)


class TestColormapPipelineIntegration:
    def test_increasing_segment_full_pipeline(self):
        bounds = CLINICAL_BOUNDS["sagittal_anterior"]
        levels = sorted(bounds)
        norm = mcolors.BoundaryNorm(levels, CMAP.N)
        vals = np.linspace(levels[0], levels[-1], 15)
        matrix = np.empty((4, 4), dtype=float)
        matrix.flat[:15] = vals
        matrix.flat[15] = np.nan
        normalized = norm(np.ma.masked_invalid(matrix))
        rgba = CMAP(normalized)
        assert rgba.shape == (4, 4, 4)
        assert np.allclose(rgba[3, 3], [1.0, 1.0, 1.0, 1.0])
        for r in range(4):
            for c in range(4):
                if r == 3 and c == 3:
                    continue
                assert not np.allclose(rgba[r, c], [1.0, 1.0, 1.0, 1.0])

    def test_decreasing_segment_full_pipeline(self):
        bounds = CLINICAL_BOUNDS["epithelial_thickness"]
        levels = sorted(bounds)
        reversed_cmap = CMAP.reversed()
        norm = mcolors.BoundaryNorm(levels, reversed_cmap.N)
        low_color = reversed_cmap(norm(levels[0]))
        high_color = reversed_cmap(norm(levels[-1]))
        assert not np.allclose(low_color, high_color)
        assert np.allclose(reversed_cmap(0.0), CMAP(1.0), atol=0.01)
        assert np.allclose(reversed_cmap(1.0), CMAP(0.0), atol=0.01)
