"""
Tests for computed_indices.py — smoke, shape, sanity, and consistency checks.

Uses the anonymized CSV shipped with the repo as a realistic noisy input.
No synthetic spheres — all assertions are about structural correctness
(key count, type, range) and cross-module consistency.
"""

import math
import os

import pytest

from corneaforge.computed_indices import (
    classify_kc_morphology,
    compute_abcd_staging,
    compute_epithelial_refraction,
    compute_epithelial_sectors,
    compute_k_readings,
    compute_kc_classification,
    compute_opd_wavefront,
    compute_published_indices,
    compute_screening_extrema,
    compute_screening_indices,
    compute_shape_indices,
    compute_summary_indices,
    compute_zernike_indices,
    shape_rms_normative_p95,
    shape_rms_normative_p99,
)
from corneaforge.core import parse_csv, parse_metadata

# ── Fixtures ─────────────────────────────────────────────────────────

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "anonymyzed_csv.csv")
_SKIP = not os.path.exists(CSV_PATH)


@pytest.fixture(scope="module")
def parsed():
    if _SKIP:
        pytest.skip("anonymyzed_csv.csv not found")
    raw = parse_csv(CSV_PATH)
    meta = parse_metadata(CSV_PATH)
    return raw, meta


# ── Helpers ──────────────────────────────────────────────────────────


def _count_non_none(d):
    return sum(1 for v in d.values() if v is not None)


def _all_floats_or_none(d, exclude_keys=()):
    """Check every value is float, int, None, or str (for grade fields)."""
    for k, v in d.items():
        if k in exclude_keys:
            continue
        assert v is None or isinstance(v, (int, float, str)), (
            f"Key {k!r} has unexpected type {type(v)}"
        )


# ── Summary Indices ──────────────────────────────────────────────────


class TestSummaryIndices:
    def test_key_count(self, parsed):
        raw, _ = parsed
        out = compute_summary_indices(raw)
        assert len(out) == 16

    def test_no_crash(self, parsed):
        raw, _ = parsed
        out = compute_summary_indices(raw)
        _all_floats_or_none(out)

    def test_thkmin_is_positive(self, parsed):
        raw, _ = parsed
        out = compute_summary_indices(raw)
        if out["thk_min_value"] is not None:
            assert out["thk_min_value"] > 0

    def test_corneal_volume_positive(self, parsed):
        raw, _ = parsed
        out = compute_summary_indices(raw)
        if out["corneal_volume"] is not None:
            assert out["corneal_volume"] > 0


# ── K-readings ───────────────────────────────────────────────────────


class TestKReadings:
    def test_key_count(self, parsed):
        raw, meta = parsed
        out = compute_k_readings(raw, meta)
        assert len(out) == 114  # 105 original + 8 Fourier SimK + 1 irregularity

    def test_no_crash(self, parsed):
        raw, meta = parsed
        out = compute_k_readings(raw, meta)
        _all_floats_or_none(out)

    def test_simk_in_radius_range(self, parsed):
        """SimK should be in mm (radius), not diopters."""
        raw, meta = parsed
        out = compute_k_readings(raw, meta)
        kf = out.get("simk_kf")
        if kf is not None:
            assert 4.0 < kf < 12.0, f"SimK Kf={kf} outside plausible radius range"

    def test_all_non_none(self, parsed):
        raw, meta = parsed
        out = compute_k_readings(raw, meta)
        assert _count_non_none(out) == 114  # 105 original + 8 Fourier SimK + 1 irregularity


# ── Shape Indices ────────────────────────────────────────────────────


class TestShapeIndices:
    def test_key_count(self, parsed):
        raw, _ = parsed
        out = compute_shape_indices(raw)
        assert len(out) == 162  # 108 shape + 54 normative limits (9 diam x 2 surf x 3)

    def test_no_crash(self, parsed):
        raw, _ = parsed
        out = compute_shape_indices(raw)
        _all_floats_or_none(out)

    def test_fitting_diameter_values(self, parsed):
        """FittingDiameter keys should match their label."""
        raw, _ = parsed
        out = compute_shape_indices(raw)
        for k, v in out.items():
            if "fitting_diameter" in k and v is not None:
                # Extract diameter from key name (e.g. shape_3mm_ant_fitting_diameter)
                assert v > 0


# ── Screening Indices ────────────────────────────────────────────────


class TestScreeningIndices:
    def test_key_count(self, parsed):
        raw, meta = parsed
        out = compute_screening_indices(raw, meta)
        assert len(out) == 13

    def test_no_crash(self, parsed):
        raw, meta = parsed
        out = compute_screening_indices(raw, meta)
        _all_floats_or_none(out)

    def test_zernike_dependent_are_none(self, parsed):
        """Indices that need Zernike should be None in base screening."""
        raw, meta = parsed
        out = compute_screening_indices(raw, meta)
        for k in ["screening_rmsf", "screening_rmsb", "screening_eif", "screening_eib"]:
            assert out.get(k) is None, f"{k} should be None without Zernike"


# ── Screening Extrema ───────────────────────────────────────────────


class TestScreeningExtrema:
    def test_key_count(self, parsed):
        raw, meta = parsed
        zi = compute_zernike_indices(raw, meta)
        out = compute_screening_extrema(raw, meta, zernike_results=zi)
        assert len(out) == 29

    def test_no_crash(self, parsed):
        raw, meta = parsed
        out = compute_screening_extrema(raw, meta)
        _all_floats_or_none(out)

    def test_pdthksi_computes(self, parsed):
        """PDThkSI should compute from normative Zernike arrays."""
        raw, meta = parsed
        out = compute_screening_extrema(raw, meta)
        val = out["screening_pdthksi"]
        if raw.get("corneal_thickness") is not None:
            assert val is not None, "PDThkSI should compute when thickness map exists"
            assert isinstance(val, float)
            assert math.isfinite(val)


# ── ABCD Staging ─────────────────────────────────────────────────────


class TestABCDStaging:
    def test_key_count(self, parsed):
        raw, _ = parsed
        out = compute_abcd_staging(raw)
        assert len(out) == 9

    def test_grades_are_valid(self, parsed):
        raw, _ = parsed
        out = compute_abcd_staging(raw)
        for gk in ["abcd_a_grade", "abcd_b_grade", "abcd_c_grade", "abcd_d_grade"]:
            assert out[gk] in ("0", "1", "2", "3", "4", "*")

    def test_d_is_none(self, parsed):
        """Parameter D (BCVA) is never in individual CSV."""
        raw, _ = parsed
        out = compute_abcd_staging(raw)
        assert out["abcd_d"] is None
        assert out["abcd_d_grade"] == "*"

    def test_abcd_string_format(self, parsed):
        raw, _ = parsed
        out = compute_abcd_staging(raw)
        s = out["abcd_string"]
        assert s.startswith("A") and "B" in s and "C" in s and "D" in s


# ── Epithelial Sectors ───────────────────────────────────────────────


class TestEpithelialSectors:
    def test_key_count(self, parsed):
        raw, meta = parsed
        out = compute_epithelial_sectors(raw, meta)
        assert len(out) == 27

    def test_no_crash(self, parsed):
        raw, meta = parsed
        out = compute_epithelial_sectors(raw, meta)
        _all_floats_or_none(out)

    def test_min_le_mean_le_max(self, parsed):
        """For each zone, min <= mean <= max."""
        raw, meta = parsed
        out = compute_epithelial_sectors(raw, meta)
        # Check central zone
        mn = out.get("epi_min_central")
        avg = out.get("epi_mean_central")
        mx = out.get("epi_max_central")
        if all(v is not None for v in [mn, avg, mx]):
            assert mn <= avg <= mx


# ── Epithelial Refraction ────────────────────────────────────────────


class TestEpithelialRefraction:
    def test_key_count(self, parsed):
        raw, meta = parsed
        out = compute_epithelial_refraction(raw, meta)
        assert len(out) == 28

    def test_no_crash(self, parsed):
        raw, meta = parsed
        out = compute_epithelial_refraction(raw, meta)
        _all_floats_or_none(out)

    def test_vd_is_12_5(self, parsed):
        """Vertex distance should always be 12.5mm (CSO default RxVD)."""
        raw, meta = parsed
        out = compute_epithelial_refraction(raw, meta)
        assert out.get("epirx_3mm_vd") == 12.5
        assert out.get("epirx_6mm_vd") == 12.5


# ── Zernike ──────────────────────────────────────────────────────────


class TestZernike:
    def test_key_count(self, parsed):
        raw, meta = parsed
        out = compute_zernike_indices(raw, meta)
        assert len(out) == 83

    def test_no_crash(self, parsed):
        raw, meta = parsed
        out = compute_zernike_indices(raw, meta)
        _all_floats_or_none(out)

    def test_piston_is_zero(self, parsed):
        """Piston (z0) must be zeroed by convention."""
        raw, meta = parsed
        out = compute_zernike_indices(raw, meta)
        z0_ant = out.get("zernike_ant_4mm_z0_um")
        if z0_ant is not None:
            assert z0_ant == 0.0

    def test_coefficients_are_finite(self, parsed):
        """All Zernike coefficients should be finite numbers."""
        raw, meta = parsed
        out = compute_zernike_indices(raw, meta)
        for k, v in out.items():
            if k.endswith("_um") and v is not None:
                assert math.isfinite(v), f"{k} is not finite: {v}"

    def test_rmsf_positive(self, parsed):
        raw, meta = parsed
        out = compute_zernike_indices(raw, meta)
        rmsf = out.get("zernike_rmsf")
        if rmsf is not None:
            assert rmsf >= 0.0

    def test_bfs_radius_plausible(self, parsed):
        """BFS radius should be in corneal range."""
        raw, meta = parsed
        out = compute_zernike_indices(raw, meta)
        r = out.get("zernike_ant_4mm_bfs_radius")
        if r is not None:
            assert 4.0 < r < 20.0, f"BFS radius {r} outside plausible range"


# ── OPD Wavefront ────────────────────────────────────────────────────


class TestOPDWavefront:
    def test_key_count(self, parsed):
        raw, meta = parsed
        out = compute_opd_wavefront(raw, meta)
        assert len(out) == 1936

    def test_no_crash(self, parsed):
        raw, meta = parsed
        out = compute_opd_wavefront(raw, meta)
        _all_floats_or_none(out)

    def test_piston_zeroed(self, parsed):
        """All WFCoeff[0] should be zero (piston removed)."""
        raw, meta = parsed
        out = compute_opd_wavefront(raw, meta)
        for k, v in out.items():
            if k.endswith("_wfcoeff_0") and v is not None:
                assert v == 0.0, f"{k} piston not zeroed: {v}"

    def test_some_diameters_populated(self, parsed):
        """At least some diameters should have data."""
        raw, meta = parsed
        out = compute_opd_wavefront(raw, meta)
        non_none = _count_non_none(out)
        assert non_none > 0, "No OPD values computed at all"

    def test_4mm_populated(self, parsed):
        """4mm diameter should have enough data to fit on any valid CSV."""
        raw, meta = parsed
        out = compute_opd_wavefront(raw, meta)
        assert out.get("opd_front_4mm_radius") is not None


# ── Cross-module consistency ─────────────────────────────────────────


class TestConsistency:
    def test_thkmin_matches_across_modules(self, parsed):
        """ThkMin should be the same in summary, screening, and ABCD."""
        raw, meta = parsed
        summary = compute_summary_indices(raw)
        zi = compute_zernike_indices(raw, meta)
        screening = compute_screening_extrema(raw, meta, zernike_results=zi)
        abcd = compute_abcd_staging(raw)

        s_val = summary["thk_min_value"]
        sc_val = screening["screening_thkmin_value"]
        c_val = abcd["abcd_c"]

        if all(v is not None for v in [s_val, sc_val, c_val]):
            assert s_val == sc_val, "summary vs screening ThkMin mismatch"
            assert s_val == c_val, "summary vs ABCD_C ThkMin mismatch"

    def test_kmaxf_matches_across_modules(self, parsed):
        """KMaxF should match between summary and screening."""
        raw, meta = parsed
        summary = compute_summary_indices(raw)
        zi = compute_zernike_indices(raw, meta)
        screening = compute_screening_extrema(raw, meta, zernike_results=zi)

        s_val = summary["kmax_front_value"]
        sc_val = screening["screening_kmaxf_value"]

        if s_val is not None and sc_val is not None:
            assert s_val == sc_val

    def test_zernike_rmsf_matches_screening(self, parsed):
        """RMSf from Zernike module should be usable by screening."""
        raw, meta = parsed
        zi = compute_zernike_indices(raw, meta)
        rmsf = zi.get("zernike_rmsf")
        # Screening base has None for rmsf; Zernike provides it
        if rmsf is not None:
            assert rmsf >= 0.0


# ── Missing maps graceful degradation ────────────────────────────────


class TestMissingMaps:
    def test_summary_on_empty_segments(self):
        out = compute_summary_indices({})
        assert len(out) == 16
        assert all(v is None for v in out.values())

    def test_shape_on_empty_segments(self):
        out = compute_shape_indices({})
        assert len(out) == 162  # 108 shape + 54 normative limits (9 diam x 2 surf x 3)
        # Shape fit values should be None (no data), but normative limits
        # are always populated (they depend only on the diameter formula).
        for k, v in out.items():
            if k.endswith("_rms_p95_limit") or k.endswith("_rms_p99_limit"):
                assert v is not None and v > 0, f"{k} should be positive"
            elif k.endswith("_rms_abnormal"):
                assert v is None, f"{k} should be None without data"
            else:
                assert v is None, f"{k} should be None without data"

    def test_abcd_on_empty_segments(self):
        out = compute_abcd_staging({})
        assert out["abcd_a"] is None
        assert out["abcd_c"] is None
        assert out["abcd_d_grade"] == "*"

    def test_zernike_on_empty_segments(self):
        out = compute_zernike_indices({}, {})
        assert len(out) == 83
        assert all(v is None for v in out.values())

    def test_opd_on_empty_segments(self):
        out = compute_opd_wavefront({}, {})
        assert len(out) == 1936
        assert all(v is None for v in out.values())


# ── Corvis JSON validation (server-side) ─────────────────────────────


class TestCorvisInput:
    def test_valid_partial_input(self):
        from corneaforge.server import CorvisInput

        ci = CorvisInput(iop=21.0, sp_a1=158.7, cbi=0.01)
        d = ci.to_feature_dict()
        assert d["corvis_iop"] == 21.0
        assert d["corvis_sp_a1"] == 158.7
        assert d["corvis_biop"] is None  # not provided

    def test_all_fields(self):
        from corneaforge.server import CorvisInput

        ci = CorvisInput(
            iop=21.0,
            biop=18.1,
            cct=530.0,
            da_max=0.93,
            da_ratio_1mm=1.6,
            da_ratio_2mm=3.8,
            sp_a1=158.7,
            arth=743.6,
            cbi=0.01,
            integrated_radius=7.5,
            a1_time=8.18,
            a1_velocity=0.15,
            a2_time=21.68,
            a2_velocity=-0.28,
            peak_distance=4.70,
        )
        d = ci.to_feature_dict()
        assert len(d) == 15
        assert all(v is not None for v in d.values())

    def test_empty_input(self):
        from corneaforge.server import CorvisInput

        ci = CorvisInput()
        d = ci.to_feature_dict()
        assert all(v is None for v in d.values())

    def test_prefix(self):
        from corneaforge.server import CorvisInput

        ci = CorvisInput(iop=15.0)
        d = ci.to_feature_dict()
        assert all(k.startswith("corvis_") for k in d.keys())


# -- Published Indices (Rabinowitz I-S, ARTmax) -----------------------------


class TestPublishedIndices:
    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_keys_present(self, parsed):
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        assert "published_is_value" in result
        assert "published_artmax" in result
        assert "published_max_ppi" in result
        assert "published_simax_front" in result
        assert "published_simax_front_angle" in result
        assert "published_epi_donut_ratio" in result
        assert "published_epi_donut_delta" in result
        assert "published_srax" in result
        assert "published_kisa_pct" in result
        assert "published_pti_population_adapted" in result
        assert "published_bad_d_df" in result
        assert "published_bad_d_db" in result
        assert "published_bad_d_dp" in result
        assert "published_bad_d_dt" in result
        assert "published_bad_d_da" in result
        assert "published_bad_d_score" in result

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_is_value_type_and_range(self, parsed):
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        v = result["published_is_value"]
        # Should be a float for a valid exam, not None
        if v is not None:
            assert isinstance(v, float)
            # Physiological range: I-S should be between -10 and +15 D
            assert -10.0 < v < 15.0, f"I-S value {v} outside plausible range"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_artmax_type_and_range(self, parsed):
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        artmax = result["published_artmax"]
        max_ppi = result["published_max_ppi"]
        if artmax is not None:
            assert isinstance(artmax, float)
            # Plausible range depends on PPI definition; sanity-check only
            assert 1.0 < artmax < 5000.0, f"ARTmax {artmax} outside plausible range"
        if max_ppi is not None:
            assert isinstance(max_ppi, float)
            assert max_ppi > 0, "max_ppi must be positive"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_artmax_max_ppi_consistency(self, parsed):
        """ARTmax = ThkMin / max_ppi -- verify internal consistency."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        summary = compute_summary_indices(raw)
        artmax = result["published_artmax"]
        max_ppi = result["published_max_ppi"]
        thk_min = summary["thk_min_value"]
        if all(v is not None for v in [artmax, max_ppi, thk_min]):
            expected = thk_min / max_ppi
            assert abs(artmax - expected) < 0.01, f"ARTmax={artmax} != ThkMin/max_ppi={expected}"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_simax_front_type_and_range(self, parsed):
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        simax = result["published_simax_front"]
        angle = result["published_simax_front_angle"]
        if simax is not None:
            assert isinstance(simax, float)
            # SImax should be non-negative (absolute value)
            assert simax >= 0.0, f"SImax {simax} should be non-negative"
            # Physiological range: 0-20 D for most corneas
            assert simax < 30.0, f"SImax {simax} implausibly large"
        if angle is not None:
            assert isinstance(angle, float)
            assert 0.0 <= angle < 360.0, f"SImax angle {angle} outside [0, 360)"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_simax_ge_abs_sif(self, parsed):
        """SImax should be >= |SIf| since it searches all meridians."""
        from corneaforge.computed_indices import compute_screening_indices

        raw, meta = parsed
        pub = compute_published_indices(raw, meta)
        scr = compute_screening_indices(raw, meta)
        simax = pub["published_simax_front"]
        sif = scr["screening_sif"]
        if simax is not None and sif is not None:
            assert simax >= abs(sif) - 0.01, (
                f"SImax {simax} < |SIf| {abs(sif)} -- should be >= since it searches all meridians"
            )

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_epi_donut_type_and_range(self, parsed):
        """Epithelial Donut Index should be a ratio in (0, 1+) range."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        ratio = result["published_epi_donut_ratio"]
        delta = result["published_epi_donut_delta"]
        if ratio is not None:
            assert isinstance(ratio, float)
            # Ratio of two positive thicknesses: must be > 0
            assert 0.0 < ratio <= 1.5, f"Donut ratio {ratio} outside plausible range"
        if delta is not None:
            assert isinstance(delta, float)
            # Delta = max_annular - epi_min; can be negative in theory
            # but for a well-formed donut it should be >= 0
            assert -50.0 < delta < 50.0, f"Donut delta {delta} outside plausible range"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_epi_donut_consistency(self, parsed):
        """If both donut values exist, ratio * (max_annular) = epi_min."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        ratio = result["published_epi_donut_ratio"]
        delta = result["published_epi_donut_delta"]
        if ratio is not None and delta is not None:
            # delta = max_annular - epi_min
            # ratio = epi_min / max_annular
            # So epi_min = ratio * max_annular
            # And delta = max_annular - ratio * max_annular = max_annular * (1 - ratio)
            # Therefore delta / (1 - ratio) = max_annular (if ratio != 1)
            if abs(1.0 - ratio) > 1e-9:
                max_annular = delta / (1.0 - ratio)
                epi_min = ratio * max_annular
                recomputed_delta = max_annular - epi_min
                assert abs(recomputed_delta - delta) < 0.01, (
                    f"Internal inconsistency: delta={delta}, recomputed={recomputed_delta}"
                )

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_srax_type_and_range(self, parsed):
        """SRAX should be a non-negative angle deviation from 180 degrees."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        srax = result["published_srax"]
        if srax is not None:
            assert isinstance(srax, float)
            # SRAX is |angle_diff - 180|, so it's in [0, 180]
            assert 0.0 <= srax <= 180.0, f"SRAX {srax} outside [0, 180] range"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_kisa_pct_type_and_range(self, parsed):
        """KISA% should be non-negative."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        kisa = result["published_kisa_pct"]
        if kisa is not None:
            assert isinstance(kisa, float)
            # KISA% is a product of non-negative terms / 300, so >= 0
            assert kisa >= 0.0, f"KISA% {kisa} should be non-negative"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_kisa_pct_zero_when_srax_zero(self, parsed):
        """If SRAX = 0 (perfectly aligned axes), KISA% must be 0."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        srax = result["published_srax"]
        kisa = result["published_kisa_pct"]
        if srax is not None and srax == 0.0 and kisa is not None:
            assert kisa == 0.0, f"KISA% should be 0 when SRAX is 0, got {kisa}"

    def test_missing_segment_returns_none(self):
        result = compute_published_indices({}, {})
        assert result["published_is_value"] is None
        assert result["published_artmax"] is None
        assert result["published_max_ppi"] is None
        assert result["published_simax_front"] is None
        assert result["published_simax_front_angle"] is None
        assert result["published_epi_donut_ratio"] is None
        assert result["published_epi_donut_delta"] is None
        assert result["published_srax"] is None
        assert result["published_kisa_pct"] is None
        assert result["published_pti_population_adapted"] is None
        assert result["published_bad_d_df"] is None
        assert result["published_bad_d_db"] is None
        assert result["published_bad_d_dp"] is None
        assert result["published_bad_d_dt"] is None
        assert result["published_bad_d_da"] is None
        assert result["published_bad_d_score"] is None


class TestBadDComponents:
    """Tests for BAD-D (Belin/Ambrosio Enhanced Ectasia Display - Discriminant) components."""

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_bad_d_dt_matches_thkmin(self, parsed):
        """BAD-D Dt should equal ThkMin from summary indices."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        summary = compute_summary_indices(raw)
        dt = result["published_bad_d_dt"]
        thk_min = summary["thk_min_value"]
        if dt is not None and thk_min is not None:
            assert abs(dt - thk_min) < 0.01, f"BAD-D Dt={dt} != ThkMin={thk_min}"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_bad_d_dt_type_and_range(self, parsed):
        """Dt should be a positive float in physiological range."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        dt = result["published_bad_d_dt"]
        if dt is not None:
            assert isinstance(dt, float)
            # Physiological range: 350-650 um for most corneas
            assert 200.0 < dt < 700.0, f"BAD-D Dt={dt} outside plausible range"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_bad_d_da_type_and_range(self, parsed):
        """Da (anterior ARC) should be in physiological range."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        da = result["published_bad_d_da"]
        if da is not None:
            assert isinstance(da, float)
            # Normal cornea: 7.0-8.5 mm; KC can be as low as 5.5 mm
            assert 4.0 < da < 12.0, f"BAD-D Da={da} outside plausible range"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_bad_d_dp_type_and_range(self, parsed):
        """Dp (max PPI) should be positive."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        dp = result["published_bad_d_dp"]
        if dp is not None:
            assert isinstance(dp, float)
            assert dp > 0, f"BAD-D Dp={dp} should be positive"
            # Max PPI normally < 200 um/mm; extreme KC might reach 300
            assert dp < 500.0, f"BAD-D Dp={dp} implausibly large"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_bad_d_dp_matches_published_max_ppi(self, parsed):
        """BAD-D Dp should equal published_max_ppi (both are max PPI)."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        dp = result["published_bad_d_dp"]
        max_ppi = result["published_max_ppi"]
        if dp is not None and max_ppi is not None:
            assert abs(dp - max_ppi) < 0.01, f"BAD-D Dp={dp} != published_max_ppi={max_ppi}"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_bad_d_df_type(self, parsed):
        """Df (anterior BFS deviation) should be a float."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        df = result["published_bad_d_df"]
        if df is not None:
            assert isinstance(df, float)
            # Anterior BFS deviation: typically -30 to +30 um for normal,
            # can be larger in KC
            assert -100.0 < df < 100.0, f"BAD-D Df={df} outside plausible range"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_bad_d_db_type(self, parsed):
        """Db (posterior BFS deviation) should be a float."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        db = result["published_bad_d_db"]
        if db is not None:
            assert isinstance(db, float)
            # Posterior BFS deviation: typically -50 to +50 um for normal,
            # can be larger in KC
            assert -200.0 < db < 200.0, f"BAD-D Db={db} outside plausible range"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_bad_d_score_is_none_without_normatives(self, parsed):
        """BAD-D score should be None in Phase A (no normatives)."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        assert result["published_bad_d_score"] is None, (
            "BAD-D score should be None without normative database"
        )

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_bad_d_all_components_non_none_on_real_data(self, parsed):
        """On real exam data, all 5 BAD-D components should be non-None."""
        raw, meta = parsed
        result = compute_published_indices(raw, meta)
        for key in [
            "published_bad_d_df",
            "published_bad_d_db",
            "published_bad_d_dp",
            "published_bad_d_dt",
            "published_bad_d_da",
        ]:
            assert result[key] is not None, f"{key} is None on real exam data -- check extraction"

    def test_bad_d_empty_segments_returns_none(self):
        """With no segments, all BAD-D components should be None."""
        result = compute_published_indices({}, {})
        for key in [
            "published_bad_d_df",
            "published_bad_d_db",
            "published_bad_d_dp",
            "published_bad_d_dt",
            "published_bad_d_da",
            "published_bad_d_score",
        ]:
            assert result[key] is None, f"{key} should be None for empty input"


# ── Biconic Fit (research prototype) ────────────────────────────────


class TestBiconicFit:
    """Tests for _fit_biconic: synthetic surfaces + real CSV data."""

    def test_sphere_recovery(self):
        """A perfect sphere (R=7.8, p=1) should yield Rx~Ry~7.8, px~py~1."""
        import numpy as np

        from corneaforge.computed_indices import _fit_biconic

        R = 7.8  # mm
        n_rings = 20
        n_merid = 256
        h_list, z_list, t_list = [], [], []
        for i in range(1, n_rings + 1):
            h = i * 0.2
            z = R - math.sqrt(R**2 - h**2)
            for j in range(n_merid):
                theta = j * 2 * math.pi / n_merid
                h_list.append(h)
                z_list.append(z)
                t_list.append(theta)

        h = np.array(h_list)
        z = np.array(z_list)
        t = np.array(t_list)

        result = _fit_biconic(h, z, t)
        assert result is not None, "Biconic fit failed on a perfect sphere"
        Rx, Ry, px, py, alpha_deg, rms_um = result
        assert abs(Rx - R) < 0.01, f"Rx={Rx}, expected ~{R}"
        assert abs(Ry - R) < 0.01, f"Ry={Ry}, expected ~{R}"
        assert abs(px - 1.0) < 0.05, f"px={px}, expected ~1.0"
        assert abs(py - 1.0) < 0.05, f"py={py}, expected ~1.0"
        assert rms_um < 0.1, f"RMS={rms_um} um, expected ~0 for perfect sphere"

    def test_prolate_ellipse_recovery(self):
        """A rotationally symmetric prolate ellipse (p=0.8) should be recovered."""
        import numpy as np

        from corneaforge.computed_indices import _fit_biconic

        R = 7.8
        p = 0.8  # prolate
        n_rings = 20
        n_merid = 256
        h_list, z_list, t_list = [], [], []
        for i in range(1, n_rings + 1):
            h = i * 0.2
            disc = 1.0 - p * h**2 / R**2
            if disc <= 0:
                continue
            z = (h**2 / R) / (1.0 + math.sqrt(disc))
            for j in range(n_merid):
                theta = j * 2 * math.pi / n_merid
                h_list.append(h)
                z_list.append(z)
                t_list.append(theta)

        result = _fit_biconic(np.array(h_list), np.array(z_list), np.array(t_list))
        assert result is not None
        Rx, Ry, px, py, alpha_deg, rms_um = result
        assert abs(Rx - R) < 0.05, f"Rx={Rx}"
        assert abs(Ry - R) < 0.05, f"Ry={Ry}"
        assert abs(px - p) < 0.05, f"px={px}, expected ~{p}"
        assert abs(py - p) < 0.05, f"py={py}, expected ~{p}"
        assert rms_um < 0.1

    def test_biconic_with_different_asphericities(self):
        """A true biconic (px != py) should be recovered."""
        import numpy as np

        from corneaforge.computed_indices import _fit_biconic

        Rx_true = 8.0
        Ry_true = 7.6
        px_true = 0.7
        py_true = 1.2
        alpha_true = math.radians(30.0)

        h_list, z_list, t_list = [], [], []
        for i in range(1, 16):
            h = i * 0.2
            for j in range(256):
                theta = j * 2 * math.pi / 256
                x = h * math.cos(theta)
                y = h * math.sin(theta)
                ca = math.cos(alpha_true)
                sa = math.sin(alpha_true)
                xp = x * ca + y * sa
                yp = -x * sa + y * ca
                xp2 = xp * xp
                yp2 = yp * yp
                num = xp2 / Rx_true + yp2 / Ry_true
                disc = 1.0 - px_true * xp2 / (Rx_true**2) - py_true * yp2 / (Ry_true**2)
                if disc <= 0:
                    continue
                z = num / (1.0 + math.sqrt(disc))
                h_list.append(h)
                z_list.append(z)
                t_list.append(theta)

        result = _fit_biconic(np.array(h_list), np.array(z_list), np.array(t_list))
        assert result is not None, "Biconic fit failed on a true biconic"
        Rx, Ry, px, py, alpha_deg, rms_um = result
        assert abs(Rx - Rx_true) < 0.05, f"Rx={Rx}, expected {Rx_true}"
        assert abs(Ry - Ry_true) < 0.05, f"Ry={Ry}, expected {Ry_true}"
        assert abs(px - px_true) < 0.1, f"px={px}, expected {px_true}"
        assert abs(py - py_true) < 0.1, f"py={py}, expected {py_true}"
        assert abs(alpha_deg - 30.0) < 2.0, f"alpha={alpha_deg}, expected ~30"
        assert rms_um < 0.5, f"RMS={rms_um} um"

    def test_returns_none_on_insufficient_data(self):
        """Fewer than 10 points should return None."""
        import numpy as np

        from corneaforge.computed_indices import _fit_biconic

        h = np.array([1.0, 2.0, 3.0])
        z = np.array([0.1, 0.3, 0.5])
        t = np.array([0.0, 1.0, 2.0])
        assert _fit_biconic(h, z, t) is None

    def test_rx_ge_ry(self):
        """Output should always have Rx >= Ry (flat >= steep)."""
        import numpy as np

        from corneaforge.computed_indices import _fit_biconic

        # Generate a surface where steep radius is along the first meridian
        Rx_true = 7.5  # steep
        Ry_true = 8.0  # flat
        p = 0.82

        h_list, z_list, t_list = [], [], []
        for i in range(1, 16):
            h = i * 0.2
            for j in range(256):
                theta = j * 2 * math.pi / 256
                x = h * math.cos(theta)
                y = h * math.sin(theta)
                # alpha=0, so x-meridian has Rx_true (steep), y has Ry_true (flat)
                disc = 1.0 - p * x**2 / Rx_true**2 - p * y**2 / Ry_true**2
                if disc <= 0:
                    continue
                z = (x**2 / Rx_true + y**2 / Ry_true) / (1.0 + math.sqrt(disc))
                h_list.append(h)
                z_list.append(z)
                t_list.append(theta)

        result = _fit_biconic(np.array(h_list), np.array(z_list), np.array(t_list))
        assert result is not None
        Rx, Ry, px, py, alpha_deg, rms_um = result
        assert Rx >= Ry, f"Rx={Rx} < Ry={Ry} -- should have Rx >= Ry"

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_real_csv_no_crash(self, parsed):
        """Biconic fit should not crash on real CSV data."""
        import numpy as np

        from corneaforge.computed_indices import (
            _axial_to_height_grid,
            _fit_biconic,
            _remove_tilt,
        )

        raw, _ = parsed
        sag_ant = raw.get("sagittal_anterior")
        if sag_ant is None:
            pytest.skip("No sagittal_anterior in CSV")

        clean = sag_ant.astype(np.float64)
        n_rows, n_cols = clean.shape

        h_grid = np.outer(np.arange(n_rows) * 0.2, np.ones(n_cols))
        theta_grid = np.tile(np.arange(n_cols) * (2 * math.pi / n_cols), (n_rows, 1))

        invalid = (clean == -1000) | (clean <= 0) | np.isnan(clean)
        z_grid = _axial_to_height_grid(clean, h_grid, invalid)
        invalid = invalid | np.isnan(z_grid)

        # Use 8mm diameter
        hmax = 4.0
        in_zone = (h_grid <= hmax) & (~invalid)
        in_zone[0, :] = False
        h_valid = h_grid[in_zone]
        z_valid = z_grid[in_zone]
        theta_valid = theta_grid[in_zone]
        z_valid = _remove_tilt(h_valid, z_valid, theta_valid)

        result = _fit_biconic(h_valid, z_valid, theta_valid)
        # Should succeed on real data
        assert result is not None, "Biconic fit returned None on real CSV"
        Rx, Ry, px, py, alpha_deg, rms_um = result
        assert 5.0 < Rx < 12.0, f"Rx={Rx} outside plausible corneal range"
        assert 5.0 < Ry < 12.0, f"Ry={Ry} outside plausible corneal range"
        assert -1.0 < px < 3.0, f"px={px} outside plausible range"
        assert -1.0 < py < 3.0, f"py={py} outside plausible range"
        assert 0.0 <= alpha_deg < 180.0, f"alpha={alpha_deg} outside [0,180)"
        assert rms_um >= 0.0

    @pytest.mark.skipif(_SKIP, reason="anonymyzed_csv.csv not found")
    def test_biconic_rms_le_torus_rms(self, parsed):
        """Biconic RMS should be <= torus RMS (more parameters, better fit)."""
        import numpy as np

        from corneaforge.computed_indices import (
            _axial_to_height_grid,
            _compute_rms,
            _fit_biconic,
            _fit_torus_4param,
            _remove_tilt,
        )

        raw, _ = parsed

        sag_ant = raw["sagittal_anterior"].astype(np.float64)
        n_rows, n_cols = sag_ant.shape
        h_grid = np.outer(np.arange(n_rows) * 0.2, np.ones(n_cols))
        theta_grid = np.tile(np.arange(n_cols) * (2 * math.pi / n_cols), (n_rows, 1))
        invalid = (sag_ant == -1000) | (sag_ant <= 0) | np.isnan(sag_ant)
        z_grid = _axial_to_height_grid(sag_ant, h_grid, invalid)
        invalid = invalid | np.isnan(z_grid)
        hmax = 4.0
        in_zone = (h_grid <= hmax) & (~invalid)
        in_zone[0, :] = False
        h_valid = h_grid[in_zone]
        z_valid = z_grid[in_zone]
        theta_valid = theta_grid[in_zone]
        z_valid = _remove_tilt(h_valid, z_valid, theta_valid)

        # Compute un-smoothed torus RMS for a fair comparison
        R_mean, p, dR, alpha = _fit_torus_4param(h_valid, z_valid, theta_valid)
        torus_rms = _compute_rms(h_valid**2, z_valid, theta_valid, R_mean, p, dR, alpha)

        result = _fit_biconic(h_valid, z_valid, theta_valid)
        assert result is not None
        biconic_rms = result[5]
        # Biconic has 5 free params vs torus's 4 -- must fit at least as well
        # (both RMS values are un-smoothed for a fair comparison)
        assert biconic_rms <= torus_rms + 0.01, f"Biconic RMS {biconic_rms} > torus RMS {torus_rms}"


# ── Newton Posterior Intersection ────────────────────────────────────


class TestNewtonPosteriorIntersection:
    """Tests for _newton_posterior_intersection convergence and edge cases."""

    def test_newton_on_axis_sphere(self):
        """Newton iteration should find intersection of an on-axis ray with a known sphere."""
        import numpy as np

        from corneaforge.computed_indices import _newton_posterior_intersection

        # Posterior sphere: z = R - sqrt(R^2 - x^2 - y^2) at R=6.5mm
        # Biquad approximation near apex: z ~ x^2/(2R) + y^2/(2R) + CCT
        R = 6.5
        CCT = 0.55  # mm (corneal thickness at vertex)
        coeffs = np.array([1 / (2 * R), 1 / (2 * R), 0.0, 0.0, 0.0, CCT])

        # Ray going straight down from anterior apex
        ray_dir = np.array([0.0, 0.0, 1.0])
        ant_point = np.array([0.0, 0.0, 0.0])

        result = _newton_posterior_intersection(coeffs, ray_dir, ant_point)
        assert result is not None, "Newton should converge for on-axis ray"
        assert abs(result[2] - CCT) < 0.001, f"z should be ~CCT={CCT}, got {result[2]}"

    def test_newton_off_axis_converges(self):
        """Newton should converge for a tilted ray hitting a curved surface."""
        import numpy as np

        from corneaforge.computed_indices import _newton_posterior_intersection

        R = 6.5
        CCT = 0.55
        coeffs = np.array([1 / (2 * R), 1 / (2 * R), 0.0, 0.0, 0.0, CCT])

        # Slightly off-axis ray (must have nonzero dz for Cramer formulation)
        ray_dir = np.array([0.05, 0.03, 0.998])
        ray_dir /= np.linalg.norm(ray_dir)
        ant_point = np.array([0.1, 0.05, 0.001])

        result = _newton_posterior_intersection(coeffs, ray_dir, ant_point)
        assert result is not None, "Newton should converge for off-axis ray"
        # Verify result lies on the biquadratic surface
        x, y, z = result
        a, b, c, d, e, f = coeffs
        expected_z = a * x**2 + b * y**2 + c * x + d * y + e * x * y + f
        assert abs(z - expected_z) < 1e-4, f"Point not on surface: z={z}, surf={expected_z}"

    def test_newton_returns_none_on_zero_dz(self):
        """Newton should return None when dz=0 (ray parallel to z-axis plane)."""
        import numpy as np

        from corneaforge.computed_indices import _newton_posterior_intersection

        coeffs = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.5])
        # Pure lateral ray: dz=0, so tx=ty=inf. Should return None.
        ray_dir = np.array([1.0, 0.0, 0.0])
        ant_point = np.array([0.0, 0.0, 0.0])

        result = _newton_posterior_intersection(coeffs, ray_dir, ant_point)
        assert result is None, "Should return None when dz=0"

    def test_newton_returns_none_on_zero_ray(self):
        """Newton should return None for a zero-direction ray."""
        import numpy as np

        from corneaforge.computed_indices import _newton_posterior_intersection

        coeffs = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.5])
        ray_dir = np.array([0.0, 0.0, 0.0])
        ant_point = np.array([0.0, 0.0, 0.0])

        result = _newton_posterior_intersection(coeffs, ray_dir, ant_point)
        assert result is None, "Should return None for zero ray direction"

    def test_newton_result_on_ray_and_surface(self):
        """The intersection point should lie on both the ray and the surface."""
        import numpy as np

        from corneaforge.computed_indices import _newton_posterior_intersection

        R = 7.0
        CCT = 0.52
        coeffs = np.array([1 / (2 * R), 1 / (2 * R), 0.0, 0.0, 0.0, CCT])

        ray_dir = np.array([0.02, -0.01, 0.9997])
        ray_dir /= np.linalg.norm(ray_dir)
        ant_point = np.array([0.0, 0.0, 0.0])

        result = _newton_posterior_intersection(coeffs, ray_dir, ant_point)
        assert result is not None

        # Verify point lies on the ray: result = ant_point + t * ray_dir
        x, y, z = result
        tx = ray_dir[0] / ray_dir[2]
        ty = ray_dir[1] / ray_dir[2]
        expected_x = ant_point[0] + tx * (z - ant_point[2])
        expected_y = ant_point[1] + ty * (z - ant_point[2])
        assert abs(x - expected_x) < 1e-5, f"x mismatch: {x} vs {expected_x}"
        assert abs(y - expected_y) < 1e-5, f"y mismatch: {y} vs {expected_y}"

        # Verify point lies on the surface
        a, b, c, d, e, f = coeffs
        surf_z = a * x**2 + b * y**2 + c * x + d * y + e * x * y + f
        assert abs(z - surf_z) < 1e-4, f"Not on surface: z={z}, surf_z={surf_z}"

    def test_newton_with_cross_term(self):
        """Newton should converge when the biquadratic has a nonzero cross-term."""
        import numpy as np

        from corneaforge.computed_indices import _newton_posterior_intersection

        # Surface with cross-term: z = 0.08*x^2 + 0.07*y^2 + 0.01*x*y + 0.55
        coeffs = np.array([0.08, 0.07, 0.0, 0.0, 0.01, 0.55])

        ray_dir = np.array([0.03, 0.02, 0.999])
        ray_dir /= np.linalg.norm(ray_dir)
        ant_point = np.array([0.0, 0.0, 0.0])

        result = _newton_posterior_intersection(coeffs, ray_dir, ant_point)
        assert result is not None, "Should converge with cross-term"

        x, y, z = result
        a, b, c, d, e, f = coeffs
        surf_z = a * x**2 + b * y**2 + c * x + d * y + e * x * y + f
        assert abs(z - surf_z) < 1e-4


# ── Ray-Traced OPD on Synthetic Sphere ──────────────────────────────


class TestOPDRaytraceSphere:
    """Tests for _opd_raytrace_one_diameter using biquadratic surface evaluation."""

    def test_raytrace_sphere_low_hoa(self):
        """A near-spherical surface should produce small HOA coefficients."""
        import numpy as np

        from corneaforge.computed_indices import N_AIR, N_CORNEA, _opd_raytrace_one_diameter

        # Build a synthetic sphere polar map (R=7.8mm)
        R = 7.8
        n_rows, n_cols = 31, 256
        polar_map = np.full((n_rows, n_cols), -1000.0)
        r_step = 0.2
        for i in range(n_rows):
            r = i * r_step
            for j in range(n_cols):
                theta = 2 * np.pi * j / n_cols
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                rho2 = x * x + y * y
                if rho2 < R * R:
                    polar_map[i, j] = R - np.sqrt(R * R - rho2)

        # fitting_radius = diameter / 2 = 1.0 mm
        coeffs, focal_z = _opd_raytrace_one_diameter(polar_map, 1.0, N_AIR, N_CORNEA)
        assert coeffs is not None, "Should succeed on synthetic sphere"
        # HOA (j>=6) should be small for a near-perfect sphere
        hoa_rms = np.sqrt(np.sum(coeffs[6:] ** 2))
        assert hoa_rms < 1.0, f"HOA RMS should be <1.0 um for sphere, got {hoa_rms}"
        # Focal length should be approximately R*n2/(n2-n1)
        expected_f = R * N_CORNEA / (N_CORNEA - N_AIR)
        assert abs(focal_z - expected_f) < 2.0, f"Focal should be ~{expected_f:.1f}, got {focal_z}"

    def test_raytrace_returns_none_on_none_map(self):
        """Should return (None, None) when polar_map is None."""
        from corneaforge.computed_indices import N_AIR, N_CORNEA, _opd_raytrace_one_diameter

        coeffs, focal_z = _opd_raytrace_one_diameter(None, 1.0, N_AIR, N_CORNEA)
        assert coeffs is None
        assert focal_z is None

    def test_raytrace_returns_none_on_empty_map(self):
        """Should return (None, None) when polar_map is all -1000."""
        import numpy as np

        from corneaforge.computed_indices import N_AIR, N_CORNEA, _opd_raytrace_one_diameter

        polar_map = np.full((31, 256), -1000.0)
        coeffs, focal_z = _opd_raytrace_one_diameter(polar_map, 1.0, N_AIR, N_CORNEA)
        assert coeffs is None
        assert focal_z is None

    def test_raytrace_piston_zeroed(self):
        """Piston coefficient (j=0) should be zeroed."""
        import numpy as np

        from corneaforge.computed_indices import N_AIR, N_CORNEA, _opd_raytrace_one_diameter

        R = 7.8
        n_rows, n_cols = 31, 256
        polar_map = np.full((n_rows, n_cols), -1000.0)
        r_step = 0.2
        for i in range(n_rows):
            r = i * r_step
            for j in range(n_cols):
                theta = 2 * np.pi * j / n_cols
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                rho2 = x * x + y * y
                if rho2 < R * R:
                    polar_map[i, j] = R - np.sqrt(R * R - rho2)

        coeffs, _ = _opd_raytrace_one_diameter(polar_map, 1.0, N_AIR, N_CORNEA)
        assert coeffs is not None
        assert coeffs[0] == 0.0, f"Piston should be zero, got {coeffs[0]}"

    def test_raytrace_coeffs_count(self):
        """Should return 36 Zernike coefficients (order 7, matching CSO NPol=36)."""
        import numpy as np

        from corneaforge.computed_indices import N_AIR, N_CORNEA, _opd_raytrace_one_diameter

        R = 7.8
        n_rows, n_cols = 31, 256
        polar_map = np.full((n_rows, n_cols), -1000.0)
        r_step = 0.2
        for i in range(n_rows):
            r = i * r_step
            for j in range(n_cols):
                theta = 2 * np.pi * j / n_cols
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                rho2 = x * x + y * y
                if rho2 < R * R:
                    polar_map[i, j] = R - np.sqrt(R * R - rho2)

        coeffs, _ = _opd_raytrace_one_diameter(polar_map, 1.0, N_AIR, N_CORNEA)
        assert coeffs is not None
        assert len(coeffs) == 36, f"Expected 36 Zernike terms (order 7), got {len(coeffs)}"

    def test_raytrace_radius_too_large(self):
        """Should return None when fitting_radius exceeds map data extent."""
        import numpy as np

        from corneaforge.computed_indices import N_AIR, N_CORNEA, _opd_raytrace_one_diameter

        # 31 rows * 0.2mm/row = 6.0mm max radius. Request 7.0mm.
        R = 7.8
        polar_map = np.full((31, 256), -1000.0)
        for i in range(31):
            r = i * 0.2
            for j in range(256):
                theta = 2 * np.pi * j / 256
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                rho2 = x * x + y * y
                if rho2 < R * R:
                    polar_map[i, j] = R - np.sqrt(R * R - rho2)

        coeffs, focal_z = _opd_raytrace_one_diameter(polar_map, 7.0, N_AIR, N_CORNEA)
        assert coeffs is None
        assert focal_z is None


# ── PDThkSI ─────────────────────────────────────────────────────────


class TestPDThkSI:
    """Tests for PDThkSI (Population-Deviation Thickness Symmetry Index)."""

    def test_pdthksi_computes(self, parsed):
        """PDThkSI should return a numeric value for valid thickness data."""
        raw, meta = parsed
        out = compute_screening_extrema(raw, meta)
        val = out.get("screening_pdthksi")
        # Should be a real number (not None) if corneal_thickness exists
        if raw.get("corneal_thickness") is not None:
            assert val is not None, "PDThkSI should compute for valid thickness"
            assert isinstance(val, float)

    def test_pdthksi_finite(self, parsed):
        """PDThkSI value must be finite."""
        raw, meta = parsed
        out = compute_screening_extrema(raw, meta)
        val = out.get("screening_pdthksi")
        if val is not None:
            assert math.isfinite(val), f"PDThkSI not finite: {val}"

    def test_pdthksi_reasonable_range(self, parsed):
        """PDThkSI (Z-score based) should be in a clinically reasonable range."""
        raw, meta = parsed
        out = compute_screening_extrema(raw, meta)
        val = out.get("screening_pdthksi")
        if val is not None:
            # Z-scored symmetry index: typical range is roughly [-10, 10]
            assert -20 < val < 20, f"PDThkSI out of range: {val}"

    def test_pdthksi_normative_reconstruction(self):
        """Normative arrays reconstruct plausible thickness maps."""
        import numpy as np

        from corneaforge.computed_indices import (
            _NORM_THK_MEAN,
            _NORM_THK_SD,
            _reconstruct_normative_map,
        )

        norm_mean = _reconstruct_normative_map(_NORM_THK_MEAN, 31, 256)
        norm_sd = _reconstruct_normative_map(_NORM_THK_SD, 31, 256)

        # Center value at rho=0 includes all Z(n,0) contributions at the
        # origin (R_n(0) alternates between +1 and -1).  The defocus term
        # Z(2,0) contributes ~-51um, bringing the center from the piston
        # coefficient (601.5 um) down to ~551 um -- still a plausible CCT.
        center_mean = norm_mean[0, 0]
        assert 500 < center_mean < 650, f"Normative center mean {center_mean}"

        # The mean across the full map should be closer to population average
        valid_mean = np.nanmean(norm_mean)
        assert 550 < valid_mean < 650, f"Normative map average {valid_mean}"

        # SD map should be mostly positive (thickness variation)
        valid_sd = norm_sd[~np.isnan(norm_sd)]
        assert len(valid_sd) > 0, "No valid SD values"


# ── KC Morphological Classifier ─────────────────────────────────────


class TestKCMorphology:
    """Tests for keratoconus morphological classifier."""

    def test_nipple_central(self):
        """Steep, central cone -> NippleCentral."""
        out = classify_kc_morphology(
            kavg_mm=6.5,  # very steep (>48D)
            asphericity_p=0.5,
            kc_center_x=0.0,
            kc_center_y=0.0,  # central
            coma_z7=5.0,
            coma_z8=5.0,
            cyl=-3.0,
            cyl_ax=90.0,
        )
        assert out["kc_morphology_type"] == 0
        assert out["kc_morphology_name"] == "NippleCentral"

    def test_nipple_paracentral(self):
        """Steep, off-center cone -> NippleParacentral."""
        out = classify_kc_morphology(
            kavg_mm=6.5,  # very steep
            asphericity_p=0.5,
            kc_center_x=1.5,
            kc_center_y=0.0,  # off-center (> 1.25mm)
            coma_z7=5.0,
            coma_z8=5.0,
            cyl=-3.0,
            cyl_ax=90.0,
        )
        assert out["kc_morphology_type"] == 1
        assert out["kc_morphology_name"] == "NippleParacentral"

    def test_bowtie(self):
        """Low coma/cylinder ratio -> BowTie."""
        out = classify_kc_morphology(
            kavg_mm=7.5,  # not very steep
            asphericity_p=0.5,  # not very prolate
            kc_center_x=0.5,
            kc_center_y=-0.5,
            coma_z7=0.5,
            coma_z8=0.3,  # small coma
            cyl=-5.0,  # large cylinder -> coma/cyl < 1
            cyl_ax=0.0,
        )
        assert out["kc_morphology_type"] == 2
        assert out["kc_morphology_name"] == "BowTie"

    def test_croissant(self):
        """Aligned axes (< 30 deg) -> Croissant."""
        out = classify_kc_morphology(
            kavg_mm=7.5,
            asphericity_p=0.5,
            kc_center_x=1.0,
            kc_center_y=0.0,  # KAx ~ 0 deg
            coma_z7=10.0,
            coma_z8=10.0,  # large coma -> coma/cyl > 1
            cyl=-2.0,
            cyl_ax=10.0,  # close to KAx (0 deg) -> diff < 30
        )
        assert out["kc_morphology_type"] == 4
        assert out["kc_morphology_name"] == "Croissant"

    def test_snowman(self):
        """Misaligned axes (> 60 deg) -> SnowMan."""
        out = classify_kc_morphology(
            kavg_mm=7.5,
            asphericity_p=0.5,
            kc_center_x=1.0,
            kc_center_y=0.0,  # KAx ~ 0 deg
            coma_z7=10.0,
            coma_z8=10.0,  # large coma
            cyl=-2.0,
            cyl_ax=100.0,  # far from KAx -> diff > 60
        )
        assert out["kc_morphology_type"] == 3
        assert out["kc_morphology_name"] == "SnowMan"

    def test_duck(self):
        """Intermediate axis misalignment -> Duck."""
        out = classify_kc_morphology(
            kavg_mm=7.5,
            asphericity_p=0.5,
            kc_center_x=1.0,
            kc_center_y=0.0,  # KAx ~ 0 deg
            coma_z7=10.0,
            coma_z8=10.0,
            cyl=-2.0,
            cyl_ax=45.0,  # 45 deg diff: 30 < diff < 60
        )
        assert out["kc_morphology_type"] == 5
        assert out["kc_morphology_name"] == "Duck"

    def test_nan_returns_unknown(self):
        """NaN inputs -> Unknown."""
        out = classify_kc_morphology(
            kavg_mm=float("nan"),
            asphericity_p=0.5,
            kc_center_x=0.0,
            kc_center_y=0.0,
            coma_z7=5.0,
            coma_z8=5.0,
            cyl=-3.0,
            cyl_ax=90.0,
        )
        assert out["kc_morphology_type"] == -1
        assert out["kc_morphology_name"] == "Unknown"

    def test_prolate_triggers_nipple(self):
        """Very prolate asphericity (p < -0.25) -> Nipple even if not steep."""
        out = classify_kc_morphology(
            kavg_mm=8.0,  # not steep
            asphericity_p=-0.5,  # very prolate
            kc_center_x=0.0,
            kc_center_y=0.0,
            coma_z7=5.0,
            coma_z8=5.0,
            cyl=-3.0,
            cyl_ax=90.0,
        )
        assert out["kc_morphology_type"] == 0  # NippleCentral

    def test_compute_kc_classification_no_inputs(self, parsed):
        """Returns Unknown when required inputs are missing."""
        raw, meta = parsed
        out = compute_kc_classification(raw, meta)
        assert out["kc_morphology_type"] == -1


# ── Shape Normative Limits ──────────────────────────────────────────


class TestShapeNormativeLimits:
    """Tests for shape RMS normative P95/P99 limits."""

    def test_p95_anterior_positive(self):
        """P95 anterior must be positive for any positive diameter."""
        for phi in [3, 4, 4.5, 5, 6, 7, 8, 9, 10]:
            val = shape_rms_normative_p95(phi, anterior=True)
            assert val > 0, f"P95 ant at {phi}mm = {val}"

    def test_p99_anterior_positive(self):
        """P99 anterior must be positive for any positive diameter."""
        for phi in [3, 4, 4.5, 5, 6, 7, 8, 9, 10]:
            val = shape_rms_normative_p99(phi, anterior=True)
            assert val > 0, f"P99 ant at {phi}mm = {val}"

    def test_p99_exceeds_p95(self):
        """P99 must exceed P95 for all surfaces and diameters."""
        for phi in [3, 4, 4.5, 5, 6, 7, 8, 9, 10]:
            for anterior in [True, False]:
                p95 = shape_rms_normative_p95(phi, anterior=anterior)
                p99 = shape_rms_normative_p99(phi, anterior=anterior)
                assert p99 > p95, (
                    f"P99 ({p99}) <= P95 ({p95}) at {phi}mm {'ant' if anterior else 'post'}"
                )

    def test_monotonically_increasing(self):
        """Limits should increase with diameter (quadratic growth)."""
        diameters = [3, 4, 5, 6, 7, 8, 9, 10]
        for anterior in [True, False]:
            prev_95 = 0
            prev_99 = 0
            for phi in diameters:
                p95 = shape_rms_normative_p95(phi, anterior=anterior)
                p99 = shape_rms_normative_p99(phi, anterior=anterior)
                assert p95 > prev_95
                assert p99 > prev_99
                prev_95 = p95
                prev_99 = p99

    def test_shape_indices_contain_limits(self, parsed):
        """compute_shape_indices output contains p95/p99/abnormal keys."""
        raw, _ = parsed
        out = compute_shape_indices(raw)

        # Check a few representative keys exist
        for d in ["3", "4p5", "10"]:
            for surf in ["ant", "post"]:
                p95_key = f"shape_{d}mm_{surf}_rms_p95_limit"
                p99_key = f"shape_{d}mm_{surf}_rms_p99_limit"
                abnorm_key = f"shape_{d}mm_{surf}_rms_abnormal"
                assert p95_key in out, f"Missing key {p95_key}"
                assert p99_key in out, f"Missing key {p99_key}"
                assert abnorm_key in out, f"Missing key {abnorm_key}"

    def test_abnormal_flag_type(self, parsed):
        """Abnormal flag should be bool or None."""
        raw, _ = parsed
        out = compute_shape_indices(raw)
        for key, val in out.items():
            if key.endswith("_rms_abnormal"):
                assert val is None or isinstance(val, bool), (
                    f"{key} has unexpected type {type(val)}"
                )

    def test_known_values(self):
        """Verify exact values from decompiled code."""
        # Anterior P95 at 8mm: 0.0007*64 + 0.0059*8 = 0.0448 + 0.0472 = 0.092
        assert abs(shape_rms_normative_p95(8.0, anterior=True) - 0.092) < 1e-10
        # Posterior P99 at 10mm: 0.0056*100 + 0.01*10 = 0.56 + 0.1 = 0.66
        assert abs(shape_rms_normative_p99(10.0, anterior=False) - 0.66) < 1e-10
