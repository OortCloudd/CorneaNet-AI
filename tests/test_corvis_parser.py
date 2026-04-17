"""
Tests for the Corvis ST PDF parser.

Unit tests run without Ollama (mock the VLM). Integration tests require
Ollama + qwen2.5vl:7b running locally and are skipped otherwise.
"""

import os

import pytest

from corneaforge.corvis_parser import (
    _KEY_TO_CORVIS_FIELD,
    _VALUE_RANGES,
    CorvisParseResult,
    _parse_json_response,
    check_ollama_available,
    parse_corvis_pdf,
)

# ── Unit tests (no Ollama needed) ───────────────────────────────────


class TestParseJsonResponse:
    """Test JSON extraction from VLM responses."""

    def test_plain_json(self):
        assert _parse_json_response('{"a": 1}') == {"a": 1}

    def test_json_in_code_fence(self):
        text = '```json\n{"IOP_mmHg": 15.0, "CCT_um": 563}\n```'
        assert _parse_json_response(text) == {"IOP_mmHg": 15.0, "CCT_um": 563}

    def test_json_in_code_fence_no_lang(self):
        text = '```\n{"a": 1}\n```'
        assert _parse_json_response(text) == {"a": 1}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"a": 1.5} as requested.'
        assert _parse_json_response(text) == {"a": 1.5}

    def test_no_json(self):
        assert _parse_json_response("no json here") is None

    def test_invalid_json(self):
        assert _parse_json_response("{invalid: json}") is None

    def test_negative_values(self):
        text = '{"A2_Velocity_ms": -0.25}'
        assert _parse_json_response(text) == {"A2_Velocity_ms": -0.25}


class TestKeyMapping:
    """Verify key mapping covers all expected fields."""

    def test_all_vlm_keys_have_corvis_mapping(self):
        expected_vlm_keys = {
            "IOP_mmHg",
            "CCT_um",
            "A1_Velocity_ms",
            "A2_Velocity_ms",
            "Peak_Distance_mm",
            "Deformation_Amplitude_mm",
            "A1_Length_mm",
            "A2_Length_mm",
            "Radius_HC_mm",
            "SP_A1",
            "ARTh",
            "bIOP_mmHg",
            "DA_Ratio",
            "Inverse_Concave_Radius_mm",
            "CBI",
            "SSI",
        }
        assert expected_vlm_keys == set(_KEY_TO_CORVIS_FIELD.keys())

    def test_all_mapped_fields_have_ranges(self):
        for corvis_field in _KEY_TO_CORVIS_FIELD.values():
            assert corvis_field in _VALUE_RANGES, f"Missing range for {corvis_field}"


class TestValueRanges:
    """Sanity check the clinical value ranges."""

    def test_ranges_are_valid(self):
        for field, (lo, hi) in _VALUE_RANGES.items():
            assert lo < hi, f"Invalid range for {field}: [{lo}, {hi}]"

    def test_a2_velocity_is_negative(self):
        lo, hi = _VALUE_RANGES["a2_velocity"]
        assert lo < 0 and hi <= 0


class TestCorvisParseResult:
    """Test the result dataclass."""

    def test_defaults(self):
        r = CorvisParseResult()
        assert r.values == {}
        assert r.warnings == []
        assert r.errors == []
        assert r.raw_responses == {}


class TestParsePdfEdgeCases:
    """Test parse_corvis_pdf input validation."""

    def test_empty_bytes(self):
        result = parse_corvis_pdf(b"")
        assert result.values == {}
        assert any("Empty" in e for e in result.errors)

    def test_not_a_pdf(self):
        result = parse_corvis_pdf(b"This is not a PDF file at all")
        assert result.values == {}
        assert any("magic bytes" in e for e in result.errors)

    def test_png_bytes(self):
        result = parse_corvis_pdf(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert result.values == {}
        assert any("magic bytes" in e for e in result.errors)

    def test_truncated_pdf(self):
        """A valid PDF header but truncated content should fail at render, not crash."""
        result = parse_corvis_pdf(b"%PDF-1.4 truncated")
        assert result.values == {}
        assert len(result.errors) > 0


# ── Integration tests (require Ollama) ──────────────────────────────

CORVIS_PDF = os.path.join(
    os.path.dirname(__file__), "..", "..", "CorneaForge_handoff", "corvis_test.pdf"
)

_ollama_available = check_ollama_available()
needs_ollama = pytest.mark.skipif(not _ollama_available, reason="Ollama not available")
needs_pdf = pytest.mark.skipif(not os.path.exists(CORVIS_PDF), reason="Test PDF not found")


@needs_ollama
@needs_pdf
class TestIntegrationParsePdf:
    """End-to-end test with real PDF and VLM."""

    # Ground truth for the test PDF
    GROUND_TRUTH = {
        "iop": 15.0,
        "cct": 563,
        "a1_velocity": 0.13,
        "a2_velocity": -0.25,
        "peak_distance": 5.01,
        "da_max": 0.96,
        "a2_length": 2.19,
        "integrated_radius": 7.36,
        "sp_a1": 119.4,
        "arth": 563.1,
        "biop": 14.7,
        "da_ratio_2mm": 3.99,
        "inverse_concave_radius": 0.14,
        "cbi": 0.17,
        "ssi": 1.05,
    }

    @pytest.fixture(scope="class")
    def parse_result(self):
        with open(CORVIS_PDF, "rb") as f:
            pdf_bytes = f.read()
        return parse_corvis_pdf(pdf_bytes)

    def test_no_errors(self, parse_result):
        assert parse_result.errors == []

    def test_extracts_all_values(self, parse_result):
        assert len(parse_result.values) >= 15

    def test_values_within_tolerance(self, parse_result):
        """Most values should be within 10% of ground truth."""
        correct = 0
        total = 0
        for key, gt_val in self.GROUND_TRUTH.items():
            val = parse_result.values.get(key)
            if val is not None:
                total += 1
                tol = max(abs(gt_val) * 0.10, 0.05)
                if abs(val - gt_val) < tol:
                    correct += 1
        # At least 80% accuracy (12/15)
        assert correct >= total * 0.8, f"Only {correct}/{total} values within tolerance"

    def test_critical_values_exact(self, parse_result):
        """IOP, CCT, CBI, SP-A1 must be exact or very close."""
        critical = {"iop": 0.5, "cct": 1, "cbi": 0.02, "sp_a1": 1.0}
        for key, tol in critical.items():
            val = parse_result.values.get(key)
            gt = self.GROUND_TRUTH[key]
            assert val is not None, f"{key} not extracted"
            assert abs(val - gt) <= tol, f"{key}={val}, expected {gt} (tol={tol})"


# ── Server endpoint tests ───────────────────────────────────────────

try:
    from fastapi.testclient import TestClient

    from corneaforge.server import app

    _has_fastapi = True
except ImportError:
    _has_fastapi = False

needs_fastapi = pytest.mark.skipif(not _has_fastapi, reason="FastAPI not installed")


@needs_fastapi
class TestCorvisEndpoints:
    """Test the /corvis/* server endpoints."""

    def test_corvis_status(self):
        client = TestClient(app)
        resp = client.get("/corvis/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data

    def test_corvis_parse_empty_file(self):
        client = TestClient(app)
        resp = client.post(
            "/corvis/parse", files={"corvis_pdf": ("test.pdf", b"", "application/pdf")}
        )
        assert resp.status_code == 400

    def test_corvis_parse_wrong_extension(self):
        client = TestClient(app)
        resp = client.post(
            "/corvis/parse",
            files={"corvis_pdf": ("image.png", b"%PDF-fake", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    @needs_ollama
    @needs_pdf
    def test_corvis_parse_real_pdf(self):
        client = TestClient(app)
        with open(CORVIS_PDF, "rb") as f:
            pdf_bytes = f.read()
        resp = client.post(
            "/corvis/parse",
            files={"corvis_pdf": ("corvis.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "values" in data
        assert len(data["values"]) >= 10


@needs_fastapi
class TestCorvisInputNewFields:
    """Test that CorvisInput accepts the new fields."""

    def test_new_fields_accepted(self):
        from corneaforge.server import CorvisInput

        ci = CorvisInput(
            iop=15.0,
            ssi=1.05,
            inverse_concave_radius=0.14,
            a1_length=2.01,
            a2_length=2.19,
        )
        assert ci.ssi == 1.05
        assert ci.inverse_concave_radius == 0.14
        assert ci.a1_length == 2.01
        assert ci.a2_length == 2.19

    def test_new_fields_in_feature_dict(self):
        from corneaforge.server import CorvisInput

        ci = CorvisInput(ssi=1.05, a1_length=2.01)
        d = ci.to_feature_dict()
        assert d["corvis_ssi"] == 1.05
        assert d["corvis_a1_length"] == 2.01
