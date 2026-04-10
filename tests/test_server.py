"""
Tests for the FastAPI prediction server.

Uses FastAPI's TestClient (backed by httpx) to simulate HTTP requests
without starting a real server process.

Skipped automatically when FastAPI is not installed (e.g. CI running
pip install ".[dev]" without the server extra).
"""

import io

import pytest

try:
    from fastapi.testclient import TestClient

    from corneaforge.server import app

    client = TestClient(app)
except ImportError:
    pytestmark = pytest.mark.skip(reason="FastAPI not installed (install with .[server])")


# ── Helpers ──────────────────────────────────────────────────────────


def _make_csv(eye="OD", n_segments=13):
    """
    Build a minimal valid MS-39 CSV in memory.

    Creates a CSV with metadata + n_segments segments of fake data.
    Each segment has 26 data rows of 256 columns (valid polar data).
    """
    SEGMENT_HEADERS = [
        "SagittalAnterior [mm]",
        "TangentialAnterior [mm]",
        "GaussianAnterior [mm]",
        "SagittalPosterior [mm]",
        "TangentialPosterior [mm]",
        "GaussianPosterior [mm]",
        "RefractiveFrontalPowerAnterior [D]",
        "RefractiveFrontalPowerPosterior [D]",
        "RefractiveEquivalentPower [D]",
        "CornealThickness [um]",
        "StromalThickness [um]",
        "EpitelialThickness [um]",
        "AnteriorChamberDepth [mm]",
    ]

    lines = [
        "Patient_ID;P001",
        "Patient_Last_Name;TEST",
        "Patient_First_Name;PATIENT",
        f"Exam_Eye;{eye}",
    ]

    data_row = ";".join(["7.80"] * 256)

    for i in range(min(n_segments, len(SEGMENT_HEADERS))):
        lines.append(SEGMENT_HEADERS[i])
        for _ in range(26):
            lines.append(data_row)
        # 6 padding rows of -1000
        padding_row = ";".join(["-1000"] * 256)
        for _ in range(6):
            lines.append(padding_row)

    return "\n".join(lines).encode("utf-8")


def _upload(ms39_individual=None, corvis_json=None):
    """Helper to POST files to /predict/disease_classification."""
    files = {}
    data = {}
    if ms39_individual is not None:
        files["ms39_individual"] = ("test.csv", io.BytesIO(ms39_individual), "text/csv")
    if corvis_json is not None:
        data["corvis"] = corvis_json
    return client.post("/predict/disease_classification", files=files, data=data)


# ── Health ───────────────────────────────────────────────────────────


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── Predict: valid inputs ────────────────────────────────────────────


def test_predict_individual_only():
    """Upload only ms39_individual — should work."""
    r = _upload(ms39_individual=_make_csv("OD"))
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "no_model"
    assert data["eye"] == "OD"
    assert "prediction" in data
    assert isinstance(data["warnings"], list)


def test_predict_with_corvis_json():
    """Upload ms39_individual + Corvis JSON — should work."""
    corvis = '{"iop": 21.0, "sp_a1": 158.7, "cbi": 0.01}'
    r = _upload(ms39_individual=_make_csv("OD"), corvis_json=corvis)
    assert r.status_code == 200
    data = r.json()
    assert data["eye"] == "OD"
    assert data["status"] == "no_model"


def test_predict_os_eye():
    """Upload an OS file — eye should be OS."""
    r = _upload(ms39_individual=_make_csv("OS"))
    assert r.status_code == 200
    assert r.json()["eye"] == "OS"


def test_predict_returns_no_model_status():
    """Without an ONNX model, status should be no_model."""
    r = _upload(ms39_individual=_make_csv("OD"))
    data = r.json()
    assert data["status"] == "no_model"
    assert data["prediction"] is None


# ── Predict: invalid inputs ──────────────────────────────────────────


def test_predict_no_files():
    """POST with no files — should fail (ms39_individual is required)."""
    r = client.post("/predict/disease_classification")
    assert r.status_code == 422


def test_predict_empty_file():
    """Upload an empty ms39_individual — should return 400."""
    r = _upload(ms39_individual=b"")
    assert r.status_code == 400


def test_predict_garbage_file():
    """Upload binary garbage — should return error status."""
    r = _upload(ms39_individual=b"\x00\x01\x02\xff\xfe")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "error"
    assert data["prediction"] is None
    assert len(data["warnings"]) > 0


def test_predict_no_segments():
    """Upload a CSV with metadata but no segments — error status."""
    csv = b"Patient_ID;P001\nExam_Eye;OD\n"
    r = _upload(ms39_individual=csv)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "error"
    assert data["prediction"] is None
    assert any("No data segments" in w for w in data["warnings"])


# ── Starlette patch ──────────────────────────────────────────────────


def test_starlette_form_patch():
    """Verify the Request.form() monkey-patch is active (unlimited file uploads)."""
    import inspect

    from starlette.requests import Request

    sig = inspect.signature(Request.form)
    max_files_default = sig.parameters["max_files"].default
    assert max_files_default == float("inf"), (
        f"Request.form() max_files default is {max_files_default}, expected inf. "
        "The monkey-patch in server.py may have broken after a Starlette upgrade."
    )
