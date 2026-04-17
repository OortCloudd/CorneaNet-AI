"""
CorneaForge Server
===================

Accepts an MS-39 individual CSV and optional Corvis ST values for disease prediction.

Inputs (per eye):
  - ms39_individual: MS-39 per-ring polar data (16 segments, 256 meridians) — file upload
  - corvis: Corvis ST biomechanical values — JSON form (typed by clinician)

Flow:
  1. Upload MS-39 individual CSV + optional Corvis JSON
  2. Parse + validate + compute indices (summary, K-readings, shape, screening,
     ABCD, epithelial, Zernike, OPD wavefront — 2372 computed features)
  3. Extract features → feed ONNX model
  4. Return: eye, prediction, warnings, maps
"""

import asyncio
import io
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Prevent OpenBLAS from spawning many threads per NumPy call.
# Parallelism is handled at a higher level (ThreadPoolExecutor intra-patient,
# SLURM inter-patient) so each BLAS call should use a single core.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from corneaforge.computed_indices import (
    compute_abcd_staging,
    compute_epithelial_refraction,
    compute_epithelial_sectors,
    compute_k_readings,
    compute_opd_wavefront,
    compute_screening_extrema,
    compute_screening_indices,
    compute_shape_indices,
    compute_summary_indices,
    compute_zernike_indices,
)
from corneaforge.core import parse_csv, parse_metadata
from corneaforge.corvis_parser import CorvisParseResult, check_ollama_available, parse_corvis_pdf
from corneaforge.descriptive_stats import process_segments as stats_process
from corneaforge.experimental_indices import compute_conoid_analysis, compute_conoid_opd
from corneaforge.nn_pipeline import process_segments as nn_process
from corneaforge.validate import validate_parsed
from corneaforge.visual_pipeline import process_segments as visual_process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# An MS-39 CSV is ~30KB. Corvis ST is similar. 5MB is generous enough
# for any legitimate file while stopping accidental large uploads from
# eating server RAM.
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Directory where prediction maps are saved, one subfolder per prediction ID.
# Old predictions are cleaned up when new ones arrive.
MAPS_DIR = os.path.join(tempfile.gettempdir(), "corneaforge_maps")
_MAX_MAP_DIRS = 10  # Keep maps for the last 10 predictions

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Number of parallel workers for /research batch jobs.
# Respects SLURM cgroup allocation, Docker --cpus, or uses all cores.
_N_WORKERS = len(os.sched_getaffinity(0))

logger = logging.getLogger("corneaforge")

app = FastAPI(
    title="CorneaForge",
    version="0.1.0",
    description="Corneal topography disease prediction",
)


# Starlette 1.0 defaults to max 1000 files per multipart upload.
# Researchers batch-process thousands. Patch both form() and _get_form()
# to default to unlimited — form() is what FastAPI actually calls.
from starlette._utils import (  # noqa: E402
    AwaitableOrContextManager,
    AwaitableOrContextManagerWrapper,
)
from starlette.requests import Request  # noqa: E402

_original_get_form = Request._get_form


def _unlimited_form(
    self, *, max_files=float("inf"), max_fields=float("inf"), max_part_size=1024 * 1024
) -> AwaitableOrContextManager:
    return AwaitableOrContextManagerWrapper(
        _original_get_form(
            self, max_files=max_files, max_fields=max_fields, max_part_size=max_part_size
        )
    )


Request.form = _unlimited_form


# Serve generated maps as static files: GET /maps/sagittal_anterior.png
os.makedirs(MAPS_DIR, exist_ok=True)
app.mount("/maps", StaticFiles(directory=MAPS_DIR), name="maps")


# ── Health check ─────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Parsing per file type ────────────────────────────────────────────


def _parse_ms39_individual(contents: bytes, filename: str) -> dict:
    """Parse and validate an MS-39 individual (per-ring polar) export."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        metadata = parse_metadata(tmp_path)
        raw_segments = parse_csv(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        return {"eye": "OD", "error": str(e), "warnings": [f"Cannot read file: {e}"]}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    validation = validate_parsed(metadata, raw_segments)

    return {
        "metadata": metadata,
        "raw_segments": raw_segments,
        "validation": validation,
        "eye": validation.eye,
    }


# ── Feature extraction ───────────────────────────────────────────────


def _extract_features(parsed: dict, corvis_input=None) -> dict:
    """
    Extract features from parsed MS-39 data and optional Corvis manual input.

    Independent compute functions run in parallel via ThreadPoolExecutor
    (intra-patient parallelism).  OPENBLAS_NUM_THREADS=1 ensures each
    thread uses one core without contention.

    Returns a dict with named keys:
      features["ms39_individual_stats"] → tabular features (descriptive stats)
      features["ms39_individual_tensor"] → (13, 224, 224) ndarray
      features["computed_indices"] → 16 summary indices recomputed from polar maps
      features["corvis"] → Corvis ST scalar values from manual input
    """
    features = {}

    # MS-39 individual
    ms39 = parsed.get("ms39_individual")
    if ms39 and "raw_segments" in ms39:
        raw = ms39["raw_segments"]
        meta = ms39["metadata"]
        eye = ms39.get("eye", "?")

        t0 = time.monotonic()
        logger.info("Processing %s exam...", eye)

        all_indices: dict = {}
        tensor = None

        with ThreadPoolExecutor() as pool:
            # --- Group 1: all independent tasks in parallel ---------------
            independent = {
                "stats_process": pool.submit(stats_process, raw, meta),
                "nn_process": pool.submit(nn_process, raw),
                "compute_summary_indices": pool.submit(compute_summary_indices, raw),
                "compute_k_readings": pool.submit(compute_k_readings, raw, meta),
                "compute_shape_indices": pool.submit(compute_shape_indices, raw),
                "compute_screening_indices": pool.submit(compute_screening_indices, raw, meta),
                "compute_abcd_staging": pool.submit(compute_abcd_staging, raw),
                "compute_epithelial_sectors": pool.submit(compute_epithelial_sectors, raw, meta),
                "compute_zernike_indices": pool.submit(compute_zernike_indices, raw, meta),
                "compute_epithelial_refraction": pool.submit(
                    compute_epithelial_refraction, raw, meta
                ),
                "compute_opd_wavefront": pool.submit(compute_opd_wavefront, raw, meta),
            }

            # Collect results as they complete
            zernike: dict = {}
            for name, fut in independent.items():
                try:
                    result = fut.result()
                except Exception as e:
                    logger.warning("%s failed: %s", name, e, exc_info=True)
                    continue

                if name == "stats_process":
                    features["ms39_individual_stats"] = _clean_stats(result)
                elif name == "nn_process":
                    tensor = result[0]  # (tensor, segments)
                elif name == "compute_zernike_indices":
                    zernike = result
                    all_indices.update(result)
                else:
                    all_indices.update(result)

            # --- Group 2: screening_extrema depends on zernike ------------
            try:
                result = pool.submit(
                    compute_screening_extrema, raw, meta, zernike_results=zernike
                ).result()
                all_indices.update(result)
            except Exception as e:
                logger.warning("compute_screening_extrema failed: %s", e, exc_info=True)

        features["computed_indices"] = _clean_stats(all_indices)
        features["ms39_individual_tensor"] = tensor

        elapsed = time.monotonic() - t0
        logger.info("Computed %d indices in %.2fs", len(all_indices), elapsed)

    # Corvis ST — manual input from clinician
    if corvis_input is not None:
        features["corvis"] = corvis_input.to_feature_dict()

    return features


def _generate_maps(parsed: dict) -> tuple[str, list[str]]:
    """
    Generate colormap PNGs in a unique subfolder of MAPS_DIR.

    Returns (prediction_id, list of segment names).
    The frontend loads them from GET /maps/{prediction_id}/{segment_name}.png
    """
    ms39 = parsed.get("ms39_individual")
    if not ms39 or "raw_segments" not in ms39:
        return "", []

    # Each prediction gets its own folder — no race condition
    pred_id = str(uuid.uuid4())[:8]
    pred_dir = os.path.join(MAPS_DIR, pred_id)

    # visual_process saves PNGs into a patient_name subfolder
    visual_process(ms39["raw_segments"], pred_dir, patient_name="maps")

    # Move PNGs up one level for clean URLs: /maps/{pred_id}/{segment}.png
    subfolder = os.path.join(pred_dir, "maps")
    maps = []
    if os.path.isdir(subfolder):
        for png in sorted(os.listdir(subfolder)):
            if png.endswith(".png"):
                shutil.move(os.path.join(subfolder, png), os.path.join(pred_dir, png))
                maps.append(png.replace(".png", ""))
        os.rmdir(subfolder)

    # Clean up old prediction folders, keep the most recent _MAX_MAP_DIRS
    _cleanup_map_dirs()

    return pred_id, maps


def _cleanup_map_dirs():
    """Keep only the _MAX_MAP_DIRS most recent prediction map folders."""
    if not os.path.isdir(MAPS_DIR):
        return
    dirs = []
    for d in os.listdir(MAPS_DIR):
        path = os.path.join(MAPS_DIR, d)
        if os.path.isdir(path):
            dirs.append((path, os.path.getmtime(path)))
    dirs.sort(key=lambda x: x[1])
    for path, _ in dirs[:-_MAX_MAP_DIRS]:
        shutil.rmtree(path)


def _predict(features: dict) -> str | None:
    """
    Run ONNX model inference.

    Placeholder until the .onnx model file is provided.
    When ready:
      import onnxruntime as ort
      session = ort.InferenceSession("model.onnx")
      result = session.run(None, build_model_input(features))
      return result
    """
    return None


def _clean_stats(stats: dict) -> dict:
    """Make stats JSON-serializable (numpy types → Python types)."""
    cleaned = {}
    for k, v in stats.items():
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        if isinstance(v, float) and np.isnan(v):
            v = None
        cleaned[k] = v
    return cleaned


# ── Upload endpoint ──────────────────────────────────────────────────


class CorvisInput(BaseModel):
    """Corvis ST values — from PDF OCR or typed by the clinician."""

    iop: float | None = Field(None, description="IOP (mmHg)")
    biop: float | None = Field(None, description="bIOP — biomechanically corrected IOP (mmHg)")
    cct: float | None = Field(None, description="CCT — central corneal thickness (um)")
    da_max: float | None = Field(None, description="DA max — maximum deformation amplitude (mm)")
    da_ratio_1mm: float | None = Field(None, description="DA Ratio at 1mm")
    da_ratio_2mm: float | None = Field(None, description="DA Ratio at 2mm")
    sp_a1: float | None = Field(None, description="SP-A1 — Stiffness Parameter at A1")
    arth: float | None = Field(None, description="ARTh — Ambrosio Relational Thickness horizontal")
    cbi: float | None = Field(None, description="CBI — Corvis Biomechanical Index")
    integrated_radius: float | None = Field(None, description="Integrated Radius (mm)")
    a1_time: float | None = Field(None, description="A1 Time (ms)")
    a1_velocity: float | None = Field(None, description="A1 Velocity (m/s)")
    a2_time: float | None = Field(None, description="A2 Time (ms)")
    a2_velocity: float | None = Field(None, description="A2 Velocity (m/s)")
    peak_distance: float | None = Field(None, description="Peak Distance (mm)")
    ssi: float | None = Field(None, description="SSI — Stress-Strain Index")
    inverse_concave_radius: float | None = Field(None, description="Inverse Concave Radius (mm⁻¹)")
    a1_length: float | None = Field(None, description="A1 Applanation Length (mm)")
    a2_length: float | None = Field(None, description="A2 Applanation Length (mm)")

    def to_feature_dict(self) -> dict:
        """Convert to a flat dict with 'corvis_' prefix for feature concatenation."""
        out = {}
        for field_name, value in self:
            out[f"corvis_{field_name}"] = value
        return out


@app.post("/predict/disease_classification")
async def predict(
    ms39_individual: UploadFile = File(..., description="MS-39 individual (per-ring polar) CSV"),
    corvis: str = Form(None, description="Corvis ST values as JSON string"),
):
    """
    Upload an MS-39 individual CSV for one eye and optionally provide
    Corvis ST values (from the printed report) as a JSON string.

    Maps are generated and served at /maps/{prediction_id}/{segment}.png
    """

    # Read and validate the upload
    contents = await ms39_individual.read()
    if not contents:
        raise HTTPException(status_code=400, detail="ms39_individual file is empty")
    if len(contents) > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE // 1024 // 1024
        raise HTTPException(status_code=413, detail=f"ms39_individual exceeds {max_mb}MB limit")

    # Parse Corvis JSON if provided
    corvis_input = None
    if corvis:
        try:
            corvis_input = CorvisInput.model_validate_json(corvis)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid Corvis JSON: {e}")

    def _process():
        warnings = []

        # Parse MS-39 individual
        parsed = {}
        parsed["ms39_individual"] = _parse_ms39_individual(contents, ms39_individual.filename)

        if "error" in parsed["ms39_individual"]:
            return {
                "status": "error",
                "eye": parsed["ms39_individual"]["eye"],
                "prediction": None,
                "warnings": parsed["ms39_individual"]["warnings"],
            }

        if not parsed["ms39_individual"]["validation"].valid:
            return {
                "status": "error",
                "eye": parsed["ms39_individual"]["eye"],
                "prediction": None,
                "warnings": parsed["ms39_individual"]["validation"].errors,
            }

        warnings.extend(parsed["ms39_individual"]["validation"].warnings)

        eye = parsed["ms39_individual"]["eye"]

        # Extract features for model
        features = _extract_features(parsed, corvis_input)

        # Generate maps for the frontend
        pred_id, maps = _generate_maps(parsed)

        # Predict
        prediction = _predict(features)

        logger.info("eye=%s prediction=%s maps=%d", eye, prediction, len(maps))

        return {
            "status": "no_model" if prediction is None else "ok",
            "eye": eye,
            "prediction": prediction,
            "warnings": warnings,
            "prediction_id": pred_id,
            "maps": maps,
        }

    result = await asyncio.to_thread(_process)
    return result


# ── Corvis ST PDF parsing ───────────────────────────────────────────


@app.post("/corvis/parse")
async def corvis_parse(
    corvis_pdf: UploadFile = File(..., description="Corvis ST PDF report"),
):
    """Parse a Corvis ST PDF and return extracted values for clinician review.

    The clinician uploads the PDF, reviews/corrects the OCR'd values in
    the UI, then submits them with the MS-39 CSV via /predict.

    Requires Ollama running locally with the VLM model pulled.
    """
    contents = await corvis_pdf.read()
    if not contents:
        raise HTTPException(status_code=400, detail="corvis_pdf file is empty")
    fname = corvis_pdf.filename or ""
    if not fname.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF (.pdf extension)")
    if len(contents) > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE // 1024 // 1024
        raise HTTPException(status_code=413, detail=f"corvis_pdf exceeds {max_mb}MB limit")

    if not check_ollama_available():
        raise HTTPException(
            status_code=503,
            detail="VLM backend not available. Check Ollama or vLLM/SGLang server.",
        )

    def _do_parse():
        return parse_corvis_pdf(contents)

    parsed: CorvisParseResult = await asyncio.to_thread(_do_parse)

    logger.info(
        "Corvis OCR: %d/16 values, %d/3 pages, %d warnings, %d errors",
        len(parsed.values),
        parsed.pages_parsed,
        len(parsed.warnings),
        len(parsed.errors),
    )
    if parsed.errors:
        logger.warning("Corvis OCR errors: %s", parsed.errors)

    return {
        "values": parsed.values,
        "warnings": parsed.warnings,
        "errors": parsed.errors,
        "pages_parsed": parsed.pages_parsed,
    }


@app.get("/corvis/status")
def corvis_status():
    """Check if Corvis PDF parsing is available (VLM backend)."""
    available = check_ollama_available()
    return {"available": available}


# ── Research endpoints ────────────────────────────────────────────────
#
# Flow: POST /research/start/{type} → returns job_id
#       GET  /research/progress/{job_id} → returns {done, total, status, eta}
#       GET  /research/download/{job_id} → returns the result file
#
# Processing runs in a background thread so the frontend can poll progress.

_jobs: dict[str, dict] = {}
_JOB_TTL = 600  # Completed jobs expire after 10 minutes


def _cleanup_jobs():
    """Remove completed jobs older than _JOB_TTL. Running jobs are never evicted."""
    now = time.time()
    stale = [
        jid
        for jid, job in _jobs.items()
        if job["status"] == "done" and now - job.get("start_time", now) > _JOB_TTL
    ]
    for jid in stale:
        del _jobs[jid]


def _parse_one_upload(contents: bytes, filename: str) -> dict | None:
    """Parse a single CSV upload. Returns parsed dict or None on failure."""
    if not contents or len(contents) > MAX_FILE_SIZE:
        return None

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        metadata = parse_metadata(tmp_path)
        raw_segments = parse_csv(tmp_path)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    name = os.path.splitext(filename or "unknown")[0]
    return {"name": name, "metadata": metadata, "raw_segments": raw_segments}


# ---------------------------------------------------------------------------
# Per-patient worker functions (top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def _worker_stats(contents: bytes, filename: str) -> dict | None:
    """Process one patient into a feature-row dict. Runs in a worker process."""
    parsed = _parse_one_upload(contents, filename)
    if not parsed:
        return None

    raw = parsed["raw_segments"]
    meta = parsed["metadata"]
    row = stats_process(raw, meta, parsed["name"])
    zernike: dict = {}

    for name, func, args in [
        ("compute_summary_indices", compute_summary_indices, (raw,)),
        ("compute_k_readings", compute_k_readings, (raw, meta)),
        ("compute_shape_indices", compute_shape_indices, (raw,)),
        ("compute_screening_indices", compute_screening_indices, (raw, meta)),
        ("compute_abcd_staging", compute_abcd_staging, (raw,)),
        ("compute_epithelial_sectors", compute_epithelial_sectors, (raw, meta)),
    ]:
        try:
            row.update(func(*args))
        except Exception as e:
            logger.warning("[batch] %s failed for %s: %s", name, parsed["name"], e)

    try:
        zernike = compute_zernike_indices(raw, meta)
        row.update(zernike)
    except Exception as e:
        logger.warning("[batch] compute_zernike_indices failed for %s: %s", parsed["name"], e)

    try:
        row.update(compute_screening_extrema(raw, meta, zernike_results=zernike))
    except Exception as e:
        logger.warning("[batch] compute_screening_extrema failed for %s: %s", parsed["name"], e)

    try:
        row.update(compute_epithelial_refraction(raw, meta))
    except Exception as e:
        logger.warning("[batch] compute_epithelial_refraction failed for %s: %s", parsed["name"], e)

    try:
        row.update(compute_opd_wavefront(raw, meta))
    except Exception as e:
        logger.warning("[batch] compute_opd_wavefront failed for %s: %s", parsed["name"], e)

    # --- Experimental indices (ML features, not clinical) ---
    try:
        conoid = compute_conoid_analysis(raw, meta)
        row.update(
            {
                k: v
                for k, v in conoid.items()
                if not k.startswith("_")
                and k != "conoid_ant_quadric_coeffs"
                and k != "conoid_post_quadric_coeffs"
            }
        )
        row.update(compute_conoid_opd(raw, meta, conoid))
    except Exception as e:
        logger.warning("[batch] experimental_indices failed for %s: %s", parsed["name"], e)

    return _clean_stats(row)


def _worker_tensors(contents: bytes, filename: str) -> tuple[str, bytes] | None:
    """Process one patient into an npz buffer. Runs in a worker process."""
    parsed = _parse_one_upload(contents, filename)
    if not parsed:
        return None
    tensor, segments = nn_process(parsed["raw_segments"])
    if tensor is None:
        return None
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, data=tensor, segments=np.array(segments))
    return parsed["name"], npz_buf.getvalue()


def _worker_maps(contents: bytes, filename: str) -> tuple[str, list[tuple[str, bytes]]] | None:
    """Process one patient into a list of (png_name, png_bytes). Runs in a worker process."""
    parsed = _parse_one_upload(contents, filename)
    if not parsed:
        return None
    with tempfile.TemporaryDirectory() as vis_dir:
        visual_process(parsed["raw_segments"], vis_dir, patient_name=parsed["name"])
        patient_folder = os.path.join(vis_dir, parsed["name"])
        if not os.path.isdir(patient_folder):
            return None
        pngs = []
        for png in sorted(os.listdir(patient_folder)):
            if png.endswith(".png"):
                png_path = os.path.join(patient_folder, png)
                with open(png_path, "rb") as f:
                    pngs.append((png, f.read()))
        return parsed["name"], pngs


# ---------------------------------------------------------------------------
# Job runners (parallel via ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _run_job_stats(job_id: str, file_data: list[tuple[bytes, str]]):
    """Process files into a stats CSV using parallel workers."""
    job = _jobs[job_id]
    job["total"] = len(file_data)
    job["start_time"] = time.time()

    rows = []
    with ProcessPoolExecutor(max_workers=_N_WORKERS) as pool:
        futures = {
            pool.submit(_worker_stats, contents, filename): i
            for i, (contents, filename) in enumerate(file_data)
        }
        file_data.clear()  # Free bytes — workers have copies

        for fut in futures:
            row = fut.result()
            if row is not None:
                rows.append(row)
            job["done"] = job.get("done", 0) + 1
            elapsed = time.time() - job["start_time"]
            if job["done"] > 0:
                job["eta"] = round(elapsed / job["done"] * (job["total"] - job["done"]))

    if rows:
        buf = io.BytesIO()
        pd.DataFrame(rows).to_csv(buf, index=False, sep=";", float_format="%.6f", quoting=1)
        buf.seek(0)
        job["result"] = buf
        job["filename"] = "descriptive_stats.csv"
        job["media_type"] = "text/csv"

    job["status"] = "done"


def _run_job_tensors(job_id: str, file_data: list[tuple[bytes, str]]):
    """Process files into a tensors zip using parallel workers."""
    job = _jobs[job_id]
    job["total"] = len(file_data)
    job["start_time"] = time.time()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        with ProcessPoolExecutor(max_workers=_N_WORKERS) as pool:
            futures = {
                pool.submit(_worker_tensors, contents, filename): i
                for i, (contents, filename) in enumerate(file_data)
            }
            file_data.clear()

            for fut in futures:
                result = fut.result()
                if result is not None:
                    name, npz_bytes = result
                    zf.writestr(f"{name}.npz", npz_bytes)
                job["done"] = job.get("done", 0) + 1
                elapsed = time.time() - job["start_time"]
                if job["done"] > 0:
                    job["eta"] = round(elapsed / job["done"] * (job["total"] - job["done"]))

    buf.seek(0)
    job["result"] = buf
    job["filename"] = "tensors.zip"
    job["media_type"] = "application/zip"
    job["status"] = "done"


def _run_job_maps(job_id: str, file_data: list[tuple[bytes, str]]):
    """Process files into a maps zip using parallel workers."""
    job = _jobs[job_id]
    job["total"] = len(file_data)
    job["start_time"] = time.time()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        with ProcessPoolExecutor(max_workers=_N_WORKERS) as pool:
            futures = {
                pool.submit(_worker_maps, contents, filename): i
                for i, (contents, filename) in enumerate(file_data)
            }
            file_data.clear()

            for fut in futures:
                result = fut.result()
                if result is not None:
                    patient_name, pngs = result
                    for png_name, png_bytes in pngs:
                        zf.writestr(f"{patient_name}/{png_name}", png_bytes)
                job["done"] = job.get("done", 0) + 1
                elapsed = time.time() - job["start_time"]
                if job["done"] > 0:
                    job["eta"] = round(elapsed / job["done"] * (job["total"] - job["done"]))

    buf.seek(0)
    job["result"] = buf
    job["filename"] = "maps.zip"
    job["media_type"] = "application/zip"
    job["status"] = "done"


_JOB_RUNNERS = {
    "stats": _run_job_stats,
    "tensors": _run_job_tensors,
    "maps": _run_job_maps,
}


@app.post("/research/start/{job_type}")
async def research_start(
    job_type: str,
    files: list[UploadFile] = File(..., description="MS-39 CSV files"),
):
    """Upload CSVs and start processing. Returns a job_id to poll progress."""
    if job_type not in _JOB_RUNNERS:
        raise HTTPException(status_code=400, detail=f"Unknown job type: {job_type}")

    # Read all files into memory so the background thread can access them
    file_data = []
    for f in files:
        contents = await f.read()
        if contents:
            file_data.append((contents, f.filename))

    if not file_data:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    _cleanup_jobs()

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "done": 0, "total": 0, "eta": 0}

    runner = _JOB_RUNNERS[job_type]
    threading.Thread(target=runner, args=(job_id, file_data), daemon=True).start()

    return {"job_id": job_id}


@app.get("/research/progress/{job_id}")
def research_progress(job_id: str):
    """Poll processing progress."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": job["status"],
        "done": job["done"],
        "total": job["total"],
        "eta": job.get("eta", 0),
    }


@app.get("/research/download/{job_id}")
def research_download(job_id: str):
    """Download the result of a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not finished yet")
    if "result" not in job:
        raise HTTPException(status_code=400, detail="No valid files were processed")

    result = job["result"]
    result.seek(0)

    # Clean up the job after download
    filename = job.get("filename", "result")
    media_type = job.get("media_type", "application/octet-stream")
    del _jobs[job_id]

    return StreamingResponse(
        result,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── UI ───────────────────────────────────────────────────────────────
# Serve the frontend. Must be mounted LAST — StaticFiles with html=True
# is a catch-all that serves index.html for any unmatched route.

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


# ── Entry point ──────────────────────────────────────────────────────


def main():
    """Console script entry point: corneaforge-server"""
    import uvicorn

    uvicorn.run("corneaforge.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
