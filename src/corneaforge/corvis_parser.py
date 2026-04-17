"""
Corvis ST PDF Parser
====================

Extracts biomechanical values from Corvis ST PDF reports using a Vision
Language Model (Qwen2.5-VL via Ollama).

Corvis ST exports are rasterized screenshots in PDF — no vector text layer.
Traditional OCR (Tesseract) achieves ~25% accuracy on these. Qwen2.5-VL 7B
achieves 100% accuracy (16/16 values) on a 6-PDF benchmark with structured
JSON prompting (~18s per PDF on L40S).

Usage:
    from corneaforge.corvis_parser import parse_corvis_pdf

    result = parse_corvis_pdf(pdf_bytes)
    # result.values       → dict of extracted values
    # result.pages_parsed → number of pages successfully parsed (out of 3)
    # result.warnings     → out-of-range value warnings
    # result.errors       → list of pages/fields that failed parsing

Requirements:
    - Ollama running locally with qwen2.5vl:7b pulled
    - pdftoppm (poppler-utils) on PATH for PDF→PNG rendering
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("corneaforge.corvis_parser")

# ── Configuration ───────────────────────────────────────────────────

# Backend: "ollama" or "openai" (for vLLM / SGLang).
VLM_BACKEND = "ollama"

# Ollama backend
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:7b"
OLLAMA_NUM_CTX = 4096

# OpenAI-compatible backend (vLLM / SGLang)
OPENAI_URL = "http://localhost:8200/v1/chat/completions"
OPENAI_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Shared
VLM_TIMEOUT = 120  # seconds per page
VLM_MAX_RETRIES = 2  # retries per page on transient errors
VLM_RETRY_BACKOFF = 2.0  # seconds, doubles each retry
PDF_DPI = 200

# ── Prompts ─────────────────────────────────────────────────────────

_EXTRACT_PROMPT = (
    "This is a page from a Corvis ST PDF report. "
    "Identify the page type and extract the relevant values.\n\n"
    "If IOP/Pachymetry page, return:\n"
    '{"page_type": "iop_pachy", "IOP_mmHg": <number>, "CCT_um": <number>}\n\n'
    "If Dynamic Corneal Response page (bottom-left table with "
    "Applanation 1/2 rows for Length and Velocity, Highest Concavity row "
    "for Peak Distance, Radius, Deformation Amplitude — "
    "note A2 Velocity is NEGATIVE), return:\n"
    '{"page_type": "dynamic_response", '
    '"A1_Velocity_ms": <number>, "A2_Velocity_ms": <negative number>, '
    '"Peak_Distance_mm": <number>, "Deformation_Amplitude_mm": <number>, '
    '"A1_Length_mm": <number>, "A2_Length_mm": <number>, '
    '"Radius_HC_mm": <number>}\n\n'
    "If Vinciguerra Screening Report page (summary table on the right "
    "with CBI, SSI, SP-A1), return:\n"
    '{"page_type": "vinciguerra", '
    '"SP_A1": <number>, "ARTh": <number>, "bIOP_mmHg": <number>, '
    '"DA_Ratio": <number>, "Inverse_Concave_Radius_mm": <number>, '
    '"CBI": <number>, "SSI": <number>}\n\n'
    "Return ONLY the JSON for the matching page type."
)

# Page types and human-readable labels (for error messages).
_PAGE_TYPE_LABELS = {
    "iop_pachy": "IOP/Pachymetry",
    "dynamic_response": "Dynamic Corneal Response",
    "vinciguerra": "Vinciguerra Screening Report",
}

# Maps VLM JSON keys → CorvisInput field names (server.py).
_KEY_TO_CORVIS_FIELD = {
    "IOP_mmHg": "iop",
    "CCT_um": "cct",
    "A1_Velocity_ms": "a1_velocity",
    "A2_Velocity_ms": "a2_velocity",
    "Peak_Distance_mm": "peak_distance",
    "Deformation_Amplitude_mm": "da_max",
    "A1_Length_mm": "a1_length",
    "A2_Length_mm": "a2_length",
    "Radius_HC_mm": "integrated_radius",
    "SP_A1": "sp_a1",
    "ARTh": "arth",
    "bIOP_mmHg": "biop",
    "DA_Ratio": "da_ratio_2mm",
    "Inverse_Concave_Radius_mm": "inverse_concave_radius",
    "CBI": "cbi",
    "SSI": "ssi",
}

# Plausible clinical ranges for sanity checks.
_VALUE_RANGES = {
    "iop": (5, 40),
    "cct": (350, 700),
    "a1_velocity": (0.0, 0.5),
    "a2_velocity": (-0.8, 0.0),
    "peak_distance": (2, 8),
    "da_max": (0.5, 2.0),
    "a1_length": (1.0, 4.0),
    "a2_length": (1.0, 4.0),
    "integrated_radius": (4, 12),
    "sp_a1": (20, 250),
    "arth": (100, 800),
    "biop": (5, 40),
    "da_ratio_2mm": (1, 10),
    "inverse_concave_radius": (0.0, 0.5),
    "cbi": (0.0, 1.0),
    "ssi": (0.5, 2.0),
}


# ── Result dataclass ────────────────────────────────────────────────


@dataclass
class CorvisParseResult:
    """Result of parsing a Corvis ST PDF."""

    values: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_responses: dict[int, str] = field(default_factory=dict)
    pages_parsed: int = 0


# ── Core functions ──────────────────────────────────────────────────


def parse_corvis_pdf(
    pdf_bytes: bytes,
    *,
    backend: str = VLM_BACKEND,
    timeout: int = VLM_TIMEOUT,
) -> CorvisParseResult:
    """Parse a Corvis ST PDF and extract biomechanical values.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw bytes of the Corvis ST PDF.
    backend : str
        ``"ollama"`` or ``"openai"`` (vLLM / SGLang).
    timeout : int
        Timeout per page in seconds.

    Returns
    -------
    CorvisParseResult
        Extracted values, warnings for out-of-range values, errors for
        pages that failed.
    """
    result = CorvisParseResult()

    # Validate input
    if not pdf_bytes:
        result.errors.append("Empty PDF data")
        return result
    if not pdf_bytes[:5].startswith(b"%PDF-"):
        result.errors.append("Not a valid PDF file (bad magic bytes)")
        return result

    # Render PDF pages to PNGs
    page_paths = _pdf_to_pngs(pdf_bytes)
    if not page_paths:
        result.errors.append("Failed to render PDF pages")
        return result

    try:
        # Query VLM for all pages in parallel
        page_responses: dict[int, str | Exception] = {}
        with ThreadPoolExecutor(max_workers=len(page_paths)) as pool:
            futures = {
                pool.submit(
                    _query_vlm,
                    page_path,
                    _EXTRACT_PROMPT,
                    backend=backend,
                    timeout=timeout,
                ): page_idx
                for page_idx, page_path in enumerate(page_paths)
            }
            for future in as_completed(futures):
                page_idx = futures[future]
                try:
                    page_responses[page_idx] = future.result()
                except Exception as exc:
                    page_responses[page_idx] = exc

        # Process responses in page order
        found_types: set[str] = set()
        for page_idx in sorted(page_responses):
            page_num = page_idx + 1
            resp = page_responses[page_idx]

            if isinstance(resp, Exception):
                logger.warning("Page %d OCR failed: %s", page_num, resp)
                result.errors.append(f"Page {page_num}: {resp}")
                continue

            result.raw_responses[page_num] = resp

            parsed = _parse_json_response(resp)
            if parsed is None:
                result.errors.append(f"Page {page_num}: could not parse JSON from VLM response")
                continue

            # Identify page type
            page_type = parsed.pop("page_type", None)
            if page_type not in _PAGE_TYPE_LABELS:
                result.errors.append(f"Page {page_num}: unrecognized page type {page_type!r}")
                continue

            if page_type in found_types:
                result.warnings.append(
                    f"Page {page_num}: duplicate {_PAGE_TYPE_LABELS[page_type]} "
                    f"page (using first occurrence)"
                )
                continue

            found_types.add(page_type)
            result.pages_parsed += 1

            for vlm_key, value in parsed.items():
                corvis_field = _KEY_TO_CORVIS_FIELD.get(vlm_key)
                if corvis_field is None:
                    continue

                if not isinstance(value, (int, float)):
                    result.errors.append(f"Page {page_num}: {vlm_key} is not a number ({value!r})")
                    continue

                value = float(value)

                # Sanity check
                lo, hi = _VALUE_RANGES.get(corvis_field, (None, None))
                if lo is not None and not (lo <= value <= hi):
                    result.warnings.append(
                        f"{corvis_field}={value} outside expected range [{lo}, {hi}]"
                    )

                result.values[corvis_field] = value

        # Report missing page types
        for ptype, label in _PAGE_TYPE_LABELS.items():
            if ptype not in found_types:
                result.errors.append(f"Missing page: {label}")

    finally:
        # Clean up temp PNGs
        for p in page_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass

    return result


def check_vlm_available(backend: str = VLM_BACKEND) -> bool:
    """Check if the VLM backend is reachable."""
    try:
        if backend == "openai":
            # vLLM / SGLang expose /v1/models
            base = OPENAI_URL.rsplit("/v1/", 1)[0]
            req = urllib.request.Request(f"{base}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return len(data.get("data", [])) > 0
        else:
            req = urllib.request.Request(
                OLLAMA_URL.replace("/api/generate", "/api/tags"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                names = [m["name"] for m in data.get("models", [])]
                if OLLAMA_MODEL in names:
                    return True
                model_base = OLLAMA_MODEL.split(":")[0]
                return (
                    any(n.startswith(model_base + ":") for n in names)
                    if ":" not in OLLAMA_MODEL
                    else False
                )
    except Exception:
        return False


# Keep old name as alias for backward compatibility in server.py
check_ollama_available = check_vlm_available


# ── Internal helpers ────────────────────────────────────────────────


def _pdf_to_pngs(pdf_bytes: bytes) -> list[str]:
    """Render PDF pages to PNG files using pdftoppm. Returns list of paths."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_pdf = tmp.name

    try:
        out_prefix = tmp_pdf.replace(".pdf", "")
        proc = subprocess.run(
            ["pdftoppm", "-r", str(PDF_DPI), "-png", tmp_pdf, out_prefix],
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            logger.error("pdftoppm failed: %s", proc.stderr.decode())
            return []

        # pdftoppm creates files like prefix-1.png, prefix-2.png, ...
        parent = Path(tmp_pdf).parent
        prefix = Path(out_prefix).name
        pages = sorted(parent.glob(f"{prefix}-*.png"))
        return [str(p) for p in pages]

    except FileNotFoundError:
        logger.error("pdftoppm not found. Install poppler-utils.")
        return []
    except subprocess.TimeoutExpired:
        logger.error("pdftoppm timed out")
        return []
    finally:
        Path(tmp_pdf).unlink(missing_ok=True)


def _query_vlm(
    image_path: str,
    prompt: str,
    *,
    backend: str,
    timeout: int,
) -> str:
    """Send an image to VLM and return the text response.

    Supports Ollama (``/api/generate``) and OpenAI-compatible
    (``/v1/chat/completions``) backends.  Retries up to
    ``VLM_MAX_RETRIES`` times on transient errors.
    """
    import base64

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    if backend == "openai":
        url = OPENAI_URL
        payload = json.dumps(
            {
                "model": OPENAI_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            }
        ).encode()
    else:
        url = OLLAMA_URL
        payload = json.dumps(
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 512,
                    "num_ctx": OLLAMA_NUM_CTX,
                },
            }
        ).encode()

    last_exc: Exception | None = None
    for attempt in range(1 + VLM_MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())

            if backend == "openai":
                return body["choices"][0]["message"]["content"]
            return body.get("response", "")

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < VLM_MAX_RETRIES:
                wait = VLM_RETRY_BACKOFF * (2**attempt)
                logger.warning(
                    "VLM request failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt + 1,
                    1 + VLM_MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

    raise last_exc  # type: ignore[misc]


def _parse_json_response(text: str) -> dict | None:
    """Extract a JSON object from a VLM response that may contain markdown fences."""
    # Strip <think>...</think> blocks (qwen3 reasoning mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try markdown code fence first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON object
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None
