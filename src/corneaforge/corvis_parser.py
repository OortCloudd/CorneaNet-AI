"""
Corvis ST PDF Parser
====================

Extracts biomechanical values from Corvis ST PDF reports using a Vision
Language Model (Qwen2.5-VL via Ollama).

Corvis ST exports are rasterized screenshots in PDF — no vector text layer.
Traditional OCR (Tesseract) achieves ~25% accuracy on these. A VLM achieves
~94% with structured JSON prompting.

Usage:
    from corneaforge.corvis_parser import parse_corvis_pdf

    result = parse_corvis_pdf(pdf_bytes)
    # result.values  → dict of extracted values
    # result.raw     → per-page raw model responses (for debugging)
    # result.errors  → list of pages/fields that failed parsing

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
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("corneaforge.corvis_parser")

# ── Configuration ───────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:7b"
OLLAMA_NUM_CTX = 4096
OLLAMA_TIMEOUT = 120  # seconds per page
PDF_DPI = 300

# ── Prompts ─────────────────────────────────────────────────────────

_PROMPT_PAGE_1 = (
    "Extract from this Corvis ST IOP/Pachy page. Return ONLY JSON:\n"
    '{"IOP_mmHg": <number>, "CCT_um": <number>}'
)

_PROMPT_PAGE_2 = (
    "This is a Corvis ST Dynamic Corneal Response report.\n"
    "Look at the bottom-left table. It has these rows:\n"
    '- "Applanation 1": Length and Velocity\n'
    '- "Highest Concavity": Peak Distance, Radius, Deformation Amplitude\n'
    '- "Applanation 2": Length and Velocity\n'
    "Note: A2 Velocity is NEGATIVE.\n"
    "Return ONLY JSON:\n"
    '{"A1_Velocity_ms": <number>, "A2_Velocity_ms": <negative number>, '
    '"Peak_Distance_mm": <number>, "Deformation_Amplitude_mm": <number>, '
    '"A1_Length_mm": <number>, "A2_Length_mm": <number>, "Radius_HC_mm": <number>}'
)

_PROMPT_PAGE_3 = (
    "Extract from this Corvis ST Vinciguerra Screening Report page. "
    "Look at the summary table on the right.\n"
    "Return ONLY JSON:\n"
    '{"SP_A1": <number>, "ARTh": <number>, "bIOP_mmHg": <number>, '
    '"DA_Ratio": <number>, "Inverse_Concave_Radius_mm": <number>, '
    '"CBI": <number>, "SSI": <number>}'
)

_PAGE_PROMPTS = {1: _PROMPT_PAGE_1, 2: _PROMPT_PAGE_2, 3: _PROMPT_PAGE_3}

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


# ── Core functions ──────────────────────────────────────────────────


def parse_corvis_pdf(
    pdf_bytes: bytes,
    *,
    ollama_url: str = OLLAMA_URL,
    model: str = OLLAMA_MODEL,
    num_ctx: int = OLLAMA_NUM_CTX,
    timeout: int = OLLAMA_TIMEOUT,
) -> CorvisParseResult:
    """Parse a Corvis ST PDF and extract biomechanical values.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw bytes of the Corvis ST PDF.
    ollama_url : str
        Ollama API endpoint.
    model : str
        VLM model name.
    num_ctx : int
        Context window size for the model.
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

    n_pages = len(page_paths)
    if n_pages < 3:
        result.warnings.append(f"Expected 3 pages, got {n_pages}. Some values may be missing.")

    try:
        for page_num in range(1, min(n_pages + 1, 4)):
            prompt = _PAGE_PROMPTS.get(page_num)
            if not prompt:
                continue

            page_path = page_paths[page_num - 1]
            try:
                raw_response = _query_vlm(
                    page_path,
                    prompt,
                    ollama_url=ollama_url,
                    model=model,
                    num_ctx=num_ctx,
                    timeout=timeout,
                )
                result.raw_responses[page_num] = raw_response

                parsed = _parse_json_response(raw_response)
                if parsed is None:
                    result.errors.append(f"Page {page_num}: could not parse JSON from VLM response")
                    continue

                for vlm_key, value in parsed.items():
                    corvis_field = _KEY_TO_CORVIS_FIELD.get(vlm_key)
                    if corvis_field is None:
                        continue

                    if not isinstance(value, (int, float)):
                        result.errors.append(
                            f"Page {page_num}: {vlm_key} is not a number ({value!r})"
                        )
                        continue

                    value = float(value)

                    # Sanity check
                    lo, hi = _VALUE_RANGES.get(corvis_field, (None, None))
                    if lo is not None and not (lo <= value <= hi):
                        result.warnings.append(
                            f"{corvis_field}={value} outside expected range [{lo}, {hi}]"
                        )

                    result.values[corvis_field] = value

            except Exception as e:
                logger.warning("Page %d OCR failed: %s", page_num, e)
                result.errors.append(f"Page {page_num}: {e}")

    finally:
        # Clean up temp PNGs
        for p in page_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass

    return result


def check_ollama_available(ollama_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        req = urllib.request.Request(
            ollama_url.replace("/api/generate", "/api/tags"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return model in models
    except Exception:
        return False


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
    ollama_url: str,
    model: str,
    num_ctx: int,
    timeout: int,
) -> str:
    """Send an image to Ollama VLM and return the text response."""
    import base64

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 512, "num_ctx": num_ctx},
        }
    ).encode()

    req = urllib.request.Request(
        ollama_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())

    return body.get("response", "")


def _parse_json_response(text: str) -> dict | None:
    """Extract a JSON object from a VLM response that may contain markdown fences."""
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
