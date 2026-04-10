# CorneaForge

On-premise corneal topography pipeline for the CSO MS-39. Parses raw CSV exports, computes ~2,400 clinical features, and serves predictions via FastAPI.

Built for production inference at CHNO des Quinze-Vingts, Paris.

## Architecture

```
MS-39 CSV ──> core.py (parse + clean + polar-to-Cartesian)
                 │
                 ├──> computed_indices.py  (~2,400 features: OPD, K-readings,
                 │       shape, screening, epithelial, BAD-D, Zernike, KC class)
                 │
                 ├──> nn_pipeline.py       (13-channel 224x224 float32 tensors)
                 ├──> visual_pipeline.py   (clinical colormap PNGs)
                 ├──> descriptive_stats.py (~1,560 tabular features)
                 │
                 └──> validate.py          (structural pre-flight checks)

server.py ──> FastAPI: upload CSV, get features + predictions + maps
```

`computed_indices.py` is the core computation engine. It includes ray-tracing (vectorial Snell's law, Newton iteration for posterior intersection, focal estimation) and biquadratic local surface fitting, all validated against MS-39 ground truth.

## Setup

```bash
uv sync --extra dev --extra server
```

## Usage

```bash
make serve        # dev server on :8000
make test         # 280 tests
make lint         # ruff check + format
make check        # lint + test
```

### API

```bash
curl -X POST localhost:8000/predict/disease_classification \
  -F "ms39_individual=@patient.csv"
```

## Publications

- Perez E\*, Louissi N\*, et al. "Machine Learning Model for Predicting Visual Acuity Improvement After Intrastromal Corneal Ring Surgery in Patients With Keratoconus." *Cornea*, 2025.
- Borderie VM, Georgeon C, Louissi N, et al. "CorvisST biomechanical indices in the diagnosis of corneal stromal and endothelial disorders." *British Journal of Ophthalmology*, 2025.

## License

MIT
