# Document Authentication CV

![CI](https://github.com/JulioPradenas/document-authentication-cv/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

End-to-end document authentication system using **EfficientNet-B0 + Grad-CAM**.  
Detects forged fiscal stamps and identity documents with visual heatmap explanation.  
Portfolio project targeting SICPA-style fiscal stamp authentication for tobacco/tax compliance.

A full MLOps pipeline: synthetic data generation → preprocessing → two-phase
fine-tuning → explainability → REST API + dashboard → PDF reports → model
comparison → robustness/quality gating → MLflow Model Registry.

> See [docs/ESTUDIO_DEL_PROYECTO.md](docs/ESTUDIO_DEL_PROYECTO.md) for the full
> decision-making study (why this stack, key engineering insights, results).

---

## Architecture

```
                     ┌──────────────────────────────────────────────────┐
Raw image (any res)  │  ImageQualityAssessor (optional gate)            │
        ──────────► │  sharpness · exposure · resolution · contrast    │
                     │  fail → label='rejected' (skip inference)        │
                     └───────────────────┬──────────────────────────────┘
                                         │  pass
                     ┌───────────────────▼──────────────────────────────┐
                     │  DocumentPreprocessor                            │
                     │  perspective correction → denoise → CLAHE        │
                     │  → bicubic resize (224×224) → ImageNet normalize │
                     └───────────────────┬──────────────────────────────┘
                                         │  float32 tensor (3, 224, 224)
                     ┌───────────────────▼──────────────────────────────┐
                     │  EfficientNet-B0 (fine-tuned, two-phase)         │
                     │  Phase A: freeze backbone, train head  (5 ep)    │
                     │  Phase B: unfreeze last 2 blocks, LR 1e-4 (15 ep)│
                     │  Head: Dropout → Linear(1280,256) → ReLU        │
                     │        → Dropout → Linear(256,1) → Sigmoid      │
                     └──────────┬────────────────────┬──────────────────┘
                                │ P(forged) ∈ [0,1]  │ backward pass
                     ┌──────────▼──────┐  ┌──────────▼──────────────────┐
                     │  Decision       │  │  Grad-CAM++ heatmap         │
                     │  threshold=0.50 │  │  overlay on original image  │
                     │  → authentic    │  │  + most-activated region    │
                     │  → forged       │  └──────────────┬──────────────┘
                     └────────┬────────┘                 │
                              └────────────┬─────────────┘
                                  ┌─────────▼──────────┐
                                  │  PDF report (A4)   │
                                  └────────────────────┘
```

Model loading is pluggable: the API serves from a local checkpoint by default, or
from the **MLflow Model Registry** by deployment alias (`production`/`staging`)
when `MODEL_REGISTRY_ALIAS` is set, with automatic fallback if the registry is
unreachable.

## Project status & results

> **The pipeline is complete and fully tested; the shipped checkpoint is not yet
> trained on real data.** It produces near-random probabilities (~0.5), so the
> authentic/forged verdict is not yet reliable — this is deliberate. The value
> demonstrated here is the **end-to-end MLOps architecture**, which is independent
> of model accuracy. Training on MIDV-500 is the final step, not a redesign
> (see [study §7](docs/ESTUDIO_DEL_PROYECTO.md)).

**Engineering quality (verified):**

| Metric | Value |
|--------|-------|
| Test suite | 270 tests, 87% coverage |
| Type checking | 24/24 modules pass mypy |
| CI | lint + type-check + tests + docker-build, all green |
| Inference latency (CPU, batch=1) | ~330 ms/image |
| Checkpoint size | 17.6 MB (EfficientNet-B0, 5.3M params) |

The synthetic forgery generator covers 4 types — `text_blur`, `color_shift`,
`splicing`, `hologram_noise` — at 3 severity levels. The robustness analysis
(notebook 07) characterizes model/quality-gate behavior under 5 capture
degradations. Backbone comparison (notebook 06) benchmarks EfficientNet-B0 vs
ResNet-18 vs MobileNetV3-Small on accuracy, latency and size.

## Quick Start

```bash
# Install dependencies (requires uv ≥ 0.4)
uv sync --all-extras

# Generate 20 synthetic training samples
make samples

# Start the REST API
make run-api          # → http://localhost:8000/docs

# Start the Streamlit dashboard
make run-dashboard    # → http://localhost:8501
```

### Authenticate a document via API

```bash
# Encode an image and call the endpoint (with quality gating enabled)
IMAGE_B64=$(base64 -i path/to/document.jpg)
curl -s -X POST http://localhost:8000/authenticate \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$IMAGE_B64\", \"return_gradcam\": true, \"check_quality\": true}" \
  | python -m json.tool
```

Response:
```json
{
  "label": "forged",
  "probability": 0.9241,
  "threshold": 0.5,
  "gradcam_b64": "<base64 PNG>",
  "most_activated_region": {"x0": 42, "y0": 18, "x1": 183, "y1": 156,
                             "cx": 112, "cy": 87, "mean_activation": 0.821},
  "inference_ms": 87.3,
  "quality": {"passed": true, "sharpness": 412.5, "brightness": 138.0,
              "resolution": [768, 1024], "reasons": []}
}
```

When `check_quality` is enabled and the image fails the gate (too blurry, dark,
or low-resolution), `label` becomes `"rejected"`, Grad-CAM is skipped, and
`quality.reasons` lists why.

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/authenticate` | Classify one image (+ optional Grad-CAM, quality gate) |
| `POST` | `/authenticate/batch` | Classify up to 32 images in one call |
| `POST` | `/report` | Classify and return a one-page PDF report (`application/pdf`) |
| `GET`  | `/health` | Liveness + model readiness |
| `GET`  | `/model/info` | Architecture, param counts, checkpoint metadata |

```bash
# Batch
curl -s -X POST http://localhost:8000/authenticate/batch \
  -H "Content-Type: application/json" \
  -d '{"images": [{"image_b64": "..."},  {"image_b64": "..."}]}'

# PDF report (saved to disk)
curl -s -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$IMAGE_B64\", \"return_gradcam\": true}" \
  -o report.pdf
```

## Dataset

**MIDV-500** — 500 video clips covering 50 document types (passports, IDs, driver's licenses).

- ~15,000 usable frames after extraction
- Authentic: original frames; Forged: synthetically perturbed (4 types)
- Raw data is **not committed** — `data/raw/` is gitignored

```bash
# Download MIDV-500 (requires datasets extra)
uv sync --extra datasets
uv run python scripts/download_dataset.py

# Generate synthetic test samples only (no download)
make samples
```

## Tech Stack

| Component           | Technology                                   |
|---------------------|----------------------------------------------|
| Model               | EfficientNet-B0 (torchvision)                |
| Backbone comparison | ResNet-18, MobileNetV3-Small (ablation)      |
| Explainability      | Grad-CAM++ / EigenCAM (grad-cam)             |
| Augmentation        | Albumentations 2.x                           |
| Quality gating      | OpenCV no-reference metrics                  |
| Reports             | reportlab (one-page A4 PDF)                  |
| Experiment tracking | MLflow 3.x (SQLite backend)                  |
| Model registry      | MLflow registry (aliases: staging/production)|
| API                 | FastAPI + Uvicorn                            |
| Dashboard           | Streamlit                                    |
| Containerization    | Docker multi-stage (api + dashboard)         |
| CI                  | GitHub Actions + uv (CPU-only torch on Linux)|
| Linting             | Ruff + mypy                                  |
| Testing             | pytest + pytest-cov (270 tests, 87%)         |
| Python              | 3.11                                         |

## Project Structure

```
document_authentication/
├── src/
│   ├── data/
│   │   ├── augmentation.py     # SyntheticForgeryGenerator (4 types × 3 severities)
│   │   └── loader.py           # DocumentDataset, create_dataloaders
│   ├── preprocessing/
│   │   ├── pipeline.py         # DocumentPreprocessor (perspective+CLAHE+denoise)
│   │   ├── quality.py          # ImageQualityAssessor (no-reference quality gate)
│   │   └── degradations.py     # 5 controlled degradations for robustness testing
│   ├── models/
│   │   ├── classifier.py       # DocumentClassifier (EfficientNet-B0 head)
│   │   ├── architectures.py    # DocumentClassifierV2 (multi-backbone factory)
│   │   ├── trainer.py          # Trainer with two-phase fine-tuning + MLflow
│   │   ├── evaluator.py        # ModelEvaluator (ROC/PR/F1/threshold search)
│   │   ├── comparator.py       # ModelComparator (ablation study + MLflow)
│   │   └── registry.py         # ModelRegistry (versioning + staging/production)
│   ├── explainability/
│   │   ├── gradcam.py          # GradCAMExplainer (gradcam / gradcam++ / eigencam)
│   │   └── visualizer.py       # overlay_heatmap, most_activated_region
│   └── reporting/
│       └── pdf_report.py       # PDFReportGenerator (one-page A4 report)
├── api/
│   ├── main.py                 # FastAPI app (5 endpoints, registry-aware loading)
│   ├── predictor.py            # DocumentPredictor (inference + Grad-CAM + quality)
│   └── schemas.py              # Pydantic request/response models
├── dashboard/
│   └── app.py                  # Streamlit UI (verifier + demo + stats, español)
├── notebooks/
│   ├── 01_eda_dataset.ipynb            05_evaluation.ipynb
│   ├── 02_preprocessing_pipeline.ipynb 06_model_comparison.ipynb
│   ├── 03_model_training.ipynb         07_robustness_analysis.ipynb
│   └── 04_gradcam_analysis.ipynb       08_model_registry.ipynb
├── tests/                      # pytest suite (270 tests, 87% coverage)
├── docs/
│   └── ESTUDIO_DEL_PROYECTO.md # decision-making & engineering study
├── scripts/
│   ├── download_dataset.py
│   └── generate_samples.py
├── models/saved/               # checkpoints (gitignored except .gitkeep)
├── reports/figures/            # notebook output figures
├── Dockerfile                  # multi-stage: builder / api / dashboard
├── pyproject.toml
└── Makefile
```

## Development

```bash
make install        # uv sync --all-extras
make fix            # ruff check --fix + ruff format
make test           # pytest --cov=src --cov=api

# Run all notebooks (requires data/samples)
uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb

# MLflow UI (view training runs)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Docker

```bash
# Build and run the API
docker build --target api -t doc-auth-api .
docker run -p 8000:8000 doc-auth-api

# Build and run the dashboard
docker build --target dashboard -t doc-auth-dashboard .
docker run -p 8501:8501 doc-auth-dashboard
```

## Model Registry

Register a checkpoint, promote it through deployment stages, and serve it by alias:

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry(model_name="document-authenticator")
version = registry.register(
    "models/saved/efficientnet_b0_best.pt",
    metrics={"val_f1": 0.94, "val_auc": 0.97},
    description="EfficientNet-B0, two-phase fine-tune",
)
registry.promote(version, alias="staging")      # validate
registry.promote(version, alias="production")    # deploy
```

Serve the production model from the API via environment variables (falls back to
the local checkpoint if the registry is unreachable):

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MODEL_REGISTRY_ALIAS=production
uvicorn api.main:app
```

Rollback is atomic: `registry.promote(previous_version, "production")`.

---

Built as a portfolio project for document forensics and computer vision applications.
