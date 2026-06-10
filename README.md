# Document Authentication CV

![CI](https://github.com/JulioPradenas/document-authentication-cv/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

End-to-end document authentication system using **EfficientNet-B0 + Grad-CAM**.  
Detects forged fiscal stamps and identity documents with visual heatmap explanation.  
Portfolio project targeting SICPA-style fiscal stamp authentication for tobacco/tax compliance.

---

## Architecture

```
                     ┌──────────────────────────────────────────────────┐
Raw image (any res)  │  DocumentPreprocessor                            │
        ──────────► │  perspective correction → denoise → CLAHE        │
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
                     │  → forged       │  └─────────────────────────────┘
                     └─────────────────┘
```

## Results (synthetic dataset — 20 samples)

> **Note:** trained on 20 synthetic samples for pipeline validation only.  
> With MIDV-500 (~15,000 frames) these metrics will be representative.

| Split | Accuracy | Precision | Recall | F1-score | AUC-ROC | AUC-PR |
|-------|----------|-----------|--------|----------|---------|--------|
| Train | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000  | 1.0000 |
| Val   | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000  | 1.0000 |
| Test  | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 1.0000  | 1.0000 |

**EfficientNet-B0:** 5.3M parameters total · Latency p95 < 200 ms on CPU (batch=1)

The model detects all 4 synthetic forgery types: `text_blur`, `color_shift`, `splicing`, `hologram_noise`.

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
# Encode an image and call the endpoint
IMAGE_B64=$(base64 -i path/to/document.jpg)
curl -s -X POST http://localhost:8000/authenticate \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$IMAGE_B64\", \"return_gradcam\": true}" \
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
  "inference_ms": 87.3
}
```

### Batch endpoint

```bash
curl -s -X POST http://localhost:8000/authenticate/batch \
  -H "Content-Type: application/json" \
  -d '{"images": [{"image_b64": "..."},  {"image_b64": "..."}]}'
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

| Component           | Technology                            |
|---------------------|---------------------------------------|
| Model               | EfficientNet-B0 (torchvision)         |
| Explainability      | Grad-CAM++ / EigenCAM (grad-cam)      |
| Augmentation        | Albumentations 2.x                    |
| Experiment tracking | MLflow 3.x (SQLite backend)           |
| API                 | FastAPI + Uvicorn                     |
| Dashboard           | Streamlit                             |
| Containerization    | Docker multi-stage (api + dashboard)  |
| CI                  | GitHub Actions + uv                   |
| Linting             | Ruff + mypy                           |
| Testing             | pytest + pytest-cov                   |
| Python              | 3.11                                  |

## Project Structure

```
document_authentication/
├── src/
│   ├── data/
│   │   ├── augmentation.py     # SyntheticForgeryGenerator (4 types)
│   │   └── loader.py           # DocumentDataset, create_dataloaders
│   ├── preprocessing/
│   │   └── pipeline.py         # DocumentPreprocessor (perspective+CLAHE+denoise)
│   ├── models/
│   │   ├── classifier.py       # DocumentClassifier (EfficientNet-B0 head)
│   │   ├── trainer.py          # Trainer with two-phase fine-tuning + MLflow
│   │   └── evaluator.py        # ModelEvaluator (ROC/PR/F1/threshold search)
│   └── explainability/
│       ├── gradcam.py          # GradCAMExplainer (gradcam / gradcam++ / eigencam)
│       └── visualizer.py       # overlay_heatmap, most_activated_region
├── api/
│   ├── main.py                 # FastAPI app (4 endpoints)
│   ├── predictor.py            # DocumentPredictor (inference + Grad-CAM)
│   └── schemas.py              # Pydantic request/response models
├── dashboard/
│   └── app.py                  # Streamlit UI (verifier + demo + stats)
├── notebooks/
│   ├── 01_eda_dataset.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_gradcam_analysis.ipynb
│   └── 05_evaluation.ipynb
├── tests/                      # pytest suite (~120 tests)
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

---

Built as a portfolio project for document forensics and computer vision applications.
