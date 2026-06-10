# Document Authentication CV

![CI](https://github.com/JulioPradenas/document-authentication-cv/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

End-to-end document authentication system using EfficientNet-B0 + Grad-CAM. Detects forged fiscal stamps and documents with visual heatmap explanation. Portfolio project targeting SICPA-style fiscal stamp authentication for tobacco/tax compliance use cases.

---

## Overview

- Binary classifier (authentic vs. forged) fine-tuned from EfficientNet-B0
- Grad-CAM heatmaps highlight the regions that triggered the decision
- REST API (FastAPI) for single-image inference
- Streamlit dashboard for interactive exploration
- MLflow experiment tracking
- Docker-ready deployment

## Architecture

_Diagram TBD — Phase 3_

```
Image → Preprocessing → EfficientNet-B0 → Authentic / Forged
                                        ↘ Grad-CAM heatmap
```

## Quick Start

```bash
# Install dependencies (requires uv)
uv sync --all-extras

# Start the API
make run-api        # http://localhost:8000

# Start the dashboard
make run-dashboard  # http://localhost:8501
```

## Metrics

_TBD — populated after Phase 4 (training)_

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| AUC-ROC   | —     |
| F1-Score  | —     |

## Dataset

_TBD — populated after Phase 1 (EDA)_

Dataset consists of authentic and forged fiscal stamp images. Raw data lives outside this repo (not tracked in git). See `scripts/download_data.py` for acquisition instructions.

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Model           | EfficientNet-B0 (torchvision)       |
| Explainability  | Grad-CAM (pytorch-grad-cam)         |
| Augmentation    | Albumentations                      |
| Experiment tracking | MLflow                          |
| API             | FastAPI + Uvicorn                   |
| Dashboard       | Streamlit                           |
| Containerization| Docker (multi-stage build)          |
| CI              | GitHub Actions + uv                 |
| Linting         | Ruff + mypy                         |
| Python          | 3.11                                |

## Development

```bash
make install        # uv sync --all-extras
make fix            # ruff check --fix + ruff format
make test           # pytest with coverage
make run-api        # uvicorn on :8000
make run-dashboard  # streamlit on :8501
```

Branch naming convention for phases:
- `feature/phase-1-eda`
- `feature/phase-2-preprocessing`
- `feature/phase-3-model`
- `feature/phase-4-training`
- `feature/phase-5-gradcam`
- `feature/phase-6-api`
- `feature/phase-7-dashboard`
- `feature/phase-8-reporting`

---

Built as part of a portfolio for document forensics and computer vision applications.
