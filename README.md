# Document Authentication CV

![CI](https://github.com/JulioPradenas/document-authentication-cv/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Demo en vivo](https://img.shields.io/badge/🤗%20Demo-en%20vivo-yellow.svg)](https://huggingface.co/spaces/Pradnas/document-authentication-dashboard)

Sistema end-to-end de autenticación de documentos con **EfficientNet-B0 + Grad-CAM**.
Detecta timbres fiscales y documentos de identidad falsificados, con explicación
visual mediante mapa de calor.
Proyecto de portafolio orientado a la autenticación de timbres fiscales tipo SICPA
(cumplimiento tributario para tabaco/impuestos).

Un pipeline de MLOps completo: generación de datos sintéticos → preprocesamiento →
fine-tuning en dos fases → explicabilidad → API REST + dashboard → informes PDF →
comparación de modelos → robustez/quality gating → MLflow Model Registry.

> **Demo en vivo:** https://huggingface.co/spaces/Pradnas/document-authentication-dashboard
>
> Consulta [docs/ESTUDIO_DEL_PROYECTO.md](docs/ESTUDIO_DEL_PROYECTO.md) para el
> estudio completo de toma de decisiones (por qué este stack, insights de
> ingeniería, resultados).

---

## Arquitectura

```
                     ┌──────────────────────────────────────────────────┐
Imagen (cualquier    │  ImageQualityAssessor (gate opcional)            │
 resolución)         │  nitidez · exposición · resolución · contraste   │
        ──────────► │  falla → label='rejected' (omite inferencia)     │
                     └───────────────────┬──────────────────────────────┘
                                         │  pasa
                     ┌───────────────────▼──────────────────────────────┐
                     │  DocumentPreprocessor                            │
                     │  corrección perspectiva → denoise → CLAHE        │
                     │  → resize bicúbico (224×224) → normalize ImageNet│
                     └───────────────────┬──────────────────────────────┘
                                         │  tensor float32 (3, 224, 224)
                     ┌───────────────────▼──────────────────────────────┐
                     │  EfficientNet-B0 (fine-tuned, dos fases)         │
                     │  Fase A: congela backbone, entrena cabeza (5 ep) │
                     │  Fase B: descongela últimos 2 bloques, LR 1e-4   │
                     │  Cabeza: Dropout → Linear(1280,256) → ReLU       │
                     │          → Dropout → Linear(256,1) → Sigmoid     │
                     └──────────┬────────────────────┬──────────────────┘
                                │ P(falsif.) ∈ [0,1] │ backward pass
                     ┌──────────▼──────┐  ┌──────────▼──────────────────┐
                     │  Decisión       │  │  Mapa Grad-CAM++            │
                     │  umbral=0.50    │  │  superpuesto a la imagen    │
                     │  → auténtico    │  │  + región más activada      │
                     │  → falsificado  │  └──────────────┬──────────────┘
                     └────────┬────────┘                 │
                              └────────────┬─────────────┘
                                  ┌─────────▼──────────┐
                                  │  Informe PDF (A4)  │
                                  └────────────────────┘
```

La carga del modelo es flexible: la API sirve desde un checkpoint local por
defecto, o desde el **MLflow Model Registry** por alias de despliegue
(`production`/`staging`) cuando `MODEL_REGISTRY_ALIAS` está definido, con fallback
automático si el registry no está disponible.

## Estado del proyecto y resultados

> **El pipeline está completo y totalmente testeado.** El repositorio **no incluye
> un checkpoint entrenado** (los `*.pt` están en `.gitignore`). El [demo en
> vivo](https://huggingface.co/spaces/Pradnas/document-authentication-dashboard)
> corre un checkpoint entrenado sobre un **dataset sintético pequeño** (ver
> [`scripts/build_training_data.py`](scripts/build_training_data.py)): produce
> verdictos reales y separables, pero es una **tarea de juguete** con
> perturbaciones visibles, no forense documental real. Entrenar con MIDV-500 es el
> paso final, no un rediseño (ver [estudio §7](docs/ESTUDIO_DEL_PROYECTO.md)). El
> valor demostrado aquí es la **arquitectura MLOps end-to-end**, independiente de
> la precisión del modelo.

**Calidad de ingeniería (verificada):**

| Métrica | Valor |
|--------|-------|
| Suite de tests | 272 tests, 87% de cobertura |
| Type checking | 24/24 módulos pasan mypy |
| CI | lint + type-check + tests + docker-build, todo en verde |
| Latencia de inferencia (CPU, batch=1) | ~330 ms/imagen |
| Tamaño del checkpoint | 17.6 MB (EfficientNet-B0, 5.3M params) |

El generador de falsificaciones sintéticas cubre 4 tipos — `text_blur`,
`color_shift`, `splicing`, `hologram_noise` — en 3 niveles de severidad. El
análisis de robustez (notebook 07) caracteriza el comportamiento del modelo y del
quality gate frente a 5 degradaciones de captura. La comparación de backbones
(notebook 06) mide EfficientNet-B0 vs ResNet-18 vs MobileNetV3-Small en precisión,
latencia y tamaño.

## Inicio rápido

```bash
# Instalar dependencias (requiere uv ≥ 0.4)
uv sync --all-extras

# Generar 20 muestras sintéticas de prueba
make samples

# Levantar la API REST
make run-api          # → http://localhost:8000/docs

# Levantar el dashboard de Streamlit
make run-dashboard    # → http://localhost:8501
```

### Entrenar el checkpoint de demo (datos sintéticos, ruta T1)

```bash
# Genera un set balanceado con falsificaciones visibles (separables)
uv run python scripts/build_training_data.py --n-per-class 200

# Fine-tuning en dos fases → models/saved/efficientnet_b0_best.pt
uv run python scripts/train_demo.py
```

### Autenticar un documento vía API

```bash
# Codifica una imagen y llama al endpoint (con quality gating activado)
IMAGE_B64=$(base64 -i ruta/al/documento.jpg)
curl -s -X POST http://localhost:8000/authenticate \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$IMAGE_B64\", \"return_gradcam\": true, \"check_quality\": true}" \
  | python -m json.tool
```

Respuesta:
```json
{
  "label": "forged",
  "probability": 0.9241,
  "threshold": 0.5,
  "gradcam_b64": "<PNG en base64>",
  "most_activated_region": {"x0": 42, "y0": 18, "x1": 183, "y1": 156,
                             "cx": 112, "cy": 87, "mean_activation": 0.821},
  "inference_ms": 87.3,
  "quality": {"passed": true, "sharpness": 412.5, "brightness": 138.0,
              "resolution": [768, 1024], "reasons": []}
}
```

Cuando `check_quality` está activado y la imagen no pasa el gate (muy borrosa,
oscura o de baja resolución), `label` pasa a `"rejected"`, se omite Grad-CAM y
`quality.reasons` lista los motivos.

### Endpoints

| Método | Ruta | Propósito |
|--------|------|-----------|
| `POST` | `/authenticate` | Clasifica una imagen (+ Grad-CAM y quality gate opcionales) |
| `POST` | `/authenticate/batch` | Clasifica hasta 32 imágenes en una llamada |
| `POST` | `/report` | Clasifica y devuelve un informe PDF (`application/pdf`) |
| `GET`  | `/health` | Liveness + disponibilidad del modelo |
| `GET`  | `/model/info` | Arquitectura, conteo de parámetros, metadata del checkpoint |

```bash
# Batch
curl -s -X POST http://localhost:8000/authenticate/batch \
  -H "Content-Type: application/json" \
  -d '{"images": [{"image_b64": "..."},  {"image_b64": "..."}]}'

# Informe PDF (guardado a disco)
curl -s -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$IMAGE_B64\", \"return_gradcam\": true}" \
  -o informe.pdf
```

## Dataset

**MIDV-500** — 500 clips de video que cubren 50 tipos de documento (pasaportes,
cédulas, licencias de conducir).

- ~15.000 frames utilizables tras la extracción
- Auténticos: frames originales; Falsificados: perturbados sintéticamente (4 tipos)
- Los datos crudos **no se commitean** — `data/raw/` está en `.gitignore`

```bash
# Descargar MIDV-500 (requiere el extra datasets)
uv sync --extra datasets
uv run python scripts/download_dataset.py

# Generar solo muestras sintéticas de prueba (sin descarga)
make samples
```

## Stack tecnológico

| Componente          | Tecnología                                   |
|---------------------|----------------------------------------------|
| Modelo              | EfficientNet-B0 (torchvision)                |
| Comparación backbones | ResNet-18, MobileNetV3-Small (ablation)    |
| Explicabilidad      | Grad-CAM++ / EigenCAM (grad-cam)             |
| Aumentación         | Albumentations 2.x                           |
| Quality gating      | Métricas sin referencia de OpenCV            |
| Informes            | reportlab (PDF A4 de una página)             |
| Tracking de experimentos | MLflow 3.x (backend SQLite)             |
| Model registry      | MLflow registry (aliases: staging/production)|
| API                 | FastAPI + Uvicorn                            |
| Dashboard           | Streamlit                                    |
| Contenedorización   | Docker multi-stage (api + dashboard)         |
| CI                  | GitHub Actions + uv (torch CPU-only en Linux)|
| Linting             | Ruff + mypy                                  |
| Testing             | pytest + pytest-cov (272 tests, 87%)         |
| Python              | 3.11                                         |

## Estructura del proyecto

```
document_authentication/
├── src/
│   ├── data/
│   │   ├── augmentation.py     # SyntheticForgeryGenerator (4 tipos × 3 severidades)
│   │   └── loader.py           # DocumentDataset, create_dataloaders
│   ├── preprocessing/
│   │   ├── pipeline.py         # DocumentPreprocessor (perspectiva+CLAHE+denoise)
│   │   ├── quality.py          # ImageQualityAssessor (quality gate sin referencia)
│   │   └── degradations.py     # 5 degradaciones controladas para robustez
│   ├── models/
│   │   ├── classifier.py       # DocumentClassifier (cabeza EfficientNet-B0)
│   │   ├── architectures.py    # DocumentClassifierV2 (factory multi-backbone)
│   │   ├── trainer.py          # Trainer con fine-tuning en dos fases + MLflow
│   │   ├── evaluator.py        # ModelEvaluator (ROC/PR/F1/búsqueda de umbral)
│   │   ├── comparator.py       # ModelComparator (ablation + MLflow)
│   │   └── registry.py         # ModelRegistry (versionado + staging/production)
│   ├── explainability/
│   │   ├── gradcam.py          # GradCAMExplainer (gradcam / gradcam++ / eigencam)
│   │   └── visualizer.py       # overlay_heatmap, most_activated_region
│   └── reporting/
│       └── pdf_report.py       # PDFReportGenerator (informe A4 de una página)
├── api/
│   ├── main.py                 # App FastAPI (5 endpoints, carga registry-aware)
│   ├── predictor.py            # DocumentPredictor (inferencia + Grad-CAM + calidad)
│   └── schemas.py              # Modelos Pydantic de request/response
├── dashboard/
│   └── app.py                  # UI Streamlit (verificador + demo + stats, español)
├── notebooks/
│   ├── 01_eda_dataset.ipynb            05_evaluation.ipynb
│   ├── 02_preprocessing_pipeline.ipynb 06_model_comparison.ipynb
│   ├── 03_model_training.ipynb         07_robustness_analysis.ipynb
│   └── 04_gradcam_analysis.ipynb       08_model_registry.ipynb
├── tests/                      # suite pytest (272 tests, 87% de cobertura)
├── docs/
│   └── ESTUDIO_DEL_PROYECTO.md # estudio de decisiones e ingeniería
├── deploy/
│   └── huggingface/            # README + guía de despliegue en HF Spaces
├── scripts/
│   ├── download_dataset.py
│   ├── generate_samples.py
│   ├── build_training_data.py  # genera el dataset sintético de entrenamiento (T1)
│   └── train_demo.py           # entrena el checkpoint de demo
├── models/saved/               # checkpoints (gitignored salvo .gitkeep)
├── reports/figures/            # figuras de salida de los notebooks
├── Dockerfile                  # multi-stage: builder / api / dashboard
├── pyproject.toml
└── Makefile
```

## Desarrollo

```bash
make install        # uv sync --all-extras
make fix            # ruff check --fix + ruff format
make test           # pytest --cov=src --cov=api

# Ejecutar todos los notebooks (requiere data/samples)
uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb

# UI de MLflow (ver runs de entrenamiento)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Docker

```bash
# Construir y correr la API
docker build --target api -t doc-auth-api .
docker run -p 8000:8000 doc-auth-api

# Construir y correr el dashboard
docker build --target dashboard -t doc-auth-dashboard .
docker run -p 8501:8501 doc-auth-dashboard
```

## Despliegue (Hugging Face Spaces)

El dashboard se despliega en **Hugging Face Spaces** (Docker SDK) reutilizando el
stage `dashboard` del Dockerfile. La guía paso a paso está en
[deploy/huggingface/DEPLOY.md](deploy/huggingface/DEPLOY.md):

1. `hf auth login` con un token de tipo **Write**.
2. Crear un Space (Docker) y copiar el proyecto.
3. Incluir el checkpoint vía Git LFS (`git add -f models/saved/*.pt`).
4. `git push` → el Space construye y queda en línea.

Demo en vivo: https://huggingface.co/spaces/Pradnas/document-authentication-dashboard

## Model Registry

Registra un checkpoint, promociónalo entre stages de despliegue y sírvelo por alias:

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry(model_name="document-authenticator")
version = registry.register(
    "models/saved/efficientnet_b0_best.pt",
    metrics={"val_f1": 0.94, "val_auc": 0.97},
    description="EfficientNet-B0, fine-tune en dos fases",
)
registry.promote(version, alias="staging")      # validar
registry.promote(version, alias="production")    # desplegar
```

Sirve el modelo de producción desde la API vía variables de entorno (con fallback
al checkpoint local si el registry no está disponible):

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MODEL_REGISTRY_ALIAS=production
uvicorn api.main:app
```

El rollback es atómico: `registry.promote(version_anterior, "production")`.

---

Construido como proyecto de portafolio para forense documental y aplicaciones de
visión por computadora.
