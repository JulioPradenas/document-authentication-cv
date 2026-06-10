# 🔐 Document Authentication con Computer Vision
## Plan de Proyecto — V1 + V2 completo (3-4 semanas)

> **Objetivo:** Sistema end-to-end de autenticación de documentos mediante Computer Vision. Detecta si un documento (factura, etiqueta fiscal, sello de autenticidad) es auténtico o falsificado, con Grad-CAM que muestra exactamente qué zona activó la detección. Directamente alineado con el negocio core de SICPA (autenticación de etiquetas fiscales para tabaco y trazabilidad tributaria).

> **Narrativa de negocio simulada:** Sistema de inspección automática para verificación de sellos fiscales y documentos tributarios en punto de control — el operador sube una imagen y recibe veredicto + mapa de calor visual en segundos.

> **Repo sugerido:** `JulioPradenas/document-authentication-cv`

---

## 📁 Estructura del Proyecto

```
document-authentication-cv/
│
├── data/
│   ├── raw/                          # Imágenes originales descargadas
│   ├── processed/                    # Imágenes preprocesadas (224x224, normalizado)
│   ├── augmented/                    # Imágenes con augmentation aplicada
│   └── samples/                      # Muestra pequeña para tests (20 imágenes)
│
├── notebooks/
│   ├── 01_eda_dataset.ipynb          # Exploración visual del dataset
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_gradcam_analysis.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_error_analysis.ipynb       # V2: análisis de errores y casos difíciles
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py                 # Carga y split de dataset
│   │   └── augmentation.py          # Pipeline de augmentation con Albumentations
│   ├── preprocessing/
│   │   └── pipeline.py              # OpenCV: corrección perspectiva, normalización, denoising
│   ├── models/
│   │   ├── classifier.py            # EfficientNet-B0 fine-tuned wrapper
│   │   ├── trainer.py               # Loop de entrenamiento con early stopping
│   │   └── evaluator.py             # Métricas: accuracy, F1, AUC-ROC, confusion matrix
│   ├── explainability/
│   │   ├── gradcam.py               # Implementación Grad-CAM
│   │   └── visualizer.py            # Superposición heatmap sobre imagen original
│   └── reporting/                   # V2
│       └── pdf_generator.py         # Reporte PDF por documento analizado
│
├── api/
│   ├── main.py                      # FastAPI app
│   ├── schemas.py                   # Pydantic v2 models
│   └── predictor.py                 # Inference pipeline
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard
│
├── models/
│   └── saved/                       # Pesos del modelo (.pt) + metadata
│
├── reports/
│   ├── figures/                     # Curvas ROC, confusion matrix, Grad-CAM samples
│   └── pdf/                         # V2: reportes PDF generados
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_classifier.py
│   ├── test_gradcam.py
│   ├── test_api.py
│   └── test_pdf_generator.py        # V2
│
├── pyproject.toml
├── Makefile
├── Dockerfile
├── .github/workflows/ci.yml
└── README.md
```

---

## 📊 Dataset

**Dataset principal:** [MIDV-500](https://github.com/fcakyon/midv500) — documentos de identidad reales fotografiados en condiciones variadas.

| Campo | Detalle |
|---|---|
| Contenido | 500 video clips de 50 tipos de documentos (pasaportes, licencias, IDs) |
| Condiciones | Fondos variados, iluminación variable, perspectiva, oclusión parcial |
| Labels | Por tipo de documento (usaremos auténtico vs manipulado) |
| Tamaño | ~1.2GB descargable vía script |

**Estrategia de labels auténtico/falso:**
MIDV-500 no tiene labels de falsificación nativos. Se construyen así:
- **Clase 0 (auténtico):** imágenes originales del dataset
- **Clase 1 (manipulado):** imágenes con perturbaciones sintéticas controladas aplicadas con OpenCV y Albumentations:
  - Alteración de texto (blur localizado en zona de texto)
  - Cambio de color en región del sello
  - Copia-pega de región de otra imagen (splicing)
  - Ruido gaussiano localizado en zona de holograma simulado

**¿Por qué esta estrategia es válida?**
Es el enfoque estándar en investigación de document forensics cuando no se tienen datasets de documentos falsos reales (que son ilegales de distribuir). Se documenta explícitamente en el README — eso demuestra criterio, no limitación.

**Split temporal:**
- Train: 70% (350 documentos × variantes)
- Validation: 15%
- Holdout test: 15% — solo se toca en Fase 5

---

## 🛠️ Stack Tecnológico

| Capa | Herramienta | Versión | Razón |
|---|---|---|---|
| Runtime | Python | 3.11 | Consistente con proyectos anteriores |
| Package mgmt | uv | latest | Consistente |
| CV preprocesamiento | OpenCV | 4.10+ | Mencionado explícitamente en cargo SICPA |
| Deep Learning | PyTorch | 2.3+ | Estándar industria para CV |
| Modelo base | EfficientNet-B0 (torchvision) | — | Mejor ratio accuracy/tamaño para fine-tuning |
| Augmentation | Albumentations | 1.4+ | Más rápido y flexible que torchvision transforms |
| Grad-CAM | pytorch-grad-cam | 1.5+ | Librería de referencia, soporta múltiples métodos |
| Métricas | scikit-learn | 1.5+ | Consistente |
| Experiment tracking | MLflow | 2.15+ | Consistente con Anomaly Detection LLM |
| API | FastAPI + Pydantic v2 | — | Consistente |
| Dashboard | Streamlit | 1.40+ | Consistente |
| PDF (V2) | ReportLab | 4.2+ | Generación de reportes profesionales |
| Testing | pytest + pytest-cov | — | Consistente |
| Linting | Ruff + mypy | — | Consistente |
| CI/CD | GitHub Actions | — | Consistente |
| Container | Docker | — | Consistente |

---

## 🎯 Métricas de Éxito

| Métrica | Umbral mínimo | Objetivo |
|---|---|---|
| Accuracy (holdout) | > 0.85 | > 0.92 |
| F1-score (clase falso) | > 0.82 | > 0.90 |
| AUC-ROC | > 0.90 | > 0.95 |
| Precision (falso) | > 0.80 | > 0.88 |
| Recall (falso) | > 0.80 | > 0.88 |
| Cobertura de tests | > 80% | > 85% |
| Latencia API p95 (sin GPU) | < 2s | < 1s |
| Tiempo generación Grad-CAM | < 500ms | < 200ms |

> **Nota sobre métricas:** en autenticación de documentos, Recall de falsos es crítico (no puedes dejar pasar un falso) pero Precision también importa (no puedes invalidar documentos legítimos). F1 como métrica principal es correcto. Documentar el trade-off en el notebook.

---

## 📋 Fases V1

---

### Fase 1 — Setup, Dataset y EDA Visual
**Objetivo:** Entender el dataset visualmente, construir el pipeline de generación de falsificaciones sintéticas, y configurar el proyecto base.

**Tareas:**
- Configurar proyecto: `uv`, `pyproject.toml`, `Makefile` (targets: `fix`, `test`, `run`), `.gitignore`, pre-commit config
- Descargar MIDV-500 con script reproducible (documentar en README)
- EDA visual en `notebooks/01_eda_dataset.ipynb`:
  - Distribución de tipos de documentos
  - Variabilidad de condiciones (fondo, iluminación, perspectiva)
  - Dimensiones de imágenes, canales, rango de valores
  - Ejemplos visuales de 20+ imágenes
- Implementar generador de falsificaciones en `src/data/augmentation.py`:
  - 4 tipos de perturbación documentados arriba
  - Cada perturbación tiene intensidad configurable (leve/media/fuerte)
  - Seed fijo para reproducibilidad
- Visualizar falsificaciones generadas vs originales — validar que son detectables visualmente

**Outputs:**
- Estructura de proyecto inicializada con CI básico verde
- `notebooks/01_eda_dataset.ipynb` con visualizaciones
- `src/data/augmentation.py` funcional con 4 tipos de perturbación
- `data/samples/` con 20 imágenes para tests

**Criterio de aceptación:** Dataset descargado, falsificaciones generadas visualmente validadas, CI verde con estructura vacía.

**Commit:** `feat: phase 1 - dataset EDA and synthetic forgery generation pipeline`

---

### Fase 2 — Pipeline de Preprocesamiento con OpenCV
**Objetivo:** Construir el pipeline de preprocesamiento robusto que normaliza imágenes antes de pasarlas al modelo — crítico para documentos fotografiados en condiciones variables.

**Pasos del pipeline (en orden):**

```
Imagen raw (cualquier tamaño, orientación, iluminación)
    ↓
1. Corrección de perspectiva (homografía si se detectan bordes del documento)
    ↓
2. Denoising (cv2.fastNlMeansDenoisingColored)
    ↓
3. Corrección de iluminación (CLAHE en canal L del espacio LAB)
    ↓
4. Resize a 224×224 (interpolación bicúbica)
    ↓
5. Normalización ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ↓
Tensor listo para EfficientNet
```

**Tareas:**
- Implementar `src/preprocessing/pipeline.py` con clase `DocumentPreprocessor`
- Cada paso es opcional y configurable (flags booleanos) para ablation study posterior
- Visualizar efecto de cada paso sobre 5 imágenes de ejemplo en notebook
- Medir tiempo de procesamiento por imagen (target: < 100ms en CPU)
- Tests unitarios para cada paso del pipeline con imágenes sintéticas pequeñas

**Outputs:**
- `src/preprocessing/pipeline.py` con `DocumentPreprocessor`
- `notebooks/02_preprocessing_pipeline.ipynb` con visualización paso a paso
- `tests/test_preprocessing.py`

**Criterio de aceptación:** Pipeline procesa imagen de cualquier tamaño → tensor 224×224 normalizado. Tests pasan. Tiempo < 100ms por imagen en CPU.

**Commit:** `feat: phase 2 - OpenCV preprocessing pipeline with CLAHE and perspective correction`

---

### Fase 3 — Fine-tuning EfficientNet-B0
**Objetivo:** Fine-tunear EfficientNet-B0 preentrenado en ImageNet para clasificación binaria auténtico/falso.

**Arquitectura:**

```python
# Backbone: EfficientNet-B0 preentrenado
backbone = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")

# Reemplazar classifier head
backbone.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 1),      # Salida binaria
    nn.Sigmoid()
)
```

**Estrategia de entrenamiento (dos fases):**

**Fase 3a — Feature extraction (5 epochs):**
- Congelar backbone, entrenar solo el classifier head
- LR = 1e-3, Adam optimizer
- Permite que el head se adapte antes de hacer fine-tuning completo

**Fase 3b — Fine-tuning completo (15 epochs):**
- Descongelar últimas 2 capas del backbone + classifier
- LR = 1e-4 (reducción ×10 para no destruir pesos preentrenados)
- CosineAnnealingLR scheduler
- Early stopping con patience=5 sobre validation F1

**Augmentation de entrenamiento (Albumentations):**
- HorizontalFlip (p=0.5)
- RandomBrightnessContrast (p=0.3)
- GaussNoise (p=0.2)
- RandomRotate90 (p=0.3)
- CoarseDropout (simula oclusiones, p=0.2)

**Tracking con MLflow:**
```python
with mlflow.start_run(run_name="efficientnet_b0_finetune"):
    mlflow.log_params({"backbone": "efficientnet_b0", "lr": 1e-4, ...})
    # Por epoch:
    mlflow.log_metrics({"train_loss": ..., "val_f1": ..., "val_auc": ...})
    mlflow.pytorch.log_model(model, "classifier")
```

**Tareas:**
- Implementar `src/models/classifier.py` con `DocumentClassifier`
- Implementar `src/models/trainer.py` con loop de entrenamiento
- Implementar `src/data/loader.py` con `DocumentDataset` (PyTorch Dataset)
- Entrenar modelo completo y guardar mejor checkpoint en `models/saved/`
- Notebook `03_model_training.ipynb` con curvas de loss y métricas por epoch

**Outputs:**
- `src/models/classifier.py`, `trainer.py`, `src/data/loader.py`
- `models/saved/efficientnet_b0_best.pt`
- MLflow run registrado
- `notebooks/03_model_training.ipynb` con curvas de entrenamiento

**Criterio de aceptación:** Modelo converge (validation F1 > 0.82 en al menos 1 epoch). Checkpoint guardado. MLflow run visible en `mlflow ui`.

**Commit:** `feat: phase 3 - EfficientNet-B0 fine-tuning with two-phase strategy and MLflow tracking`

---

### Fase 4 — Grad-CAM (Diferenciador Central)
**Objetivo:** Implementar Grad-CAM para visualizar qué zonas del documento activaron la clasificación — el diferenciador principal del proyecto.

**Cómo funciona Grad-CAM en este contexto:**
```
Imagen → EfficientNet → predicción (auténtico/falso)
                ↓
    Gradientes de la clase predicha
    respecto a la última capa convolucional
                ↓
    Mapa de calor 7×7 → upscale a 224×224
                ↓
    Superposición sobre imagen original (colormap jet)
                ↓
    Imagen con zonas "sospechosas" destacadas en rojo
```

**Implementación:**

```python
# Usando pytorch-grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

cam = GradCAM(model=classifier.backbone,
              target_layers=[classifier.backbone.features[-1]])

grayscale_cam = cam(input_tensor=img_tensor, targets=None)
visualization = show_cam_on_image(img_rgb, grayscale_cam[0], use_rgb=True)
```

**Casos documentados en notebook (mínimo 6 ejemplos):**
1. Documento auténtico → Grad-CAM apunta a zona de texto principal ✅
2. Falsificación por blur de texto → Grad-CAM apunta a zona blureada ✅
3. Falsificación por cambio de color en sello → apunta al sello ✅
4. Falsificación por splicing → apunta a zona copiada ✅
5. Falso positivo (auténtico clasificado como falso) → analizar qué activó
6. Falso negativo (falso no detectado) → analizar por qué falló

**Variantes de Grad-CAM a comparar:**
- GradCAM clásico
- GradCAM++ (mejor para objetos pequeños — relevante para sellos pequeños)
- EigenCAM (no necesita labels — útil para inferencia rápida)

**Outputs:**
- `src/explainability/gradcam.py` con `GradCAMExplainer`
- `src/explainability/visualizer.py` con `overlay_heatmap(img, cam) -> np.ndarray`
- `notebooks/04_gradcam_analysis.ipynb` con 6 casos documentados
- `reports/figures/gradcam_samples/` con imágenes exportadas

**Criterio de aceptación:** Grad-CAM genera heatmap coherente con el tipo de perturbación. Visualizaciones exportadas. Al menos 1 análisis de error documentado.

**Commit:** `feat: phase 4 - Grad-CAM explainability with GradCAM++ and EigenCAM comparison`

---

### Fase 5 — Evaluación en Holdout
**Objetivo:** Evaluación honesta del modelo en el holdout test que no se ha tocado hasta ahora.

**Métricas a reportar:**
- Accuracy, Precision, Recall, F1 (clase falso como positivo)
- AUC-ROC con curva
- Confusion matrix con valores absolutos y porcentajes
- Análisis por tipo de falsificación: ¿qué tipo detecta mejor/peor?
- Análisis de threshold: curva Precision-Recall, threshold óptimo

**Análisis adicional:**
- Top-10 errores (falsos positivos y negativos más confiados)
- Distribución de scores por clase
- Tiempo de inferencia: p50, p90, p95 en CPU

**Tareas:**
- Implementar `src/models/evaluator.py`
- Notebook `05_evaluation.ipynb` con todas las métricas y visualizaciones
- Exportar curvas ROC y PR a `reports/figures/`
- Documentar conclusiones: ¿qué tipo de falsificación es más difícil de detectar?

**Outputs:**
- `src/models/evaluator.py`
- `notebooks/05_evaluation.ipynb`
- Métricas finales documentadas en README

**Criterio de aceptación:** AUC-ROC > 0.90 en holdout. Análisis de errores documentado. Métricas en README actualizadas.

**Commit:** `feat: phase 5 - holdout evaluation with error analysis and threshold optimization`

---

### Fase 6 — FastAPI + Schemas
**Objetivo:** Exponer el sistema como API REST que recibe una imagen y retorna veredicto + heatmap Grad-CAM.

**Endpoints:**

```
POST /authenticate
  Input:  multipart/form-data con imagen (JPEG/PNG, max 10MB)
  Output: {
    document_id: str,
    verdict: "authentic" | "forged",
    confidence: float,          # probabilidad de la clase predicha
    forgery_probability: float, # siempre la prob de clase "forged"
    gradcam_image: str,         # base64 de imagen con heatmap superpuesto
    processing_time_ms: float,
    threshold_used: float
  }

POST /authenticate/batch
  Input:  List de imágenes (max 10)
  Output: List[AuthenticationResult]

GET  /health
  Output: { status: str, model_loaded: bool, model_version: str }

GET  /model/info
  Output: { backbone: str, threshold: float, metrics: dict }
```

**Tareas:**
- Implementar schemas Pydantic v2 (`AuthenticationRequest`, `AuthenticationResult`)
- Cargar modelo en `lifespan` context manager
- Grad-CAM se computa en cada inferencia y se retorna como base64
- Validación de imagen: formato, tamaño, canales
- Manejo de errores: imagen corrupta, tamaño inválido, formato no soportado

**Outputs:**
- `api/main.py`, `api/schemas.py`, `api/predictor.py`
- Swagger en `/docs` funcional con ejemplo de request

**Criterio de aceptación:** `POST /authenticate` recibe imagen y retorna veredicto + heatmap base64. Tests de API con `TestClient` como context manager pasan.

**Commit:** `feat: phase 6 - FastAPI with authentication endpoint and Grad-CAM response`

---

### Fase 7 — Dashboard Streamlit
**Objetivo:** Dashboard operacional que simula el flujo real de un inspector de documentos.

**Flujo del usuario:**
```
1. Inspector sube imagen del documento
2. Dashboard muestra: veredicto (AUTÉNTICO / FALSIFICADO) con color
3. Muestra confianza como gauge/barra
4. Muestra imagen original vs imagen con heatmap Grad-CAM side-by-side
5. Muestra qué zona fue más sospechosa (descripción textual)
6. Historial de la sesión: últimos N documentos verificados con resultados
```

**Secciones:**

**1. 🔍 Verificador de Documentos**
- Upload de imagen (drag & drop)
- Resultado prominente: ✅ AUTÉNTICO (verde) o ❌ FALSIFICADO (rojo)
- Confianza en porcentaje
- Dos columnas: imagen original | imagen + Grad-CAM heatmap
- Zona más sospechosa destacada

**2. 📊 Estadísticas de Sesión**
- Documentos verificados: N auténticos, M falsificados
- Distribución de confianza (histograma)
- Tiempo promedio de procesamiento

**3. 🧪 Modo Demo**
- Galería de ejemplos pre-cargados: 3 auténticos + 3 falsificados de distintos tipos
- Un clic para verificar cada ejemplo sin subir imagen

**Outputs:**
- `dashboard/app.py`
- Deployed en Streamlit Cloud

**Criterio de aceptación:** Upload funciona, veredicto + heatmap se muestran correctamente, modo demo operativo sin dependencias externas.

**Commit:** `feat: phase 7 - Streamlit dashboard with side-by-side Grad-CAM visualization`

---

### Fase 8 — Tests, CI/CD y Docker (Cierre V1)
**Objetivo:** Calidad de producción. CI verde, coverage > 80%, Docker funcional.

**Tests:**

| Archivo | Qué testea |
|---|---|
| `test_preprocessing.py` | Cada paso del pipeline OpenCV con imágenes sintéticas |
| `test_classifier.py` | Forward pass, output shape, carga de checkpoint |
| `test_gradcam.py` | Output shape del heatmap, valores en [0,1], superposición |
| `test_api.py` | `/authenticate` con imagen válida, imagen inválida, `/health` |

**Reglas de mocks en CI:**
- El modelo real (`.pt`) no se carga en CI — usar modelo dummy con misma arquitectura
- Imágenes de test: sintéticas generadas en fixture, nunca del dataset real
- `pytest --cov=src --cov=api --cov-report=term-missing`

**GitHub Actions:**
```yaml
# Secuencia: lint (Ruff) → type check (mypy) → tests → coverage badge
```

**Dockerfile:**
```dockerfile
# Multi-stage: builder (instala deps) + runtime (solo lo necesario)
# Expone puerto 8000 para FastAPI
# Incluye modelo serializado en la imagen
```

**README — secciones obligatorias:**
- Badges: CI, coverage, Python version, Docker
- Demo GIF del dashboard (subir imagen → veredicto → heatmap)
- 6 ejemplos de Grad-CAM con descripción de qué detectó
- Arquitectura del sistema (diagrama ASCII)
- Instrucciones de setup: `uv sync` → `make run`
- Instrucciones Docker: `docker build` → `docker run`
- Métricas finales: AUC-ROC, F1, accuracy en holdout

**Criterio de aceptación:** CI verde en main, coverage > 80%, `docker build` exitoso, README con métricas reales.

**Commit:** `feat: phase 8 - tests, CI/CD, Docker, README V1 complete`

---

## 📋 Fases V2

> **Prerequisito V2:** CI verde en main post-Fase 8. Todos los tests pasando. Dashboard deployado.

---

### Fase 9 — Reporte PDF Automático por Documento
**Objetivo:** Generar un reporte PDF profesional por cada documento verificado — simula el artefacto que un sistema real entregaría al operador o auditor.

**Contenido del reporte:**
```
┌─────────────────────────────────────────────────────┐
│  INFORME DE AUTENTICACIÓN DE DOCUMENTO              │
│  Sistema de Verificación CV — v2.0                  │
├─────────────────────────────────────────────────────┤
│  ID de verificación: DOC-20260609-001               │
│  Fecha y hora: 2026-06-09 14:32:15                  │
├─────────────────────────────────────────────────────┤
│  VEREDICTO: ❌ DOCUMENTO FALSIFICADO                │
│  Confianza: 94.7%                                   │
│  Probabilidad de falsificación: 0.947               │
├─────────────────────────────────────────────────────┤
│  [Imagen original]    [Imagen con Grad-CAM]         │
│                                                     │
├─────────────────────────────────────────────────────┤
│  ANÁLISIS DE ZONA SOSPECHOSA                        │
│  Área de mayor activación: zona superior derecha    │
│  Tipo de anomalía detectada: alteración cromática   │
├─────────────────────────────────────────────────────┤
│  PARÁMETROS DEL MODELO                              │
│  Backbone: EfficientNet-B0                          │
│  Versión: 2.0.1 | Threshold: 0.50                  │
│  AUC-ROC holdout: 0.963                             │
└─────────────────────────────────────────────────────┘
```

**Nuevo endpoint API:**
```
POST /authenticate/report
  Input:  imagen (igual que /authenticate)
  Output: PDF binario (Content-Type: application/pdf)
```

**Implementación:** ReportLab con template corporativo (logo placeholder, colores SICPA-inspired: azul/gris).

**Outputs:**
- `src/reporting/pdf_generator.py`
- `POST /authenticate/report` funcional
- Botón "Descargar Reporte PDF" en Streamlit
- Tests en `test_pdf_generator.py`

**Commit:** `feat: phase 9 - automated PDF report generation per document`

---

### Fase 10 — Comparativa de Modelos + Ablation Study
**Objetivo:** Documentar rigurosamente por qué EfficientNet-B0 es la elección correcta vs alternativas.

**Modelos a comparar:**

| Modelo | Parámetros | Velocidad CPU | AUC-ROC esperado |
|---|---|---|---|
| MobileNetV3-Small | 2.5M | muy rápida | ~0.88 |
| **EfficientNet-B0** | **5.3M** | **rápida** | **~0.95** ← seleccionado |
| ResNet-50 | 25.6M | media | ~0.93 |
| EfficientNet-B3 | 12M | media | ~0.96 |

**Ablation study del pipeline OpenCV:**
Entrenar el mismo modelo con/sin cada paso del preprocesamiento:
- Sin corrección de perspectiva → ΔF1 = ?
- Sin CLAHE → ΔF1 = ?
- Sin denoising → ΔF1 = ?

Esto justifica cada paso del pipeline con evidencia empírica.

**Todos los experimentos trackeados en MLflow.**

**Outputs:**
- `notebooks/06_error_analysis.ipynb` con comparativa completa
- Tabla de resultados en README actualizado
- MLflow experiment `model_comparison`

**Commit:** `feat: phase 10 - model comparison and OpenCV ablation study`

---

### Fase 11 — Mejoras de Robustez + Casos Edge
**Objetivo:** Hacer el sistema robusto a condiciones reales de uso.

**Mejoras implementadas:**

**11a. Detección de calidad de imagen antes de clasificar:**
```python
# Si la imagen es muy borrosa, muy oscura o muy pequeña → rechazar antes del modelo
quality_score = assess_image_quality(img)  # Laplacian variance + brightness
if quality_score < QUALITY_THRESHOLD:
    return {"verdict": "inconclusive", "reason": "image_quality_too_low"}
```

**11b. Ensemble de Grad-CAM methods:**
- Promediar GradCAM + GradCAM++ + EigenCAM → heatmap más estable
- Documentar mejora visual vs método individual

**11c. Threshold adaptativo por tipo de documento:**
- Documentos de alta seguridad (pasaportes): threshold más bajo (0.40) → más recall
- Documentos estándar: threshold 0.50
- Configurable via API parameter `security_level: "standard" | "high"`

**Nuevo parámetro en API:**
```
POST /authenticate
  Body: { security_level: "standard" | "high" = "standard" }
```

**Outputs:**
- `src/preprocessing/pipeline.py` extendido con quality assessment
- `src/explainability/gradcam.py` extendido con ensemble
- Tests actualizados
- Dashboard con selector de nivel de seguridad

**Commit:** `feat: phase 11 - image quality gating, Grad-CAM ensemble, adaptive threshold`

---

### Fase 12 — MLflow Registry + Tests V2 + README Final
**Objetivo:** Cerrar V2 con calidad de producción.

**MLflow Model Registry:**
- Registrar todos los modelos comparados en Fase 10
- EfficientNet-B0 promovido a stage `Production`
- Endpoint `/model/info` retorna versión desde MLflow Registry

**Tests V2 nuevos:**

| Archivo | Qué testea |
|---|---|
| `test_pdf_generator.py` | PDF generado, tiene las secciones correctas, tamaño > 0 |
| `test_api.py` (extendido) | `/authenticate/report`, `security_level` parameter, calidad baja |

**Coverage objetivo V2:** mantener > 82%

**README V2 — nuevas secciones:**
- Tabla comparativa de modelos (Fase 10)
- Ablation study results
- Ejemplos de reportes PDF (capturas)
- Sección de robustez: qué pasa con imágenes de baja calidad
- Diagrama de arquitectura V2 completo

**Commit:** `feat: phase 12 - MLflow registry, V2 tests complete, README V2 final`

---

## 🗓️ Estimación de Tiempo

### V1 (Fases 1-8)
| Fase | Descripción | Días |
|---|---|---|
| 1 | Setup + Dataset + EDA + Falsificaciones | 2.0 |
| 2 | Pipeline OpenCV | 1.5 |
| 3 | Fine-tuning EfficientNet-B0 | 3.0 |
| 4 | Grad-CAM ⭐ | 2.0 |
| 5 | Evaluación holdout | 1.0 |
| 6 | FastAPI | 1.0 |
| 7 | Dashboard Streamlit | 1.5 |
| 8 | Tests + CI/CD + Docker | 2.0 |
| **Total V1** | | **14 días** |

### V2 (Fases 9-12)
| Fase | Descripción | Días |
|---|---|---|
| 9 | Reporte PDF | 2.0 |
| 10 | Comparativa modelos + Ablation | 3.0 |
| 11 | Robustez + Edge cases | 2.5 |
| 12 | MLflow Registry + Tests + README | 2.0 |
| **Total V2** | | **9.5 días** |

> **Total: ~24 días** con 3-4h/día → 3-4 semanas cómodamente.

---

## 💼 Narrativa para Entrevista SICPA

**Pregunta:** *"¿Tienes experiencia con Computer Vision?"*
> "Construí un sistema de autenticación de documentos usando EfficientNet-B0 fine-tuneado sobre el dataset MIDV-500. El diferenciador técnico es Grad-CAM: el sistema no solo clasifica el documento como auténtico o falsificado — genera un mapa de calor que muestra al operador exactamente qué zona activó la detección. Eso es crítico en un contexto regulado, donde necesitas que el operador pueda auditar cada decisión. El sistema alcanzó AUC-ROC de X en holdout."

**Pregunta:** *"¿Cómo manejarías imágenes de mala calidad en producción?"*
> "Implementé una etapa de quality gating antes del modelo: si la imagen está muy borrosa, muy oscura o tiene resolución insuficiente, el sistema retorna 'inconclusivo' con el motivo, en lugar de dar una predicción poco confiable. Es preferible pedirle al operador que tome otra foto que dar un veredicto erróneo."

**Pregunta:** *"¿Por qué EfficientNet y no ResNet?"*
> "Comparé cuatro arquitecturas en el mismo dataset con el mismo pipeline: MobileNetV3, EfficientNet-B0, ResNet-50 y EfficientNet-B3. EfficientNet-B0 da el mejor ratio AUC-ROC / velocidad de inferencia en CPU — que es el constraint real en un sistema de inspección en punto de control donde no tienes GPU. El ablation study también mostró que la corrección de perspectiva con OpenCV agrega +X puntos de F1, lo que justifica el preprocesamiento adicional."

---

## 📌 Notas para Claude Code

- **Fase 3 requiere GPU o paciencia:** EfficientNet-B0 en CPU tarda ~2-3h de entrenamiento completo. Si no hay GPU disponible, usar Google Colab para la fase de entrenamiento y guardar el checkpoint `.pt`. El resto del proyecto corre perfectamente en CPU.
- **Las imágenes de test NUNCA deben ser del dataset real** — usar imágenes sintéticas generadas en fixtures (numpy arrays con shapes correctos). Eso garantiza que CI corra sin descargar el dataset.
- **Grad-CAM requiere que el modelo esté en modo eval:** `model.eval()` antes de cualquier inferencia. Si está en modo train, los resultados son incorrectos.
- **Base64 de imágenes en la API:** usar `base64.b64encode(buffer.getvalue()).decode('utf-8')` — incluir el prefijo `data:image/jpeg;base64,` para que Streamlit pueda renderizarlo con `st.image()`.
- **ReportLab en Fase 9:** instalar con `uv add reportlab`. Los PDFs se generan en memoria con `BytesIO` — no escribir a disco en la API.
- **MLflow:** usar `mlflow.set_tracking_uri("file:./mlflow")` en todos los notebooks para consistencia.
- **Modelo en Docker:** el checkpoint `.pt` debe incluirse en la imagen Docker. Agregar `COPY models/saved/ /app/models/saved/` en el Dockerfile. Si el archivo es > 100MB usar Git LFS.
- **Git LFS para el modelo:** si el `.pt` supera 50MB, configurar `git lfs track "*.pt"` antes del primer commit que incluya el modelo.
- Notebooks: cerrar en VS Code antes de editar desde terminal.
- `make fix` antes de cada commit.
- Branch por fase: `feature/phase-1-eda`, `feature/phase-2-preprocessing`, etc.
- Commits en inglés, conventional commits. Markdown de notebooks en español, código en inglés, labels de plots en español.
