# Estudio del Proyecto — Document Authentication CV

Documento de análisis: el **porqué**, el **cómo** y los **resultados** de un sistema
de autenticación de documentos basado en visión por computadora. Pensado para
entender la toma de decisiones técnica, no solo el código.

---

## 1. Resumen ejecutivo

Sistema end-to-end que clasifica documentos como **auténticos** o **falsificados**
usando EfficientNet-B0 + Grad-CAM, con foco en la autenticación de timbres
fiscales (caso SICPA). No es un notebook de Kaggle: es un proyecto de **MLOps
completo** — desde la generación de datos hasta el despliegue versionado.

| Dimensión | Valor |
|-----------|-------|
| Fases implementadas | 12 / 12 |
| Módulos de código | 25 archivos Python en `src/`, `api/`, `dashboard/` |
| Tests | 270 (cobertura 87%) |
| Notebooks de análisis | 8 |
| Superficie desplegable | API REST (FastAPI) + Dashboard (Streamlit) + Docker |
| CI/CD | GitHub Actions: lint + type-check + tests + docker-build |

**Honestidad metodológica**: el checkpoint actual está sin entrenar con datos
reales — el pipeline funciona de punta a punta, pero el veredicto del modelo no
es confiable hasta entrenar con un dataset real (ver §7). Esto es deliberado: el
valor demostrado es la **arquitectura de sistema**, no una métrica de accuracy.

---

## 2. El problema y el contexto

**Autenticar un documento físico fotografiado es un problema de visión fina.** La
diferencia entre un timbre fiscal real y uno falsificado puede estar en:

- micro-texturas (patrones de impresión, hologramas)
- consistencia de color bajo iluminación variable
- artefactos de manipulación (splicing, retoque, reimpresión)

Tres restricciones definieron el diseño:

1. **Las imágenes vienen del mundo real** — fotos de móvil con perspectiva,
   sombras, ruido, resolución variable. El modelo no puede asumir escaneos limpios.
2. **El error tiene costo asimétrico** — un falso negativo (dejar pasar una
   falsificación) y un falso positivo (rechazar un documento legítimo) tienen
   consecuencias distintas. Esto exige control del umbral de decisión y
   explicabilidad, no solo una probabilidad.
3. **Necesita ser auditable** — en un contexto fiscal/legal, "el modelo dijo que
   es falso" no basta. Hay que mostrar *dónde* miró el modelo (Grad-CAM) y generar
   evidencia (informe PDF).

---

## 3. Decisiones de stack — el porqué de cada elección

Cada elección se tomó por una razón concreta, no por defecto.

### 3.1 Modelo: EfficientNet-B0 (no ResNet-50, no un ViT)

| Alternativa | Por qué NO se eligió como base |
|-------------|-------------------------------|
| ResNet-50 | 25M params, más pesado, sin mejor accuracy en visión fina de este tamaño |
| Vision Transformer | Necesita muchos más datos para entrenar desde cero; overkill para clasificación binaria |
| MobileNetV3 | Excelente latencia pero menor capacidad de representación |

**EfficientNet-B0** da el mejor compromiso accuracy/eficiencia para tareas de
textura fina: 5.3M parámetros, entrada 224×224, y un *compound scaling* que lo
hace fuerte en detalles pequeños (justo lo que importa en sellos). La **fase 10**
no asumió esto: construyó un `ModelComparator` que evalúa EfficientNet-B0 vs
ResNet-18 vs MobileNetV3-Small sobre el mismo split, para que la elección quede
respaldada empíricamente y sea reproducible.

**Estrategia de entrenamiento en dos fases** (`src/models/trainer.py`):
- Fase A: congelar el backbone, entrenar solo la cabeza binaria (aprendizaje rápido)
- Fase B: descongelar las últimas capas + cabeza con LR menor (ajuste fino)

Esto evita destruir las features pre-entrenadas de ImageNet en las primeras épocas
— un error clásico que arruina el transfer learning.

### 3.2 Explicabilidad: Grad-CAM (no SHAP, no LIME)

Para imágenes, **Grad-CAM es el estándar**: produce un mapa de calor sobre la
imagen mostrando qué regiones activaron la decisión. SHAP/LIME son caros y menos
interpretables visualmente para CNNs. Se implementaron tres variantes
(`src/explainability/gradcam.py`):
- **GradCAM** — clásico
- **GradCAM++** — mejor localización de objetos pequeños/múltiples (ideal para sellos)
- **EigenCAM** — no necesita labels, rápido para batch

Más un modo *ensemble* que promedia los tres para estabilidad.

### 3.3 Preprocesamiento: OpenCV clásico (no solo dejárselo a la red)

Decisión deliberada: **normalizar la entrada antes de la red** en vez de confiar
en que el modelo aprenda invarianzas. `src/preprocessing/pipeline.py` aplica:

1. Corrección de perspectiva (Canny + homografía) — endereza fotos inclinadas
2. Denoising (fastNlMeansDenoisingColored) — reduce ruido de sensor
3. CLAHE en canal L de LAB — normaliza contraste bajo iluminación dispareja
4. Resize bicúbico + normalización ImageNet

**Por qué**: con datasets pequeños, reducir la varianza de entrada vale más que
pedirle a la red que la modele. Cada paso es *togglable* para poder medir su aporte.

### 3.4 Servicio: FastAPI + Streamlit (separados)

- **FastAPI** para la API REST: tipado con Pydantic v2, async, OpenAPI automático,
  el estándar de facto para servir modelos en Python.
- **Streamlit** para el dashboard: permite construir una UI funcional sin frontend
  dedicado — correcto para una herramienta interna / demo de portafolio.

Se mantuvieron **desacoplados**: el dashboard usa el `DocumentPredictor`
directamente (no llama a la API por HTTP). Esto simplifica el despliegue y evita
una dependencia de red innecesaria entre dos procesos del mismo sistema.

### 3.5 Tracking y registro: MLflow (no Weights & Biases)

**MLflow** con backend SQLite: open-source, self-hosted, sin dependencia de un
servicio externo (importante para datos sensibles). Cubre las dos necesidades:
- **Tracking** de experimentos (métricas, params por época) — fase 4
- **Model Registry** con versionado y aliases de despliegue — fase 12

### 3.6 Tooling: uv + Ruff + mypy + pytest

- **uv** como gestor de paquetes: resolución y sync mucho más rápidos que pip/poetry,
  con lockfile reproducible.
- **Ruff** para lint+format: un solo binario, ultrarrápido.
- **mypy** para type-checking estático: atrapa errores antes de runtime.
- **pytest + cobertura**: 270 tests como red de seguridad.

---

## 4. Cómo está construido — las 12 fases

El proyecto se construyó incrementalmente, **una rama y un PR por fase**, cada uno
con CI verde antes de mergear. Esto no es burocracia: mantiene `main` siempre
desplegable y hace el historial auditable.

| Fase | Módulo principal | Qué resuelve |
|------|-----------------|--------------|
| 1 | `src/data/augmentation.py` | Genera falsificaciones sintéticas (4 tipos × 3 severidades) |
| 2 | `src/preprocessing/pipeline.py` | Normaliza la entrada del mundo real |
| 3 | `src/data/loader.py` | Dataset + DataLoaders con Albumentations |
| 4 | `src/models/trainer.py` | Entrenamiento 2 fases + MLflow |
| 5 | `src/models/evaluator.py` | ROC/PR, threshold óptimo, calibración |
| 6 | `api/` | API REST (4 endpoints) |
| 7 | `dashboard/app.py` | Dashboard en español, multi-formato |
| 8 | `Dockerfile`, `.github/` | CI/CD + Docker multi-stage |
| 9 | `src/reporting/pdf_report.py` | Informes PDF auditables |
| 10 | `src/models/comparator.py` | Comparación de backbones (ablation) |
| 11 | `src/preprocessing/quality.py` | Quality gating + análisis de robustez |
| 12 | `src/models/registry.py` | Model Registry (staging/production) |

**Flujo de datos en inferencia**:
```
imagen → decode → quality gate → preprocesamiento → EfficientNet-B0
       → probabilidad → [Grad-CAM] → resultado + heatmap + PDF
```

---

## 5. Insights relevantes — lo que se aprendió en el camino

Esta sección es la más valiosa: los problemas reales que aparecieron y cómo se
resolvieron. Aquí está el aprendizaje de ingeniería, no solo el "qué".

### 5.1 El quality gate previene falsos positivos silenciosos

**Insight**: un clasificador binario *siempre* devuelve una respuesta, incluso
ante una imagen basura (borrosa, oscura, diminuta). Sin un filtro previo, el
sistema da veredictos confiados sobre entradas fuera de su competencia.

La **fase 11** añadió `ImageQualityAssessor`: métricas no-referenciadas (nitidez
vía varianza del Laplaciano, brillo, contraste, recorte de tonos, resolución) que
rechazan la imagen *antes* de inferir (`label='rejected'`). El análisis de
robustez (notebook 07) midió a qué severidad de cada degradación el gate empieza
a rechazar — validando que reacciona en la dirección correcta. Conclusión: el gate
detecta bien desenfoque y baja resolución; la compresión JPEG y el ruido son más
sutiles para métricas clásicas y degradan al modelo más que a las métricas.

### 5.2 torch arrastra ~3GB de CUDA inútil en un contenedor CPU

**Insight de despliegue**: el contenedor de inferencia corre en CPU, pero
`torch` por defecto instala todo el stack CUDA transitivamente (`nvidia-cublas`
403MB, `nvidia-cudnn` 349MB, `triton` 192MB...). Con `cache-to mode=max`, el
export del cache de GitHub Actions superaba el límite de 10GB y **rompía el build**.

**Solución**: configurar uv para resolver `torch`/`torchvision` desde el índice
CPU de PyTorch **solo en Linux** (CI/Docker), manteniendo MPS en macOS para
desarrollo local:
```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
```
Esto eliminó todos los `nvidia-*`, redujo la imagen ~3GB y aceleró el build.
**Lección**: el default de una dependencia rara vez es el correcto para producción.

### 5.3 La carrera de cache en CI: push + pull_request

**Insight de CI**: el workflow disparaba en `push` (a feature branches) **y** en
`pull_request`, generando dos builds simultáneos sobre el mismo commit que se
peleaban por la misma clave de cache GHA → `error writing layer blob: not_found`
intermitente.

**Solución**: `push` solo corre en `main`; las feature branches obtienen CI vía
el evento `pull_request` (un solo run). Más un `concurrency` group que cancela
runs superados. **Lección**: el fallo no estaba en el código sino en la
*orquestación* — diagnosticar la infraestructura es parte del trabajo.

### 5.4 MLflow 3.x rompió el registro por artefacto

**Insight de mantenimiento**: `mlflow.register_model("runs:/.../model", name)`
funcionaba en MLflow 2.x con artefactos `.pt` planos, pero en 3.x exige un
"LoggedModel" formal y falla con *"Unable to find a logged_model"*.

**Solución**: usar `MlflowClient.create_model_version(source=...)` apuntando
directo al artefacto, y aliases (`staging`/`production`) en vez de las stage
transitions deprecadas. **Lección**: las APIs evolucionan; los tests de
integración (no solo unitarios) atrapan estas rupturas.

### 5.5 El bug del checkpoint que nunca se guardaba

**Insight de entrenamiento**: el `Trainer` inicializaba `_best_val_f1 = 0.0`, pero
un modelo aleatorio puede devolver `val_f1 = 0.0`, y `0.0 > 0.0` es `False` → el
checkpoint **nunca se guardaba**. Corregido a `-1.0`. **Lección**: los valores
centinela importan; un `0.0` aparentemente inocuo era un bug de borde.

### 5.6 Tests de UI con estado aislado

**Insight**: `AppTest.from_file` de Streamlit re-ejecuta el script en un contexto
aislado, así que un `patch()` aplicado en el proceso de test no afecta a la app.
El test de "modelo cargado" tuvo que reescribirse para verificar el estado *real*
del sidebar (`at.sidebar.success` / `at.sidebar.error`) en vez de depender del
mock. **Lección**: testear frameworks con ejecución propia requiere entender su
modelo de aislamiento.

---

## 6. Resultados

**Lo que funciona y está verificado:**

- ✅ Pipeline end-to-end operativo: subida (JPEG/PNG/TIFF/PDF multipágina) →
  inferencia → Grad-CAM → quality gate → informe PDF
- ✅ API REST con 5 endpoints, validación Pydantic, manejo de errores
- ✅ Dashboard interactivo en español con 3 pestañas
- ✅ 270 tests, 87% de cobertura, CI verde
- ✅ Docker multi-stage desplegable (imagen liviana, non-root, healthchecks)
- ✅ Comparación de modelos reproducible con tracking en MLflow
- ✅ Model Registry con versionado, promoción por stages y rollback atómico
- ✅ Robustez caracterizada frente a 5 tipos de degradación

**Métricas de calidad de ingeniería** (no de modelo):

| Métrica | Valor |
|---------|-------|
| Cobertura de tests | 87% |
| Módulos con type hints | 24/24 pasan mypy |
| Tiempo de inferencia (CPU) | ~330ms por imagen |
| Tamaño del checkpoint | 17.6 MB |

---

## 7. Limitaciones honestas y próximos pasos

**La limitación principal**: el modelo no está entrenado con datos reales. El
checkpoint produce probabilidades ~0.5 (azar), por lo que el veredicto
auténtico/falsificado **no es confiable todavía**. Esto se demostró al subir una
cédula real y obtener "Falsificado" — no porque la cédula sea falsa, sino porque
el modelo no discrimina aún.

Esto es **deliberado y honesto**: el proyecto demuestra la *arquitectura de
sistema MLOps*, que es independiente de la calidad del modelo. El siguiente paso
es puramente de datos+cómputo:

1. **Entrenar con datos reales**: descargar MIDV-500 (script de fase 1), generar
   falsificaciones sintéticas con el `ForgeryGenerator`, correr el `Trainer`.
2. **Calibrar el umbral** según el costo asimétrico real (fase 5 ya provee
   `find_optimal_threshold`).
3. **Registrar el campeón** en el Model Registry y servirlo con
   `MODEL_REGISTRY_ALIAS=production`.
4. **Validar con datos externos** (out-of-distribution) antes de cualquier uso real.

**Otras mejoras posibles**:
- Detección de artefactos de compresión por DCT (la §5.1 mostró que el quality
  gate clásico no los captura bien)
- Cuantización del modelo para inferencia más rápida en edge (MobileNetV3 ya
  evaluado en fase 10)
- Autenticación multi-clase por tipo de documento

---

## 8. Conclusión

El valor de este proyecto no está en una métrica de accuracy, sino en haber
construido un **sistema de ML completo, mantenible y desplegable**, con las
decisiones de ingeniería justificadas y documentadas. Cada elección de stack
responde a una restricción concreta del problema; cada problema encontrado en el
camino (CUDA, cache de CI, MLflow 3.x, bugs de borde) se diagnosticó y resolvió de
raíz, no con parches. El resultado es una base sobre la que el entrenamiento real
es el último paso, no un rediseño.
