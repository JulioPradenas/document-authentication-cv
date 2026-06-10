"""Streamlit dashboard — Autenticación de Documentos.

Ejecutar:
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

import fitz  # noqa: E402  (pymupdf)
import numpy as np
import streamlit as st
from PIL import Image

# Ensure src/ is importable regardless of working directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.predictor import DocumentPredictor
from src.reporting.pdf_report import PDFReportGenerator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT = _ROOT / "models" / "saved" / "efficientnet_b0_best.pt"
DEVICE = "cpu"
SAMPLE_DIR = _ROOT / "data" / "samples"

st.set_page_config(
    page_title="Autenticación de Documentos",
    page_icon=":mag:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {
        "history": [],
        "predictor": None,
        "model_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Cargando modelo…")
def load_predictor() -> DocumentPredictor | None:
    if not CHECKPOINT.exists():
        return None
    return DocumentPredictor(checkpoint=CHECKPOINT, device=DEVICE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64_str)))


def load_uploaded_image(uploaded_file) -> tuple[list[Image.Image], str]:
    """Convierte cualquier archivo subido en una lista de imágenes RGB."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".pdf"):
        doc = fitz.open(stream=raw, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
        return pages, f"PDF ({len(pages)} página{'s' if len(pages) > 1 else ''})"

    if name.endswith((".tif", ".tiff")):
        img = Image.open(BytesIO(raw))
        pages = []
        try:
            while True:
                pages.append(img.copy().convert("RGB"))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        return pages, f"TIFF ({len(pages)} página{'s' if len(pages) > 1 else ''})"

    img = Image.open(BytesIO(raw)).convert("RGB")
    return [img], name.split(".")[-1].upper()


def _etiqueta_color(label: str) -> str:
    return "🟢" if label == "authentic" else "🔴"


def _etiqueta_es(label: str) -> str:
    return "Auténtico" if label == "authentic" else "Falsificado"


def _render_result_card(result: dict, image: Image.Image) -> None:
    label = result["label"]
    prob = result["probability"]
    color = "#2ecc71" if label == "authentic" else "#e74c3c"
    etiqueta = _etiqueta_es(label)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Imagen analizada", use_container_width=True)

    with col2:
        st.markdown(
            f"""
            <div style="border-left: 4px solid {color}; padding: 12px 16px;
                        background: #f8f9fa; border-radius: 4px;">
                <h3 style="color:{color}; margin:0;">
                    {_etiqueta_color(label)} {etiqueta}
                </h3>
                <p style="font-size: 1.1rem; margin: 8px 0 0 0;">
                    P(falsificado) = <strong>{prob:.4f}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric("Confianza", f"{prob:.1%}")
        st.caption(
            f"Umbral: {result['threshold']:.2f}  |  Latencia: {result['inference_ms']:.1f} ms"
        )

        if result.get("gradcam_b64"):
            st.image(
                b64_to_pil(result["gradcam_b64"]),
                caption="Mapa de activación Grad-CAM",
                use_container_width=True,
            )
            if result.get("most_activated_region"):
                r = result["most_activated_region"]
                st.caption(
                    f"Región más activa — centro: ({r['cx']:.0f}, {r['cy']:.0f})  "
                    f"activación media: {r['mean_activation']:.3f}"
                )

        pdf_bytes = PDFReportGenerator().generate(
            result=result,
            image_b64=pil_to_b64(image, fmt="PNG"),
            model_info=st.session_state.get("model_info"),
            filename=result.get("_filename"),
        )
        st.download_button(
            "Descargar informe PDF",
            data=pdf_bytes,
            file_name="informe_autenticacion.pdf",
            mime="application/pdf",
            key=f"pdf_{len(st.session_state.history)}_{id(result)}",
            use_container_width=True,
        )


def _render_stats() -> None:
    history = st.session_state.history
    if not history:
        st.info("No hay documentos analizados en esta sesión.")
        return

    n_total = len(history)
    n_forged = sum(1 for h in history if h["label"] == "forged")
    n_auth = n_total - n_forged
    avg_prob = float(np.mean([h["probability"] for h in history]))
    avg_ms = float(np.mean([h["inference_ms"] for h in history]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total analizados", n_total)
    c2.metric("Auténticos", n_auth)
    c3.metric("Falsificados", n_forged)
    c4.metric("P(falsificado) media", f"{avg_prob:.3f}")

    st.caption(f"Latencia media: {avg_ms:.1f} ms/imagen")

    rows = [
        {
            "Imagen": f"#{i + 1}",
            "Resultado": _etiqueta_es(h["label"]),
            "P(falsificado)": f"{h['probability']:.4f}",
            "Latencia (ms)": f"{h['inference_ms']:.1f}",
        }
        for i, h in enumerate(reversed(history))
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Autenticación de Documentos")
st.sidebar.caption("EfficientNet-B0 + Grad-CAM")
st.sidebar.divider()

predictor = load_predictor()
if predictor is None:
    st.sidebar.error(f"Checkpoint no encontrado:\n`{CHECKPOINT}`")
else:
    st.sidebar.success("Modelo cargado")
    info = predictor.model_info()
    st.session_state.model_info = info
    st.sidebar.caption(
        f"Arquitectura: {info['architecture']}  \n"
        f"Parámetros: {info['total_params']:,}  \n"
        f"Dispositivo: {info['device'].upper()}"
    )

st.sidebar.divider()
threshold = st.sidebar.slider(
    "Umbral de decisión",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="P(falsificado) ≥ umbral → clasificado como falsificado",
)
show_gradcam = st.sidebar.toggle("Mostrar mapa Grad-CAM", value=True)
gradcam_method = st.sidebar.selectbox(
    "Método Grad-CAM",
    options=["gradcam++", "gradcam", "eigencam", "ensemble"],
    disabled=not show_gradcam,
)

st.sidebar.divider()
if st.sidebar.button("Limpiar historial", use_container_width=True):
    st.session_state.history = []
    st.rerun()

# ---------------------------------------------------------------------------
# Pestañas principales
# ---------------------------------------------------------------------------

tab_verify, tab_demo, tab_stats = st.tabs(
    ["Verificador de Documentos", "Modo Demo", "Estadísticas de Sesión"]
)

# ------- Pestaña 1: Verificador -------

with tab_verify:
    st.header("Sube un documento para verificar")

    uploaded = st.file_uploader(
        "JPEG, PNG, TIFF o PDF — cualquier resolución",
        type=["jpg", "jpeg", "png", "tif", "tiff", "pdf"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        if predictor is None:
            st.error("Modelo no cargado — no se puede ejecutar la inferencia.")
        else:
            pages, fmt_label = load_uploaded_image(uploaded)
            st.caption(f"Formato detectado: **{fmt_label}** · {len(pages)} imagen(es) a analizar")

            if len(pages) > 1:
                page_idx = st.slider("Página a analizar", 1, len(pages), 1) - 1
                pages_to_run = [pages[page_idx]]
            else:
                pages_to_run = pages

            for image in pages_to_run:
                with st.spinner("Ejecutando inferencia…"):
                    b64 = pil_to_b64(image)
                    result = predictor.predict(
                        image_b64=b64,
                        threshold=threshold,
                        return_gradcam=show_gradcam,
                        gradcam_method=gradcam_method,
                    )
                st.session_state.history.append(result)
                _render_result_card(result, image)

# ------- Pestaña 2: Demo -------

with tab_demo:
    st.header("Demo — muestras sintéticas")
    st.caption(
        "Ejecuta inferencia sobre las muestras generadas en `data/samples/`. "
        "Incluye documentos auténticos y con 4 tipos de falsificación sintética."
    )

    sample_images = sorted(SAMPLE_DIR.glob("*.jpg")) + sorted(SAMPLE_DIR.glob("*.png"))

    if not sample_images:
        st.warning(f"No se encontraron imágenes en `{SAMPLE_DIR}`.")
    elif predictor is None:
        st.error("Modelo no cargado — no se puede ejecutar el demo.")
    else:
        selected = st.multiselect(
            "Selecciona las muestras a analizar",
            options=[p.name for p in sample_images],
            default=[p.name for p in sample_images[: min(3, len(sample_images))]],
        )

        if st.button("Ejecutar demo", type="primary", use_container_width=True):
            paths = [SAMPLE_DIR / name for name in selected]
            progress = st.progress(0)

            for i, path in enumerate(paths):
                image = Image.open(path).convert("RGB")
                b64 = pil_to_b64(image)
                result = predictor.predict(
                    image_b64=b64,
                    threshold=threshold,
                    return_gradcam=show_gradcam,
                    gradcam_method=gradcam_method,
                )
                result["_filename"] = path.name
                st.session_state.history.append(result)
                progress.progress((i + 1) / len(paths))

                with st.expander(
                    f"{_etiqueta_color(result['label'])} {path.name} — "
                    f"P(falsificado)={result['probability']:.4f}",
                    expanded=True,
                ):
                    _render_result_card(result, image)

            progress.empty()

# ------- Pestaña 3: Estadísticas -------

with tab_stats:
    st.header("Estadísticas de sesión")
    _render_stats()
