"""Streamlit dashboard — Document Authentication.

Run:
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT = _ROOT / "models" / "saved" / "efficientnet_b0_best.pt"
DEVICE = "cpu"
SAMPLE_DIR = _ROOT / "data" / "samples"

st.set_page_config(
    page_title="Document Authentication",
    page_icon=":mag:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {
        "history": [],  # list[dict] — one per analyzed image
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


@st.cache_resource(show_spinner="Loading model…")
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
    """Convert any supported upload to a list of RGB PIL images.

    Returns (pages, format_label) where pages has one entry for images/TIFF
    and one entry per page for PDFs.
    """
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
        label = f"TIFF ({len(pages)} página{'s' if len(pages) > 1 else ''})"
        return pages, label

    # JPEG / PNG
    img = Image.open(BytesIO(raw)).convert("RGB")
    return [img], name.split(".")[-1].upper()


def _label_color(label: str) -> str:
    return "🟢" if label == "authentic" else "🔴"


def _render_result_card(result: dict, image: Image.Image) -> None:
    label = result["label"]
    prob = result["probability"]
    color = "#2ecc71" if label == "authentic" else "#e74c3c"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Input image", use_container_width=True)

    with col2:
        st.markdown(
            f"""
            <div style="border-left: 4px solid {color}; padding: 12px 16px;
                        background: #f8f9fa; border-radius: 4px;">
                <h3 style="color:{color}; margin:0;">
                    {_label_color(label)} {label.capitalize()}
                </h3>
                <p style="font-size: 1.1rem; margin: 8px 0 0 0;">
                    P(forged) = <strong>{prob:.4f}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric("Confidence", f"{prob:.1%}", delta=None)
        st.caption(
            f"Threshold: {result['threshold']:.2f}  |  Latency: {result['inference_ms']:.1f} ms"
        )

        if result.get("gradcam_b64"):
            st.image(
                b64_to_pil(result["gradcam_b64"]),
                caption="Grad-CAM overlay",
                use_container_width=True,
            )
            if result.get("most_activated_region"):
                r = result["most_activated_region"]
                st.caption(
                    f"Most activated region — center: ({r['cx']:.0f}, {r['cy']:.0f})  "
                    f"mean activation: {r['mean_activation']:.3f}"
                )


def _render_stats() -> None:
    history = st.session_state.history
    if not history:
        st.info("No documents analyzed yet.")
        return

    n_total = len(history)
    n_forged = sum(1 for h in history if h["label"] == "forged")
    n_auth = n_total - n_forged
    avg_prob = float(np.mean([h["probability"] for h in history]))
    avg_ms = float(np.mean([h["inference_ms"] for h in history]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total analyzed", n_total)
    c2.metric("Authentic", n_auth)
    c3.metric("Forged", n_forged)
    c4.metric("Avg P(forged)", f"{avg_prob:.3f}")

    st.caption(f"Avg latency: {avg_ms:.1f} ms/image")

    # History table
    rows = [
        {
            "Image": f"#{i + 1}",
            "Label": h["label"],
            "P(forged)": f"{h['probability']:.4f}",
            "Latency (ms)": f"{h['inference_ms']:.1f}",
        }
        for i, h in enumerate(reversed(history))
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Document Authentication")
st.sidebar.caption("EfficientNet-B0 + Grad-CAM")
st.sidebar.divider()

predictor = load_predictor()
if predictor is None:
    st.sidebar.error(f"Checkpoint not found:\n`{CHECKPOINT}`")
else:
    st.sidebar.success("Model loaded")
    info = predictor.model_info()
    st.sidebar.caption(
        f"Architecture: {info['architecture']}  \n"
        f"Params: {info['total_params']:,}  \n"
        f"Device: {info['device'].upper()}"
    )

st.sidebar.divider()
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="P(forged) ≥ threshold → classified as forged",
)
show_gradcam = st.sidebar.toggle("Show Grad-CAM overlay", value=True)
gradcam_method = st.sidebar.selectbox(
    "Grad-CAM method",
    options=["gradcam++", "gradcam", "eigencam", "ensemble"],
    disabled=not show_gradcam,
)

st.sidebar.divider()
if st.sidebar.button("Clear history", use_container_width=True):
    st.session_state.history = []
    st.rerun()

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_verify, tab_demo, tab_stats = st.tabs(["Document Verifier", "Demo Mode", "Session Stats"])

# ------- Tab 1: Document Verifier -------

with tab_verify:
    st.header("Upload a document image")

    uploaded = st.file_uploader(
        "JPEG, PNG, TIFF o PDF — cualquier resolución",
        type=["jpg", "jpeg", "png", "tif", "tiff", "pdf"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        if predictor is None:
            st.error("Model not loaded — cannot run inference.")
        else:
            pages, fmt_label = load_uploaded_image(uploaded)
            st.caption(f"Formato detectado: **{fmt_label}** · {len(pages)} imagen(es) a analizar")

            if len(pages) > 1:
                page_idx = st.slider("Página a analizar", 1, len(pages), 1) - 1
                pages_to_run = [pages[page_idx]]
            else:
                pages_to_run = pages

            for image in pages_to_run:
                with st.spinner("Running inference…"):
                    b64 = pil_to_b64(image)
                    result = predictor.predict(
                        image_b64=b64,
                        threshold=threshold,
                        return_gradcam=show_gradcam,
                        gradcam_method=gradcam_method,
                    )
                st.session_state.history.append(result)
                _render_result_card(result, image)

# ------- Tab 2: Demo Mode -------

with tab_demo:
    st.header("Demo — synthetic samples")
    st.caption("Run inference on the pre-generated synthetic samples in `data/samples/`.")

    sample_images = sorted(SAMPLE_DIR.glob("*.jpg")) + sorted(SAMPLE_DIR.glob("*.png"))

    if not sample_images:
        st.warning(f"No sample images found in `{SAMPLE_DIR}`.")
    elif predictor is None:
        st.error("Model not loaded — cannot run demo.")
    else:
        cols_per_row = 3
        selected = st.multiselect(
            "Select samples to analyze",
            options=[p.name for p in sample_images],
            default=[p.name for p in sample_images[: min(3, len(sample_images))]],
        )

        if st.button("Run demo", type="primary", use_container_width=True):
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
                    f"{_label_color(result['label'])} {path.name} — "
                    f"P(forged)={result['probability']:.4f}",
                    expanded=True,
                ):
                    _render_result_card(result, image)

            progress.empty()

# ------- Tab 3: Session Stats -------

with tab_stats:
    st.header("Session statistics")
    _render_stats()
