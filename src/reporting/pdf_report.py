"""PDF report generation for document authentication results.

PDFReportGenerator builds a one-page report (A4) with verdict, probability,
analyzed image, optional Grad-CAM overlay and model metadata.

Usage:
    generator = PDFReportGenerator()
    pdf_bytes = generator.generate(result, image_b64, model_info)
"""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

GREEN = colors.HexColor("#2ecc71")
RED = colors.HexColor("#e74c3c")
GRAY = colors.HexColor("#7f8c8d")


def _b64_to_flowable(image_b64: str, max_width: float, max_height: float) -> Image:
    """Convert a base64 image into a reportlab Image flowable, preserving aspect ratio."""
    raw = base64.b64decode(image_b64)
    pil = PILImage.open(BytesIO(raw))
    width_px, height_px = pil.size
    scale = min(max_width / width_px, max_height / height_px)
    return Image(BytesIO(raw), width=width_px * scale, height=height_px * scale)


class PDFReportGenerator:
    """Builds authentication reports as PDF bytes (A4, one page per document)."""

    def __init__(self) -> None:
        styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            "ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=4
        )
        self.subtitle_style = ParagraphStyle(
            "ReportSubtitle", parent=styles["Normal"], fontSize=9, textColor=GRAY
        )
        self.heading_style = ParagraphStyle(
            "SectionHeading", parent=styles["Heading2"], fontSize=12, spaceBefore=10
        )
        self.body_style = styles["Normal"]

    def generate(
        self,
        result: dict[str, Any],
        image_b64: str,
        model_info: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> bytes:
        """Render the report and return PDF bytes.

        Args:
            result: Output of DocumentPredictor.predict (label, probability,
                threshold, inference_ms, optionally gradcam_b64 / most_activated_region).
            image_b64: Base64 of the analyzed image (original).
            model_info: Optional output of DocumentPredictor.model_info.
            filename: Optional original filename shown in the metadata table.
        """
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            title="Informe de Autenticación de Documento",
        )

        elements: list[Any] = [
            Paragraph("Informe de Autenticación de Documento", self.title_style),
            Paragraph(
                f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                self.subtitle_style,
            ),
            Spacer(1, 0.5 * cm),
        ]

        elements.extend(self._verdict_block(result))
        elements.extend(self._images_block(result, image_b64))
        elements.extend(self._details_block(result, model_info, filename))

        elements.append(Spacer(1, 0.8 * cm))
        elements.append(
            Paragraph(
                "Este informe fue generado automáticamente por un modelo de clasificación "
                "binaria (EfficientNet-B0). El resultado es una estimación probabilística "
                "y no constituye un peritaje documental.",
                self.subtitle_style,
            )
        )

        doc.build(elements)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def _verdict_block(self, result: dict[str, Any]) -> list[Any]:
        is_authentic = result["label"] == "authentic"
        verdict = "AUTÉNTICO" if is_authentic else "FALSIFICADO"
        color = GREEN if is_authentic else RED

        table = Table(
            [
                [verdict],
                [
                    f"P(falsificado) = {result['probability']:.4f}   |   Umbral = {result['threshold']:.2f}"
                ],
            ],
            colWidths=[17 * cm],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), color),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 16),
                    ("FONTSIZE", (0, 1), (-1, 1), 10),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                    ("TOPPADDING", (0, 1), (-1, 1), 6),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
                    ("BOX", (0, 0), (-1, -1), 0.5, color),
                ]
            )
        )
        return [table, Spacer(1, 0.5 * cm)]

    def _images_block(self, result: dict[str, Any], image_b64: str) -> list[Any]:
        max_w, max_h = 8 * cm, 8 * cm
        original = _b64_to_flowable(image_b64, max_w, max_h)

        if result.get("gradcam_b64"):
            gradcam = _b64_to_flowable(result["gradcam_b64"], max_w, max_h)
            row = Table(
                [
                    [original, gradcam],
                    [
                        Paragraph("Documento analizado", self.subtitle_style),
                        Paragraph("Mapa de activación Grad-CAM", self.subtitle_style),
                    ],
                ],
                colWidths=[8.5 * cm, 8.5 * cm],
            )
        else:
            row = Table(
                [[original], [Paragraph("Documento analizado", self.subtitle_style)]],
                colWidths=[17 * cm],
            )

        row.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return [row, Spacer(1, 0.3 * cm)]

    def _details_block(
        self,
        result: dict[str, Any],
        model_info: dict[str, Any] | None,
        filename: str | None,
    ) -> list[Any]:
        rows = []
        if filename:
            rows.append(["Archivo", filename])
        rows.append(["Latencia de inferencia", f"{result['inference_ms']:.1f} ms"])

        region = result.get("most_activated_region")
        if region:
            rows.append(
                [
                    "Región más activada",
                    f"centro ({region['cx']:.0f}, {region['cy']:.0f}) — "
                    f"activación media {region['mean_activation']:.3f}",
                ]
            )

        if model_info:
            rows.append(["Modelo", model_info.get("architecture", "—")])
            rows.append(["Checkpoint", model_info.get("checkpoint", "—")])
            total_params = model_info.get("total_params")
            if total_params:
                rows.append(["Parámetros", f"{total_params:,}"])

        table = Table(rows, colWidths=[5 * cm, 12 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 0), (0, -1), GRAY),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.lightgrey),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return [Paragraph("Detalles del análisis", self.heading_style), table]
