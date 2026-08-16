"""
MedRax Gradio User Interface
==============================
Defines the complete Gradio Blocks interface for the MedRax medical image
analysis platform.

Layout:
  ┌─────────────────────────────────────────────────┐
  │                    Header                       │
  ├────────────────────┬────────────────────────────┤
  │   Image Upload     │   Analysis Settings        │
  │                    │   - Analysis Type dropdown  │
  │                    │   - Custom Query textbox    │
  │                    │   - Analyze button          │
  ├────────────────────┴────────────────────────────┤
  │              Analysis Output                    │
  │   - Markdown report                             │
  │   - Download buttons (MD / PDF)                 │
  ├─────────────────────────────────────────────────┤
  │              Model Status Bar                   │
  └─────────────────────────────────────────────────┘

This module does NOT import torch or the model directly — it communicates
with the inference pipeline through callback functions, keeping the UI
layer lightweight and decoupled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from config import settings
from models.report_generator import ReportGenerator
from services.image_loader import ImageLoader, ImageLoadError
from services.inference import AnalysisType, InferenceService
from services.model_manager import ModelManager
from utils.image import prepare_image_for_model
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Analysis type choices for the dropdown
# ---------------------------------------------------------------------------
_ANALYSIS_CHOICES: list[tuple[str, str]] = [
    ("🔍 General Analysis", AnalysisType.GENERAL.value),
    ("📋 Detailed Findings", AnalysisType.FINDINGS.value),
    ("🩺 Differential Diagnosis", AnalysisType.DIFFERENTIAL.value),
    ("📝 Formal Radiology Report", AnalysisType.REPORT.value),
    ("📊 Comparative Analysis", AnalysisType.COMPARISON.value),
    ("💬 Patient-Friendly Summary", AnalysisType.PATIENT_FRIENDLY.value),
]

# Default queries per analysis type
_DEFAULT_QUERIES: dict[str, str] = {
    AnalysisType.GENERAL.value: (
        "Please analyze this medical image and provide a comprehensive assessment "
        "of any visible findings, abnormalities, or notable features."
    ),
    AnalysisType.FINDINGS.value: (
        "List all findings visible in this medical image in a structured format, "
        "organized by location and clinical significance."
    ),
    AnalysisType.DIFFERENTIAL.value: (
        "Based on the findings in this medical image, provide a differential "
        "diagnosis with the most likely conditions ranked by probability."
    ),
    AnalysisType.REPORT.value: (
        "Generate a formal radiology report for this medical image including "
        "TECHNIQUE, COMPARISON, FINDINGS, and IMPRESSION sections."
    ),
    AnalysisType.COMPARISON.value: (
        "Describe the findings in this medical image, noting features that "
        "would be important for comparison with prior studies."
    ),
    AnalysisType.PATIENT_FRIENDLY.value: (
        "Explain the findings in this medical image in simple, easy-to-understand "
        "language suitable for a patient."
    ),
}


def create_interface() -> gr.Blocks:
    """Build and return the complete Gradio Blocks interface.

    Returns:
        A configured ``gr.Blocks`` instance ready to be launched.
    """
    # Initialize services
    image_loader = ImageLoader()
    inference_service = InferenceService()
    report_generator = ReportGenerator()

    # ── Custom CSS ────────────────────────────────────────────────────────
    custom_css = """
    .main-header {
        text-align: center;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.7;
    }
    .status-bar {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        text-align: center;
    }
    .disclaimer-box {
        padding: 0.8rem;
        border-radius: 8px;
        font-size: 0.8rem;
        opacity: 0.8;
        text-align: center;
    }
    """

    # ── Build the interface ───────────────────────────────────────────────
    with gr.Blocks(
        title="MedRax — Medical Image Analysis",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        ),
        css=custom_css,
    ) as interface:

        # ── Header ───────────────────────────────────────────────────────
        gr.HTML(
            """
            <div class="main-header">
                <h1>🏥 MedRax</h1>
                <p>AI-Powered Medical Image Analysis · Powered by MedGemma</p>
            </div>
            """
        )

        # ── Model status ─────────────────────────────────────────────────
        model_status = gr.Textbox(
            value="⏳ Model not loaded — click 'Load Model' or upload an image to begin.",
            label="Model Status",
            interactive=False,
            elem_classes=["status-bar"],
        )

        # Hidden state for tracking
        model_loaded_state = gr.State(value=False)

        # ── Main layout ──────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            # ── Left column: Image upload ────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Medical Image",
                    type="filepath",
                    height=400,
                    sources=["upload"],
                    elem_id="image-upload",
                )

                gr.Markdown(
                    "**Supported formats:** JPEG, PNG, DICOM (.dcm)\n\n"
                    f"**Max file size:** {settings.max_image_size_mb:.0f} MB"
                )

                load_model_btn = gr.Button(
                    "🔄 Load Model",
                    variant="secondary",
                    size="sm",
                )

            # ── Right column: Settings & controls ────────────────────────
            with gr.Column(scale=1):
                analysis_type = gr.Dropdown(
                    choices=_ANALYSIS_CHOICES,
                    value=AnalysisType.GENERAL.value,
                    label="Analysis Type",
                    info="Select the type of analysis to perform.",
                    elem_id="analysis-type",
                )

                query_input = gr.Textbox(
                    value=_DEFAULT_QUERIES[AnalysisType.GENERAL.value],
                    label="Clinical Query",
                    placeholder="Enter your clinical question about the image...",
                    lines=4,
                    max_lines=8,
                    elem_id="query-input",
                )

                analyze_btn = gr.Button(
                    "🔬 Analyze Image",
                    variant="primary",
                    size="lg",
                    elem_id="analyze-btn",
                )

                clear_btn = gr.Button(
                    "🗑️ Clear All",
                    variant="secondary",
                    size="sm",
                )

        # ── Output section ───────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("## 📄 Analysis Results")

        output_report = gr.Markdown(
            value="*Upload an image and click 'Analyze Image' to see results here.*",
            label="Report",
            elem_id="output-report",
        )

        with gr.Row():
            download_md_btn = gr.Button("📥 Download Report (Markdown)", size="sm")
            download_pdf_btn = gr.Button("📥 Download Report (PDF)", size="sm")

        download_file = gr.File(
            label="Download",
            visible=False,
            elem_id="download-file",
        )

        # Hidden state to hold the current report object
        current_report_state = gr.State(value=None)

        # ── Disclaimer ───────────────────────────────────────────────────
        gr.HTML(
            """
            <div class="disclaimer-box">
                ⚠️ <strong>For educational and research purposes only.</strong>
                Not a substitute for professional medical advice.
                Always consult a qualified healthcare professional.
            </div>
            """
        )

        # ══════════════════════════════════════════════════════════════════
        # Event handlers
        # ══════════════════════════════════════════════════════════════════

        def update_default_query(analysis_type_value: str) -> str:
            """Update the query textbox when analysis type changes."""
            return _DEFAULT_QUERIES.get(analysis_type_value, _DEFAULT_QUERIES[AnalysisType.GENERAL.value])

        def load_model_handler() -> tuple[str, bool]:
            """Load the MedGemma model."""
            try:
                manager = ModelManager()
                if manager.is_loaded:
                    return "✅ Model is already loaded and ready.", True

                manager.load_model()
                device = manager.device_info
                return f"✅ Model loaded successfully on {device}.", True

            except Exception as exc:
                logger.error("Model loading failed: %s", exc)
                return f"❌ Failed to load model: {exc}", False

        def analyze_image_handler(
            image_path: Optional[str],
            query: str,
            analysis_type_value: str,
            is_model_loaded: bool,
        ) -> tuple[str, Optional[object], str, bool]:
            """Main analysis pipeline: load image → preprocess → infer → report.

            Returns:
                Tuple of (report_markdown, report_object, status_text, model_loaded).
            """
            if image_path is None:
                return (
                    "⚠️ **Please upload a medical image first.**",
                    None,
                    "⚠️ No image uploaded.",
                    is_model_loaded,
                )

            if not query.strip():
                return (
                    "⚠️ **Please enter a clinical query.**",
                    None,
                    "⚠️ No query provided.",
                    is_model_loaded,
                )

            try:
                # Step 1: Load and validate the image
                status_msg = "🔄 Loading image..."
                pil_image = image_loader.load(image_path)
                image_filename = Path(image_path).name

                # Step 2: Preprocess
                pil_image = prepare_image_for_model(pil_image)

                # Step 3: Ensure model is loaded
                manager = ModelManager()
                if not manager.is_loaded:
                    status_msg = "🔄 Loading model (first time — this may take a few minutes)..."
                    manager.load_model()
                    is_model_loaded = True

                # Step 4: Run inference
                status_msg = "🧠 Analyzing image with MedGemma..."
                analysis_type_enum = AnalysisType(analysis_type_value)
                raw_result = inference_service.analyze(
                    image=pil_image,
                    query=query,
                    analysis_type=analysis_type_enum,
                    add_disclaimer=False,  # Report generator adds its own
                )

                # Step 5: Generate structured report
                report = report_generator.create_report(
                    analysis_text=raw_result,
                    query=query,
                    analysis_type=analysis_type_value,
                    image_filename=image_filename,
                )

                # Step 6: Save report
                report_generator.save_report(report)

                # Step 7: Optional GCS upload
                if settings.gcs_enabled:
                    report_generator.save_to_gcs(report)

                return (
                    report.markdown,
                    report,
                    f"✅ Analysis complete — Report ID: {report.report_id[:8]}",
                    is_model_loaded,
                )

            except ImageLoadError as exc:
                logger.error("Image loading error: %s", exc)
                return (
                    f"❌ **Image Error:** {exc}",
                    None,
                    f"❌ Image error: {exc}",
                    is_model_loaded,
                )
            except Exception as exc:
                logger.error("Analysis failed: %s", exc, exc_info=True)
                return (
                    f"❌ **Analysis Failed:** {exc}\n\n"
                    "Please check the logs for more details.",
                    None,
                    f"❌ Analysis error: {exc}",
                    is_model_loaded,
                )

        def download_markdown_handler(
            report_obj: Optional[object],
        ) -> Optional[str]:
            """Generate a downloadable Markdown file from the current report."""
            if report_obj is None:
                gr.Warning("No report to download. Run an analysis first.")
                return None

            try:
                path = report_generator.save_report(report_obj)
                return str(path)
            except Exception as exc:
                logger.error("Markdown download failed: %s", exc)
                gr.Warning(f"Download failed: {exc}")
                return None

        def download_pdf_handler(
            report_obj: Optional[object],
        ) -> Optional[str]:
            """Generate a downloadable PDF from the current report."""
            if report_obj is None:
                gr.Warning("No report to download. Run an analysis first.")
                return None

            try:
                path = report_generator.export_pdf(report_obj)
                return str(path)
            except Exception as exc:
                logger.error("PDF download failed: %s", exc)
                gr.Warning(f"PDF export failed: {exc}")
                return None

        def clear_all_handler():
            """Reset all inputs and outputs."""
            return (
                None,  # image_input
                AnalysisType.GENERAL.value,  # analysis_type
                _DEFAULT_QUERIES[AnalysisType.GENERAL.value],  # query_input
                "*Upload an image and click 'Analyze Image' to see results here.*",  # output
                None,  # report state
                "⏳ Ready for a new analysis.",  # status
            )

        # ── Wire up events ───────────────────────────────────────────────

        # Update query when analysis type changes
        analysis_type.change(
            fn=update_default_query,
            inputs=[analysis_type],
            outputs=[query_input],
        )

        # Load model button
        load_model_btn.click(
            fn=load_model_handler,
            outputs=[model_status, model_loaded_state],
        )

        # Main analyze button
        analyze_btn.click(
            fn=analyze_image_handler,
            inputs=[image_input, query_input, analysis_type, model_loaded_state],
            outputs=[output_report, current_report_state, model_status, model_loaded_state],
        )

        # Download buttons
        download_md_btn.click(
            fn=download_markdown_handler,
            inputs=[current_report_state],
            outputs=[download_file],
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[download_file],
        )

        download_pdf_btn.click(
            fn=download_pdf_handler,
            inputs=[current_report_state],
            outputs=[download_file],
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[download_file],
        )

        # Clear button
        clear_btn.click(
            fn=clear_all_handler,
            outputs=[
                image_input,
                analysis_type,
                query_input,
                output_report,
                current_report_state,
                model_status,
            ],
        )

    logger.info("Gradio interface created.")
    return interface
