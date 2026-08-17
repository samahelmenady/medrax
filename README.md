# 🏥 MedRax – AI-Powered Medical Image Analysis

MedRax is an AI-powered medical imaging analysis platform built with **Google MedGemma**, designed to analyze medical images and generate structured clinical reports. The application supports multiple medical image formats, provides different analysis modes, and exports professional reports in Markdown and PDF formats.

---

## ✨ Features

- Upload medical images (JPEG, PNG, DICOM)
- AI-powered image analysis using **Google MedGemma**
- Multiple analysis modes:
  - General Analysis
  - Findings
  - Differential Diagnosis
  - Formal Radiology Report
  - Comparison Analysis
  - Patient-Friendly Explanation
- Automatic image preprocessing
- Structured medical report generation
- Export reports as Markdown and PDF
- Optional Google Cloud Storage integration
- Interactive web interface built with Gradio

---

# Project Architecture

```
User
   │
   ▼
Gradio UI (interface.py)
   │
   ├── Upload Image
   ├── Choose Analysis Type
   ├── Write Query
   │
   ▼
ImageLoader
   │
   ├── Validate file
   ├── Read PNG/JPG/DICOM
   └── Return PIL Image
   │
   ▼
prepare_image_for_model()
   │
   ├── Verify image
   ├── Convert RGB
   └── Resize
   │
   ▼
InferenceService
   │
   ├── Build Prompt
   ├── Add System Prompt
   ├── Add Image
   ├── Add User Query
   │
   ▼
ModelManager
   │
   ├── Load MedGemma (once)
   ├── Load Processor
   └── Generate Response
   │
   ▼
InferenceService
   │
   ├── Decode Output
   ├── Clean Text
   └── Return Raw Analysis
   │
   ▼
ReportGenerator
   │
   ├── Parse Sections
   ├── Add Metadata
   ├── Generate Markdown
   ├── Save Report
   ├── Export PDF
   └── Upload to GCS (optional)
   │
   ▼
Gradio UI
   │
   ├── Display Report
   ├── Download MD
   └── Download PDF
```

---

# Project Structure

```
medrax/
├── .dockerignore
├── .gitignore
├── .python-version
├── .env
├── Dockerfile
├── README.md
├── app.py
├── config.py
├── deploy.sh
├── requirements.txt
│
├── assets/                 
│
├── data/                   
│   ├── uploads/
│   └── reports/
│
├── models/
│   ├── __init__.py
│   └── report_generator.py
│
├── services/
│   ├── __init__.py
│   ├── image_loader.py
│   ├── inference.py
│   └── model_manager.py
│
├── ui/
│   ├── __init__.py
│   └── interface.py
│
└── utils/
    ├── __init__.py
    ├── image.py
    └── logger.py

```

---

# Workflow

1. Upload a medical image.
2. Select an analysis type.
3. Enter a clinical question (optional).
4. The image is validated and preprocessed.
5. MedGemma performs multimodal analysis.
6. The AI response is converted into a structured report.
7. The report is displayed and can be downloaded as Markdown or PDF.

---

# Supported Image Formats

- JPEG
- PNG
- DICOM (.dcm)

---

# Analysis Types

| Analysis | Description |
|----------|-------------|
| General | Overall image analysis |
| Findings | Lists important findings |
| Differential | Suggests possible diagnoses |
| Report | Generates a formal radiology report |
| Comparison | Highlights findings for follow-up comparison |
| Patient Friendly | Explains results in simple language |

---

# Technologies Used

- Python
- Gradio
- PyTorch
- Hugging Face Transformers
- Google MedGemma
- Pillow (PIL)
- NumPy
- pydicom
- FPDF2
- Google Cloud Storage (Optional)

---

# Main Components

## UI Layer
Provides the Gradio web interface for image upload, analysis settings, and report visualization.

## Image Loader
Loads, validates, and processes JPEG, PNG, and DICOM images.

## Image Utilities
Converts images to RGB, validates dimensions, and resizes images before inference.

## Model Manager
Loads and manages the MedGemma model using the Singleton pattern.

## Inference Service
Builds multimodal prompts, runs MedGemma inference, and decodes model outputs.

## Report Generator
Transforms raw AI responses into structured medical reports and exports them as Markdown or PDF.

---

# Output

The generated report contains:

- Report Information
- Clinical Query
- Medical Findings
- Impression / Analysis
- AI Disclaimer

Reports can be exported as:

- Markdown (.md)
- PDF (.pdf)

---

# Future Improvements

- Multi-image comparison
- Medical image segmentation
- Patient history integration
- Voice input
- Electronic Health Record (EHR) integration
- User authentication
- Report history dashboard

---

# Disclaimer

This application is intended for **educational and research purposes only**.

It is **not** a substitute for professional medical diagnosis, treatment, or clinical decision-making. Always consult a qualified healthcare professional for medical advice.

---

# Author

Developed as an AI-powered medical imaging analysis system using **Google MedGemma** and **Gradio**.
