# 🩺 MedRAX – AI-Powered Medical Imaging Analysis & Reporting

MedRAX is an AI-powered medical imaging analysis platform that assists healthcare professionals by analyzing medical images and generating structured clinical reports using state-of-the-art Large Language Models (LLMs) and computer vision models.

The project combines medical image understanding, multimodal AI, Retrieval-Augmented Generation (RAG), and modern web technologies to provide an intelligent assistant for radiology workflows.

> **Disclaimer:** MedRAX is intended for educational, research, and decision-support purposes only. It is **not** a replacement for professional medical diagnosis.

---

# ✨ Features

* 📤 Upload medical images
* 🤖 AI-powered image analysis
* 📝 Automatic medical report generation
* 💬 Intelligent medical chatbot
* 🔍 Retrieval-Augmented Generation (RAG)
* 📚 Medical knowledge retrieval
* 🧠 Large Language Model integration
* 🌐 Modern responsive web interface
* ⚡ FastAPI backend
* ☁️ Cloud-ready deployment

---

# 🏗️ Project Architecture

```text
                Medical Image
                      │
                      ▼
             Image Processing Layer
                      │
                      ▼
         Vision / Multimodal AI Model
                      │
                      ▼
            Clinical Findings Extraction
                      │
                      ▼
              RAG Knowledge Retrieval
                      │
                      ▼
            Large Language Model (LLM)
                      │
                      ▼
      Structured Radiology Report + Chat
```

---

# 📂 Project Structure

```text
MedRAX/
│
├── backend/
│   ├── api/
│   ├── services/
│   ├── models/
│   ├── rag/
│   ├── prompts/
│   ├── utils/
│   └── main.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   └── components/
│
├── data/
│   ├── images/
│   ├── reports/
│   └── knowledge_base/
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
├── .env.example
├── README.md
└── LICENSE
```

---

# 🚀 Technology Stack

### Backend

* Python 3.11+
* FastAPI
* Uvicorn

### AI & Machine Learning

* Hugging Face Transformers
* Vision-Language Models (VLMs)
* OpenAI / Gemini compatible APIs
* LangChain
* Sentence Transformers

### Retrieval

* FAISS
* RAG Pipeline
* Embedding Models

### Frontend

* React
* TypeScript
* Tailwind CSS

### Database & Storage

* SQLite / PostgreSQL
* Local Storage
* Cloud Storage

### Deployment

* Docker
* Google Cloud Platform
* Cloud Run

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/samahelmenady/medrax.git
cd medrax
```

Create a virtual environment:

```bash
conda create -n medrax python=3.11
conda activate medrax
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Add your API keys inside `.env`.

---

# ▶️ Running the Project

Start the backend:

```bash
uvicorn backend.main:app --reload
```

Start the frontend:

```bash
npm install
npm run dev
```

---

# 🧠 AI Workflow

1. User uploads a medical image.
2. The vision model extracts visual findings.
3. Relevant medical knowledge is retrieved through RAG.
4. The LLM combines image findings with retrieved knowledge.
5. A structured radiology report is generated.
6. Users can ask follow-up questions through the AI assistant.

---

# 📋 Generated Report Includes

* Examination Type
* Clinical Findings
* Impression
* Possible Differential Diagnoses
* Recommendations
* Follow-up Suggestions

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

---

# 👨‍💻 Author

Developed as an AI-powered medical imaging assistant project using modern Generative AI, Computer Vision, and Retrieval-Augmented Generation technologies.

---

## ⭐ If you find this project useful, consider giving it a star on GitHub!
