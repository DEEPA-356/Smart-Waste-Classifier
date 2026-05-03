# ♻️ Smart Waste Classifier (End-to-End MLOps System)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red.svg)](https://streamlit.io/)
[![Triton Inference Server](https://img.shields.io/badge/Triton-23.10-76B900.svg)](https://developer.nvidia.com/nvidia-triton-inference-server)
[![SDG 12](https://img.shields.io/badge/SDG-12_Responsible_Consumption-orange.svg)](https://sdgs.un.org/goals/goal12)

An autonomous, end-to-end MLOps pipeline integrating deep learning for image classification, NVIDIA Triton for high-performance inference, and Generative AI (Google Gemini) for providing context-aware sustainability advice. 

---

## 🌍 Sustainable Development Goal 12 Mapping

This project directly aligns with **SDG 12: Responsible Consumption and Production**:
*   **Target 12.5:** Substantially reduce waste generation through prevention, reduction, recycling, and reuse.
*   **Target 12.8:** Ensure that people everywhere have the relevant information and awareness for sustainable development and lifestyles in harmony with nature.

By bridging raw ML predictions with a Generative AI model, this system actively educates users with actionable insights:
1. **Decomposition Timelines** to understand long-term environmental impact.
2. **Specific Recycling Instructions** to reduce contamination in recycling streams.
3. **Upcycling Ideas** to encourage creative reuse and the circular economy.

---

## 🏛️ System Architecture

1. **Model Training (Phase 1):** PyTorch-based fine-tuning of MobileNetV2 for high accuracy on edge-capable architectures. The model is exported to `ONNX` format.
2. **High-Performance Serving (Phase 2):** NVIDIA Triton Inference Server deployed with GPU acceleration (`tensorrt`, `FP16`) to dynamically batch and serve the ONNX model.
3. **GenAI Bridge Backend (Phase 3):** A FastAPI microservice that utilizes `tritonclient` to get image classifications and confidence scores, and uses LangChain to connect to Google's Gemini-3 model to generate sustainability metadata.
4. **Interactive Dashboard (Phase 4):** A fully revamped Streamlit frontend utilizing custom CSS for a modern "Glassmorphism" aesthetic. It includes confidence score tracking via progress bars, visually engaging results, and developer credentials (SRM Institute of Science and Technology, AI & ML Department) in the sidebar.

---

## 🚀 Quick Start (Docker Deployment)

The entire architecture is containerized and orchestrated via `docker-compose`.

### Prerequisites
*   Docker & Docker Compose
*   NVIDIA Container Toolkit (for GPU pass-through)
*   A Google Gemini API Key

### Setup
1. Clone the repository and navigate to the root directory.
2. Create a `.env` file and add your Gemini API Key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```
3. Ensure the exported model exists at `model_repository/waste_model/1/model.onnx`.

### Run
Launch the entire stack using Docker Compose:
```bash
docker-compose up -d --build
```
Access the Streamlit Dashboard at: `http://localhost:8501`

---

## 📦 Production Packaging

To build, push, and compress the system into a distributable archive (`WasteAI_Production.rar`), run:
```bash
bash export_system.sh
```
