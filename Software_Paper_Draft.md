# Software Architecture Paper: Smart Waste Classifier

## Abstract
This document outlines the system design and integration of the "Smart Waste Classifier", an end-to-end MLOps pipeline integrating deep learning for image classification, NVIDIA Triton for high-performance inference, and Generative AI (Gemini) for providing context-aware sustainability advice. The project aligns with Sustainable Development Goal 12 (SDG 12) to ensure sustainable consumption and production patterns.

**Author:** Deepa K


## 1. Introduction
The Smart Waste Classifier is designed to automate the identification of waste types (Organic vs. Recyclable) using a fine-tuned MobileNetV2 architecture. By deploying this model via NVIDIA Triton and augmenting the classification with a Large Language Model (LLM), the system bridges the gap between raw ML predictions and actionable, user-centric recycling guidance.

## 2. System Design

### 2.1 Model Training & Export (Phase 1)
- **Base Architecture**: MobileNetV2, chosen for its efficiency and edge-deployment capabilities.
- **Framework**: PyTorch.
- **Export Format**: ONNX (Open Neural Network Exchange).
- The training pipeline is containerized or managed via standard `requirements.txt` to ensure reproducibility. The final model (`waste_model.onnx`) is exported with dynamic batching support.

### 2.2 High-Performance Serving (Phase 2)
- **Inference Server**: NVIDIA Triton Inference Server.
- **Backend**: ONNX Runtime with CUDA/TensorRT execution accelerators.
- **Configuration**: The `config.pbtxt` defines dynamic batch sizes and FP16 precision mode for optimal GPU utilization.

### 2.3 Backend & GenAI Bridge (Phase 3)
- **API**: FastAPI provides the RESTful interface.
- **Orchestration**: The backend pre-processes the image, sends a gRPC request to Triton using `tritonclient`, and retrieves both the class prediction and AI confidence score.
- **LLM Integration**: LangChain interfaces with the Gemini model. Based on the Triton classification, the LLM generates a structured response containing:
  - Decomposition Timeline
  - Specific Recycling Instructions
  - Upcycling Idea


### 2.4 Interactive Dashboard & UI/UX (Phase 4)
- **Framework**: Streamlit.
- **Aesthetic & UX**: A complete UI overhaul utilizing custom CSS to implement a modern "Glassmorphism" design.
- **Visual Analytics**: AI confidence scores are visually tracked via responsive progress bars to enhance user trust and engagement.
- **Project Context**: The sidebar displays project workflow and developer information (**Deepa K**).

## 3. Production Deployment (Phase 5)
The entire system is containerized using Docker Compose. The `triton-server` container uses GPU pass-through (`deploy.resources.reservations.devices`) for hardware acceleration. A unified deployment script (`export_system.sh`) packages the Docker images, configurations, and models into a `.rar`/`.zip` file for offline distribution and deployment.

## 4. SDG 12 Mapping
This project directly addresses SDG Target 12.5 (substantially reduce waste generation through prevention, reduction, recycling and reuse) and Target 12.8 (ensure that people everywhere have the relevant information and awareness for sustainable development and lifestyles in harmony with nature). By providing real-time, context-specific advice on waste decomposition and upcycling, the system actively educates users and promotes a circular economy.

## 5. Deployment Artifacts
The official source code and deployed container images are available below:
- **GitHub Repository**: [https://github.com/DEEPA-356/Smart-Waste-Classifier](https://github.com/DEEPA-356/Smart-Waste-Classifier)
- **Docker Hub Backend Image**: `deepa356/waste_fastapi_backend:latest`
- **Docker Hub Frontend Image**: `deepa356/waste_streamlit_frontend:latest`

## 6. Conclusion
The Smart Waste Classifier demonstrates a production-ready MLOps workflow. The combination of edge-efficient models, scalable inference servers, and generative AI provides a comprehensive solution for intelligent waste management.
