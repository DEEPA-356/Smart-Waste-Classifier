#!/bin/bash

# Exit on error
set -e

DOCKER_USERNAME=$1

echo "Building Docker images..."
docker-compose build

if [ ! -z "$DOCKER_USERNAME" ]; then
    echo "♻️ Tagging and pushing images to Docker Hub for user: $DOCKER_USERNAME"
    
    # Tag images
    docker tag waste_fastapi_backend "$DOCKER_USERNAME/waste_fastapi_backend:latest"
    docker tag waste_streamlit_frontend "$DOCKER_USERNAME/waste_streamlit_frontend:latest"
    
    # Push images
    docker push "$DOCKER_USERNAME/waste_fastapi_backend:latest"
    docker push "$DOCKER_USERNAME/waste_streamlit_frontend:latest"
    echo "✅ Successfully pushed images to Docker Hub."
else
    echo "ℹ️ No Docker username provided. Skipping docker push."
    echo "Usage to push: ./export_system.sh <dockerhub_username>"
fi

echo "Saving Triton Server image..."
docker save -o triton-server.tar nvcr.io/nvidia/tritonserver:23.10-py3

echo "Saving Backend App image..."
docker save -o app-backend.tar waste_fastapi_backend

echo "Saving Frontend App image..."
docker save -o app-frontend.tar waste_streamlit_frontend

echo "Compressing into WasteAI_Production.rar..."
# Using rar if installed, else zip as fallback
if command -v rar &> /dev/null; then
    rar a WasteAI_Production.rar docker-compose.yml model_repository/ triton-server.tar app-backend.tar app-frontend.tar .env
else
    echo "rar not found, using zip instead."
    zip -r WasteAI_Production.zip docker-compose.yml model_repository/ triton-server.tar app-backend.tar app-frontend.tar .env
fi

echo "Cleaning up tar files..."
rm triton-server.tar app-backend.tar app-frontend.tar

echo "Export complete: WasteAI_Production archive created."
