# Model Training Report

## Environment
- Device: cpu
- Base Model: MobileNetV2
- Classes: ['O', 'R']

## Hyperparameters
- Batch Size: 16
- Epochs: 5
- Learning Rate: 0.001

## Metrics (Final Epoch)
- Training Loss: 0.6958
- Training Accuracy: 0.5000
- Validation Accuracy: 0.5000

## Artifacts
- ONNX Model exported to: `waste_model.onnx`
- Supported Input Shape: `[-1, 3, 224, 224]`
- Output Shape: `[-1, 2]`

This model has been prepared for deployment via NVIDIA Triton Inference Server.
