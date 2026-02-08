# Basketball Jersey Numbers OCR

Automated detection and recognition of jersey numbers in basketball images using computer vision. This project leverages YOLOv8-based models trained via Roboflow for real-time OCR inference with local GPU acceleration.

![baskgif](https://github.com/user-attachments/assets/4e4ccf56-d92b-4a2e-933b-de5ae8b02853)


## Technical Overview

| Component | Description |
|-----------|-------------|
| Model | basketball-jersey-numbers-ocr/7 (Roboflow) |
| Architecture | YOLOv8 object detection with VLM support |
| Inference | Local GPU (100% free, no API credit consumption) |
| Interface | Gradio web application |
| Platform | Google Colab with NVIDIA T4 GPU |

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (or Google Colab T4)
- Roboflow API key (free tier available)

### Dependencies

```
inference
supervision
gradio
roboflow
pillow
opencv-python
torch
```

## Quick Start

### Option A: Google Colab (Recommended)

1. Open `test-colab.ipynb` in Google Colab
2. Set runtime to GPU: Runtime > Change runtime type > GPU (T4)
3. Execute cells sequentially (1-6)
4. Enter your Roboflow API key in Cell 6
5. Access the Gradio interface via the generated public URL

### Option B: Local Execution

```bash
pip install -r requirements.txt
python basketball_jersey_analyzer.py
```

### Obtaining Roboflow API Key

1. Create account at https://app.roboflow.com
2. Navigate to Settings > API Keys
3. Copy your Private API Key

## Project Structure

```
basketball-jersey-numbers/
    test-colab.ipynb              # Main notebook for Colab
    basketball_jersey_analyzer.py # Python script version
    requirements.txt              # Python dependencies
    sample_images/                # Test images
    .gitignore                    # Git exclusions
```

## Features

### Inference Pipeline

- Downloads model weights on first run via Roboflow API
- Executes inference locally on GPU (no per-request API costs)
- Supports both YOLO detection and VLM response formats
- Automatic bounding box visualization with confidence scores

### Gradio Interface

- Real-time image upload and webcam capture
- Adjustable confidence threshold (0.1 - 0.9)
- Detection statistics display
- CSV export functionality
- Automatic detection logging

### Detection Output

- Bounding boxes with jersey number labels
- Confidence score for each detection
- Aggregated statistics (total, average, max, min confidence)
- Persistent CSV log of all detections

## Technical Details

### Model Architecture

The system uses a fine-tuned YOLOv8 model trained on basketball jersey images. The model supports:

- Single and multi-digit number detection
- Partial occlusion handling
- Variable jersey colors and lighting conditions

### VLM Integration

For models that return VLM (Visual Language Model) responses, the system:

1. Parses text responses using regex
2. Extracts numeric values from natural language output
3. Generates synthetic bounding boxes centered on the image

### Memory Management

- Automatic GPU cache clearing before model loading
- Graceful handling of CUDA memory allocation
- Support for 16GB VRAM GPUs (T4)

## Usage Example

```python
from jersey_analyzer import JerseyAnalyzer

analyzer = JerseyAnalyzer(api_key="YOUR_API_KEY")
image = cv2.imread("basketball_player.jpg")
annotated_image, detections = analyzer.detectar_numeros(image)

for det in detections:
    print(f"Number: {det['numero']}, Confidence: {det['confianza']}")
```

## Configuration Files

| File | Purpose |
|------|---------|
| COLAB_NOTEBOOK_GUIDE.txt | Detailed Colab setup instructions |
| INICIO_RAPIDO.txt | Quick start guide |
| COMO_PROBAR.txt | Testing instructions |
| requirements.txt | Python package dependencies |

## References

- [Roboflow Universe - Basketball Jersey Numbers Dataset](https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr/dataset/7)
- [Roboflow Inference Documentation](https://inference.roboflow.com/)
- [Supervision Library Documentation](https://supervision.roboflow.com/)

## License

Academic project - Universidad Peruana de Ciencias Aplicadas (UPC) 2025
