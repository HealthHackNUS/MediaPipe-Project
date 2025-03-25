# Welcome to MoveSync :smile:

## Introduction 
Movesync is a pilot project developed for a hackathon, serving as a proof of concept for remote rehabilitation technology. This cutting-edge device leverages advanced computer vision and artificial intelligence to aid patients in their physiotherapy and rehabilitation journey from the comfort of their homes.

### Key features of MoveSync include:
1. Real-time exercise form detection
2. Personalized progress tracking
3. AI-driven exercise recommendations
4. Edge AI processing with Hailo-8 AI Accelerator

### Hardware components:
1. Raspberry Pi 5, Jetson Nano, or any Linux-based edge device
2. Hailo-8 AI Accelerator (M.2 HAT)
3. Raspberry Pi Camera or USB Webcam

### Software stack:
1. GStreamer for high-performance video streaming
2. GObject Introspection (GI) for integrating GStreamer in Python
3. Hailo TAPPAS SDK for optimizing models on Hailo-8

By combining state-of-the-art motion capture technology, Movesync aims to bridge the gap between in-person physiotherapy sessions and at-home exercises.The Hailo-8 AI Accelerator provides powerful __edge AI capabilities__, enabling __real-time processing__ of exercise movements. GStreamer ensures smooth video streaming, while the Hailo TAPPAS SDK optimizes AI models for efficient execution on the Hailo-8 hardware.

This pilot project demonstrates the potential for accessible, efficient, and effective remote rehabilitation solutions, paving the way for improved patient outcomes and reduced healthcare costs. The modular hardware design allows for easy integration and future upgrades, ensuring Movesync can evolve with advancing technology.

### About MoveSync
![moveSync](/image.png)
*Image of Movesync*


### The required libraries for this project include: 
- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- GStreamer
- GObhect Introspection (GI)
- Hailo TAPPAS SDK
