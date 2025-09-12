# Sign Language Detection iOS App

A complete iOS app for real-time American Sign Language (ASL) alphabet recognition using Core ML and Vision framework.

## Features

- **Real-time Detection**: Live camera feed with sign language recognition
- **Hand Detection**: Automatically crops hand regions for better accuracy
- **29 ASL Signs**: Supports A-Z plus del, nothing, space
- **Beautiful UI**: Modern SwiftUI interface with gradients and animations
- **Core ML Integration**: Uses trained EfficientNet model
- **Confidence Display**: Shows detection confidence percentage

## Requirements

- iOS 17.0+
- Xcode 15.0+
- Camera access permission

## How to Build

1. **Open Xcode**:
   ```bash
   open demo.xcodeproj
   ```

2. **Select Target**:
   - Choose iPhone simulator or device
   - Recommended: iPhone 15 or newer

3. **Build & Run**:
   - Press ⌘+R or click the Run button
   - Grant camera permission when prompted

## App Structure

- **demoApp.swift**: Main app entry point
- **ContentView.swift**: Beautiful SwiftUI interface
- **CameraManager.swift**: Camera & detection management
- **HandDetectionHelper.swift**: Hand detection using Vision
- **SignLanguageModel.swift**: Core ML model wrapper
- **SignLanguageModel.mlmodel**: Trained model (19MB)
- **SignLanguageModel_optimized.mlmodel**: Optimized model (5MB)

## Usage

1. **Main Screen**: Shows detection status and "Start Camera Detection" button
2. **Camera Screen**: Live camera feed with real-time sign recognition
3. **Detection**: Shows detected sign letter and confidence percentage

## Model Information

- **Architecture**: EfficientNet-B0
- **Input Size**: 224x224 pixels
- **Classes**: 29 ASL alphabet signs
- **Optimization**: Quantized for mobile deployment

## Troubleshooting

- **Camera Permission**: Make sure to grant camera access
- **Model Loading**: Ensure .mlmodel files are in the app bundle
- **Hand Detection**: Works best with clear hand visibility and good lighting

## Performance

- **Model Size**: 5MB (optimized) / 19MB (full)
- **Inference Speed**: ~50ms on iPhone 15
- **Memory Usage**: ~100MB during detection
