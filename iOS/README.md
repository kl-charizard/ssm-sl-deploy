# 📱 iOS Sign Language Detector App

A real-time sign language detection iOS app built with Core ML and Vision framework.

## ✨ Features

- **🤚 Real-time Hand Detection**: Uses Vision framework for robust hand detection
- **🎯 Sign Language Recognition**: Core ML-powered ASL alphabet recognition
- **📱 Native iOS Experience**: Built with Swift and UIKit
- **⚡ Optimized Performance**: Hand-cropped images for better accuracy
- **🎨 Modern UI**: Clean, intuitive interface with live camera preview

## 🚀 Quick Start

### Prerequisites

- **Xcode 14.0+** (iOS 14.0+ deployment target)
- **macOS 12.0+** for development
- **Trained PyTorch model** from the main framework

### 1. Convert Your Model to Core ML

First, convert your trained PyTorch model to Core ML format:

```bash
# From the main project directory
python scripts/convert_to_coreml.py \
    --model-path checkpoints/best_model.pth \
    --output-path ios/SignLanguageDetector/SignLanguageModel.mlmodel \
    --optimize
```

This will create:
- `SignLanguageModel.mlmodel` - The Core ML model
- `SignLanguageModel_optimized.mlmodel` - Quantized version for better performance

### 2. Open the iOS Project

1. Open `ios/SignLanguageDetector.xcodeproj` in Xcode
2. The project should automatically detect the Core ML model
3. If not, drag the `.mlmodel` file into the Xcode project

### 3. Build and Run

1. Select your target device or simulator
2. Press `Cmd + R` to build and run
3. Grant camera permissions when prompted

## 📁 Project Structure

```
ios/SignLanguageDetector/
├── SignLanguageDetector.xcodeproj/     # Xcode project file
├── SignLanguageDetector/               # Source code
│   ├── AppDelegate.swift               # App lifecycle
│   ├── SceneDelegate.swift             # Scene management
│   ├── ViewController.swift            # Main menu
│   ├── CameraViewController.swift      # Camera and detection
│   ├── SignLanguageModel.swift         # Core ML integration
│   ├── HandDetectionHelper.swift       # Vision framework helpers
│   ├── Main.storyboard                 # UI layout
│   ├── LaunchScreen.storyboard         # Launch screen
│   ├── Assets.xcassets/                # App icons and images
│   └── Info.plist                      # App configuration
└── README.md                           # This file
```

## 🔧 Configuration

### Model Settings

Edit `SignLanguageModel.swift` to customize:

```swift
// ASL Alphabet classes (A-Z + space, del, nothing)
private let classNames = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "space", "del", "nothing"
]
```

### Hand Detection Settings

Modify `HandDetectionHelper.swift`:

```swift
// Maximum number of hands to detect
handPoseRequest.maximumHandCount = 2

// Padding around detected hand (20% extra)
let padding: CGFloat = 0.2
```

### UI Customization

The app uses Storyboard for UI layout. Key elements:

- **Main View**: Welcome screen with start button
- **Camera View**: Live camera preview with prediction overlay
- **Labels**: Real-time prediction, confidence, and hand detection status

## 🎯 How It Works

### 1. Hand Detection
- Uses Vision framework's `VNDetectHumanHandPoseRequest`
- Detects hand landmarks in real-time
- Crops hand region with padding for better recognition

### 2. Sign Language Recognition
- Preprocesses cropped hand image (resize to 224x224)
- Feeds image to Core ML model
- Returns predicted letter with confidence score

### 3. Real-time Processing
- Processes video frames asynchronously
- Updates UI on main thread
- Handles camera permissions and errors gracefully

## 📱 App Flow

1. **Launch**: Welcome screen with app description
2. **Start Detection**: Tap "Start Detection" to open camera
3. **Camera View**: 
   - Live camera preview
   - Hand detection status
   - Real-time predictions
   - Confidence scores
4. **Close**: Tap "✕" to return to main menu

## 🔍 Troubleshooting

### Common Issues

#### Model Not Found
```
Core ML model not found. Please add the converted model to the app bundle.
```
**Solution**: 
1. Run the conversion script
2. Drag the `.mlmodel` file into Xcode project
3. Ensure it's added to the app target

#### Camera Permission Denied
**Solution**: 
1. Go to Settings > Privacy & Security > Camera
2. Enable camera access for the app
3. Or delete and reinstall the app

#### Poor Recognition Accuracy
**Solutions**:
1. Ensure good lighting
2. Keep hand centered in frame
3. Use the optimized model version
4. Check that hand detection is working (green checkmark)

#### App Crashes on Launch
**Solutions**:
1. Check iOS version (requires 14.0+)
2. Verify model file is valid
3. Check Xcode console for error messages
4. Try running on simulator first

### Performance Optimization

#### For Better Performance:
1. Use the optimized/quantized model
2. Close other apps to free memory
3. Use a physical device instead of simulator
4. Ensure good lighting conditions

#### For Better Accuracy:
1. Keep hand well-lit and centered
2. Avoid busy backgrounds
3. Use consistent hand positioning
4. Ensure hand is fully visible in frame

## 🛠️ Development

### Adding New Features

#### Custom Hand Detection
Modify `HandDetectionHelper.swift`:

```swift
// Add custom hand detection logic
func detectHandsWithCustomLogic() {
    // Your custom implementation
}
```

#### Additional UI Elements
Edit `Main.storyboard` or `CameraViewController.swift`:

```swift
// Add new UI elements
@IBOutlet weak var newLabel: UILabel!

// Update in viewDidLoad()
newLabel.text = "Custom text"
```

#### Model Customization
Update `SignLanguageModel.swift`:

```swift
// Add new classes
private let classNames = [
    // ... existing classes
    "new_gesture"
]
```

### Testing

#### Unit Tests
Create test files in Xcode:
1. File > New > Target > iOS Unit Testing Bundle
2. Test model loading and prediction logic

#### UI Tests
1. File > New > Target > iOS UI Testing Bundle
2. Test app flow and camera functionality

## 📋 Requirements

### System Requirements
- **iOS**: 14.0+
- **Device**: iPhone/iPad with camera
- **Storage**: ~50MB for app + model

### Development Requirements
- **Xcode**: 14.0+
- **macOS**: 12.0+
- **Swift**: 5.0+
- **Deployment Target**: iOS 14.0

## 🚀 Deployment

### App Store Deployment

1. **Archive the app**:
   - Product > Archive in Xcode
   - Follow the distribution wizard

2. **Upload to App Store Connect**:
   - Use Xcode Organizer
   - Or Application Loader

3. **Configure in App Store Connect**:
   - Add app metadata
   - Upload screenshots
   - Submit for review

### Enterprise Distribution

1. **Create Enterprise Certificate**:
   - Apple Developer Program
   - Enterprise distribution profile

2. **Build for Enterprise**:
   - Product > Archive
   - Export for Enterprise Distribution

3. **Distribute**:
   - Upload to your distribution platform
   - Or distribute via MDM

## 📚 Additional Resources

- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Vision Framework Guide](https://developer.apple.com/documentation/vision)
- [AVFoundation Camera Guide](https://developer.apple.com/documentation/avfoundation)
- [iOS Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the main project README for details.
