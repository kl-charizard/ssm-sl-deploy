#!/bin/bash

# Setup script for iOS Sign Language Detection App
# This script ensures the Core ML models are properly set up

echo "🤟 Setting up Sign Language Detection iOS App..."

# Check if we're in the right directory
if [ ! -f "demo.xcodeproj/project.pbxproj" ]; then
    echo "❌ Error: Please run this script from the iOS/demo directory"
    exit 1
fi

# Check if models exist
if [ ! -f "SignLanguageModel.mlmodel" ] || [ ! -f "SignLanguageModel_optimized.mlmodel" ]; then
    echo "📥 Converting PyTorch model to Core ML..."
    
    # Go to project root
    cd ../../
    
    # Convert model
    python scripts/convert_to_coreml.py \
        --model-path checkpoints/best_model.pth \
        --output-path iOS/demo/SignLanguageModel.mlmodel \
        --optimize
    
    # Go back to iOS demo directory
    cd iOS/demo
fi

echo "✅ Core ML models are ready!"
echo "📱 You can now open demo.xcodeproj in Xcode and build the app"
echo ""
echo "🚀 Next steps:"
echo "1. Open demo.xcodeproj in Xcode"
echo "2. Select iPhone simulator"
echo "3. Press ⌘+R to build and run"
echo "4. Grant camera permission when prompted"
