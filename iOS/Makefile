# iOS Sign Language Detector Makefile
# Provides convenient commands for iOS development and deployment

.PHONY: help setup convert build run test clean archive

# Default target
help:
	@echo "🤟 iOS Sign Language Detector - Available Commands:"
	@echo ""
	@echo "📋 Setup & Preparation:"
	@echo "  setup          - Install dependencies and prepare environment"
	@echo "  convert        - Convert PyTorch model to Core ML format"
	@echo "  convert-opt    - Convert and optimize model for mobile"
	@echo ""
	@echo "🔨 Development:"
	@echo "  build          - Build the iOS project"
	@echo "  run            - Build and run on simulator"
	@echo "  run-device     - Build and run on connected device"
	@echo "  test           - Run unit tests"
	@echo "  clean          - Clean build artifacts"
	@echo ""
	@echo "📱 Deployment:"
	@echo "  archive        - Create archive for distribution"
	@echo "  testflight     - Upload to TestFlight"
	@echo "  appstore       - Upload to App Store"
	@echo ""
	@echo "🔧 Utilities:"
	@echo "  open           - Open project in Xcode"
	@echo "  simulator      - Open iOS Simulator"
	@echo "  devices        - List connected devices"
	@echo "  logs           - Show device logs"

# Variables
MODEL_PATH ?= ../checkpoints/best_model.pth
MODEL_NAME ?= SignLanguageModel
PROJECT_NAME = SignLanguageDetector
SCHEME = SignLanguageDetector
CONFIGURATION ?= Release

# Setup environment
setup:
	@echo "🔧 Setting up iOS development environment..."
	@echo "Installing Python dependencies..."
	pip install coremltools
	@echo "✅ Setup complete!"

# Convert PyTorch model to Core ML
convert:
	@echo "🔄 Converting PyTorch model to Core ML..."
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "❌ Error: Model file not found at $(MODEL_PATH)"; \
		echo "Please train a model first or specify MODEL_PATH"; \
		exit 1; \
	fi
	python ../scripts/convert_to_coreml.py \
		--model-path "$(MODEL_PATH)" \
		--output-path "$(PROJECT_NAME)/$(MODEL_NAME).mlmodel"
	@echo "✅ Model converted successfully!"

# Convert and optimize model
convert-opt: convert
	@echo "⚡ Optimizing model for mobile deployment..."
	python ../scripts/convert_to_coreml.py \
		--model-path "$(MODEL_PATH)" \
		--output-path "$(PROJECT_NAME)/$(MODEL_NAME).mlmodel" \
		--optimize
	@echo "✅ Model converted and optimized!"

# Build the project
build:
	@echo "🔨 Building iOS project..."
	xcodebuild \
		-project "$(PROJECT_NAME).xcodeproj" \
		-scheme "$(SCHEME)" \
		-configuration "$(CONFIGURATION)" \
		build
	@echo "✅ Build complete!"

# Run on simulator
run:
	@echo "📱 Running on iOS Simulator..."
	xcodebuild \
		-project "$(PROJECT_NAME).xcodeproj" \
		-scheme "$(SCHEME)" \
		-destination "platform=iOS Simulator,name=iPhone 14" \
		build
	@echo "✅ App launched on simulator!"

# Run on connected device
run-device:
	@echo "📱 Running on connected device..."
	@DEVICE_ID=$$(xcrun xctrace list devices | grep "iPhone\|iPad" | head -1 | grep -o '\[.*\]' | tr -d '[]'); \
	if [ -z "$$DEVICE_ID" ]; then \
		echo "❌ No connected device found. Please connect an iPhone or iPad."; \
		exit 1; \
	fi; \
	echo "Using device: $$DEVICE_ID"; \
	xcodebuild \
		-project "$(PROJECT_NAME).xcodeproj" \
		-scheme "$(SCHEME)" \
		-destination "platform=iOS,id=$$DEVICE_ID" \
		build
	@echo "✅ App launched on device!"

# Run tests
test:
	@echo "🧪 Running unit tests..."
	xcodebuild test \
		-project "$(PROJECT_NAME).xcodeproj" \
		-scheme "$(SCHEME)" \
		-destination "platform=iOS Simulator,name=iPhone 14"
	@echo "✅ Tests complete!"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	xcodebuild clean \
		-project "$(PROJECT_NAME).xcodeproj" \
		-scheme "$(SCHEME)"
	rm -rf DerivedData
	@echo "✅ Clean complete!"

# Create archive for distribution
archive:
	@echo "📦 Creating archive for distribution..."
	xcodebuild archive \
		-project "$(PROJECT_NAME).xcodeproj" \
		-scheme "$(SCHEME)" \
		-configuration "$(CONFIGURATION)" \
		-archivePath "build/$(PROJECT_NAME).xcarchive"
	@echo "✅ Archive created at build/$(PROJECT_NAME).xcarchive"

# Upload to TestFlight
testflight: archive
	@echo "🚀 Uploading to TestFlight..."
	xcodebuild -exportArchive \
		-archivePath "build/$(PROJECT_NAME).xcarchive" \
		-exportPath "build/TestFlight" \
		-exportOptionsPlist "ExportOptions-TestFlight.plist"
	@echo "✅ Uploaded to TestFlight!"

# Upload to App Store
appstore: archive
	@echo "🏪 Uploading to App Store..."
	xcodebuild -exportArchive \
		-archivePath "build/$(PROJECT_NAME).xcarchive" \
		-exportPath "build/AppStore" \
		-exportOptionsPlist "ExportOptions-AppStore.plist"
	@echo "✅ Uploaded to App Store!"

# Open project in Xcode
open:
	@echo "🔧 Opening project in Xcode..."
	open "$(PROJECT_NAME).xcodeproj"

# Open iOS Simulator
simulator:
	@echo "📱 Opening iOS Simulator..."
	open -a Simulator

# List connected devices
devices:
	@echo "📱 Connected devices:"
	xcrun xctrace list devices

# Show device logs
logs:
	@echo "📋 Device logs (press Ctrl+C to stop):"
	xcrun simctl spawn booted log stream --predicate 'processImagePath contains "SignLanguageDetector"'

# Create export options plists
create-export-plists:
	@echo "📄 Creating export options plists..."
	@mkdir -p build
	@cat > build/ExportOptions-TestFlight.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>app-store</string>
    <key>teamID</key>
    <string>YOUR_TEAM_ID</string>
    <key>uploadBitcode</key>
    <false/>
    <key>uploadSymbols</key>
    <true/>
    <key>compileBitcode</key>
    <false/>
</dict>
</plist>
EOF
	@cat > build/ExportOptions-AppStore.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>app-store</string>
    <key>teamID</key>
    <string>YOUR_TEAM_ID</string>
    <key>uploadBitcode</key>
    <false/>
    <key>uploadSymbols</key>
    <true/>
    <key>compileBitcode</key>
    <false/>
</dict>
</plist>
EOF
	@echo "✅ Export options plists created!"
	@echo "⚠️  Remember to update YOUR_TEAM_ID in the plist files"

# Quick development workflow
dev: clean convert build run
	@echo "🚀 Development workflow complete!"

# Full deployment workflow
deploy: clean convert-opt build test archive
	@echo "🚀 Deployment workflow complete!"

# Show project info
info:
	@echo "📊 Project Information:"
	@echo "  Project: $(PROJECT_NAME)"
	@echo "  Scheme: $(SCHEME)"
	@echo "  Configuration: $(CONFIGURATION)"
	@echo "  Model Path: $(MODEL_PATH)"
	@echo "  Model Name: $(MODEL_NAME)"
	@echo ""
	@echo "📱 Available Simulators:"
	@xcrun simctl list devices | grep "iPhone\|iPad" | head -5
	@echo ""
	@echo "🔗 Connected Devices:"
	@xcrun xctrace list devices | grep "iPhone\|iPad" || echo "  No devices connected"
