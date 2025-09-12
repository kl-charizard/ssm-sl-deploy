//
//  SignLanguageModel.swift
//  demo
//
//  Created by Kenny Lam on 12/9/25.
//

import Foundation
import CoreML
import UIKit
import Vision

class SignLanguageModel {
    private var model: MLModel?
    private let aslAlphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        print("🔍 Searching for Core ML models in bundle...")
        
        // List all available resources for debugging
        if let resourcePath = Bundle.main.resourcePath {
            print("📁 Bundle resource path: \(resourcePath)")
            do {
                let contents = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
                let mlmodelFiles = contents.filter { $0.hasSuffix(".mlmodel") }
                print("📄 Found .mlmodel files: \(mlmodelFiles)")
            } catch {
                print("❌ Error listing bundle contents: \(error)")
            }
        }
        
        // Try to load the optimized model first, then fallback to regular model
        // Xcode compiles .mlmodel files into .mlmodelc directories
        if let optimizedModelURL = Bundle.main.url(forResource: "SignLanguageModel_optimized", withExtension: "mlmodelc") {
            print("🎯 Found optimized model at: \(optimizedModelURL)")
            do {
                model = try MLModel(contentsOf: optimizedModelURL)
                print("✅ Loaded optimized Core ML model")
                return
            } catch {
                print("❌ Failed to load optimized model: \(error)")
            }
        } else {
            print("❌ Optimized model not found in bundle")
        }
        
        if let modelURL = Bundle.main.url(forResource: "SignLanguageModel", withExtension: "mlmodelc") {
            print("🎯 Found regular model at: \(modelURL)")
            do {
                model = try MLModel(contentsOf: modelURL)
                print("✅ Loaded regular Core ML model")
                return
            } catch {
                print("❌ Failed to load regular model: \(error)")
            }
        } else {
            print("❌ Regular model not found in bundle")
        }
        
        print("❌ No Core ML model found in bundle")
        print("💡 Make sure to run ./setup_models.sh to convert and add models")
    }
    
    func predict(image: UIImage, completion: @escaping (String, Double) -> Void) {
        guard let model = model else {
            print("❌ Model not loaded for prediction")
            completion("Model not loaded", 0.0)
            return
        }
        
        print("🔍 Starting prediction on image...")
        
        // Resize image to 224x224 (model input size)
        guard let resizedImage = resizeImage(image, to: CGSize(width: 224, height: 224)) else {
            print("❌ Failed to resize image")
            completion("Image resize failed", 0.0)
            return
        }
        
        print("✅ Image resized to 224x224")
        
        // Convert UIImage to CVPixelBuffer
        guard let pixelBuffer = imageToPixelBuffer(resizedImage) else {
            print("❌ Failed to convert image to pixel buffer")
            completion("Pixel buffer conversion failed", 0.0)
            return
        }
        
        print("✅ Image converted to pixel buffer, running inference...")
        
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)])
            let prediction = try model.prediction(from: input)
            
            print("🔍 Model prediction output keys: \(prediction.featureNames)")
            
            // Debug: Print all feature values
            for key in prediction.featureNames {
                if let value = prediction.featureValue(for: key) {
                    print("🔍 Key '\(key)': \(value)")
                }
            }
            
            // Get the prediction results - try different possible output names
            var output: String?
            var confidence: MLMultiArray?
            
            // Try different possible output names
            let possibleOutputNames = ["classLabel", "label", "output", "prediction"]
            let possibleConfidenceNames = ["classLabelProbs", "probabilities", "confidence", "probs"]
            
            for outputName in possibleOutputNames {
                if let multiArray = prediction.featureValue(for: outputName)?.multiArrayValue {
                    // Handle MultiArray output (raw logits)
                    print("✅ Found MultiArray output in key: \(outputName)")
                    let (predictedClass, confidenceValue) = getPredictionFromMultiArray(multiArray)
                    output = predictedClass
                    confidence = multiArray
                    print("✅ Extracted prediction: \(predictedClass) with confidence: \(confidenceValue)")
                    break
                } else if let value = prediction.featureValue(for: outputName)?.stringValue {
                    output = value
                    print("✅ Found output in key: \(outputName) = '\(value)'")
                    break
                }
            }
            
            for confName in possibleConfidenceNames {
                if let value = prediction.featureValue(for: confName)?.multiArrayValue {
                    confidence = value
                    print("✅ Found confidence in key: \(confName)")
                    break
                }
            }
            
            if let output = output, !output.isEmpty {
                let maxConfidence = confidence != nil ? getMaxConfidence(from: confidence!) : 0.5
                print("🎯 Model prediction: \(output) (confidence: \(maxConfidence))")
                completion(output, maxConfidence)
            } else {
                print("❌ Failed to extract prediction results")
                print("❌ Available keys: \(prediction.featureNames)")
                print("❌ Output found: \(output != nil)")
                print("❌ Output empty: \(output?.isEmpty ?? true)")
                print("❌ Confidence found: \(confidence != nil)")
                
                // Try to get any available output as fallback
                if let fallbackOutput = prediction.featureValue(for: "classLabel")?.stringValue {
                    print("🔄 Fallback output: '\(fallbackOutput)'")
                    completion(fallbackOutput.isEmpty ? "Unknown" : fallbackOutput, 0.3)
                } else {
                    completion("Prediction failed", 0.0)
                }
            }
            
        } catch {
            print("❌ Prediction error: \(error)")
            completion("Prediction error", 0.0)
        }
    }
    
    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    private func imageToPixelBuffer(_ image: UIImage) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, 224, 224, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: 224, height: 224, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: 224)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        image.draw(in: CGRect(x: 0, y: 0, width: 224, height: 224))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
    
    private func getPredictionFromMultiArray(_ multiArray: MLMultiArray) -> (String, Double) {
        var maxIndex = 0
        var maxValue = multiArray[0].doubleValue
        
        // Find the index with the highest value
        for i in 1..<multiArray.count {
            let value = multiArray[i].doubleValue
            if value > maxValue {
                maxValue = value
                maxIndex = i
            }
        }
        
        // Convert index to ASL letter
        let predictedLetter = aslAlphabet[maxIndex]
        
        // Convert logits to probability using softmax
        let probabilities = softmax(multiArray)
        let confidence = probabilities[maxIndex]
        
        // Debug: Print top 5 predictions
        var topPredictions: [(String, Double)] = []
        for i in 0..<min(5, multiArray.count) {
            let letter = aslAlphabet[i]
            let prob = probabilities[i]
            topPredictions.append((letter, prob))
        }
        topPredictions.sort { $0.1 > $1.1 }
        
        print("🔍 Top 5 predictions: \(topPredictions.map { "\($0.0): \(String(format: "%.3f", $0.1))" }.joined(separator: ", "))")
        print("🔍 Selected: \(predictedLetter) (index: \(maxIndex), confidence: \(String(format: "%.3f", confidence)))")
        
        return (predictedLetter, confidence)
    }
    
    private func softmax(_ multiArray: MLMultiArray) -> [Double] {
        // Find max value for numerical stability
        var maxValue = multiArray[0].doubleValue
        for i in 1..<multiArray.count {
            maxValue = max(maxValue, multiArray[i].doubleValue)
        }
        
        // Calculate exponentials
        var expValues: [Double] = []
        var sumExp = 0.0
        
        for i in 0..<multiArray.count {
            let expValue = exp(multiArray[i].doubleValue - maxValue)
            expValues.append(expValue)
            sumExp += expValue
        }
        
        // Normalize to get probabilities
        return expValues.map { $0 / sumExp }
    }
    
    private func getMaxConfidence(from multiArray: MLMultiArray) -> Double {
        var maxConfidence: Double = 0.0
        for i in 0..<multiArray.count {
            let confidence = multiArray[i].doubleValue
            maxConfidence = max(maxConfidence, confidence)
        }
        return maxConfidence
    }
}
