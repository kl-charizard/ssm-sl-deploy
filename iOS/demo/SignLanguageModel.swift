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
        // Try to load the optimized model first, then fallback to regular model
        if let optimizedModelURL = Bundle.main.url(forResource: "SignLanguageModel_optimized", withExtension: "mlmodel") {
            do {
                model = try MLModel(contentsOf: optimizedModelURL)
                print("✅ Loaded optimized Core ML model")
            } catch {
                print("❌ Failed to load optimized model: \(error)")
            }
        }
        
        if model == nil {
            if let modelURL = Bundle.main.url(forResource: "SignLanguageModel", withExtension: "mlmodel") {
                do {
                    model = try MLModel(contentsOf: modelURL)
                    print("✅ Loaded regular Core ML model")
                } catch {
                    print("❌ Failed to load model: \(error)")
                }
            }
        }
        
        if model == nil {
            print("❌ No Core ML model found in bundle")
        }
    }
    
    func predict(image: UIImage, completion: @escaping (String, Double) -> Void) {
        guard let model = model else {
            completion("Model not loaded", 0.0)
            return
        }
        
        // Resize image to 224x224 (model input size)
        guard let resizedImage = resizeImage(image, to: CGSize(width: 224, height: 224)) else {
            completion("Image resize failed", 0.0)
            return
        }
        
        // Convert UIImage to CVPixelBuffer
        guard let pixelBuffer = imageToPixelBuffer(resizedImage) else {
            completion("Pixel buffer conversion failed", 0.0)
            return
        }
        
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)])
            let prediction = try model.prediction(from: input)
            
            // Get the prediction results
            if let output = prediction.featureValue(for: "classLabel")?.stringValue,
               let confidence = prediction.featureValue(for: "classLabelProbs")?.multiArrayValue {
                
                let maxConfidence = getMaxConfidence(from: confidence)
                completion(output, maxConfidence)
            } else {
                completion("Prediction failed", 0.0)
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
    
    private func getMaxConfidence(from multiArray: MLMultiArray) -> Double {
        var maxConfidence: Double = 0.0
        for i in 0..<multiArray.count {
            let confidence = multiArray[i].doubleValue
            maxConfidence = max(maxConfidence, confidence)
        }
        return maxConfidence
    }
}
