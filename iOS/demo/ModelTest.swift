//
//  ModelTest.swift
//  demo
//
//  Created by Kenny Lam on 12/9/25.
//

import Foundation
import CoreML

class ModelTest {
    static func testModelLoading() {
        print("🔍 Testing Core ML model loading...")
        
        // List all available resources
        if let resourcePath = Bundle.main.resourcePath {
            print("📁 Bundle resource path: \(resourcePath)")
            do {
                let contents = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
                print("📄 All files in bundle: \(contents)")
                let mlmodelFiles = contents.filter { $0.hasSuffix(".mlmodel") }
                print("📄 Found .mlmodel files: \(mlmodelFiles)")
            } catch {
                print("❌ Error listing bundle contents: \(error)")
            }
        }
        
        // Try to find models (Xcode compiles .mlmodel to .mlmodelc)
        let modelNames = ["SignLanguageModel", "SignLanguageModel_optimized"]
        
        for modelName in modelNames {
            // Try .mlmodelc first (compiled version)
            if let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") {
                print("✅ Found \(modelName) (compiled) at: \(modelURL)")
                
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    print("✅ Successfully loaded \(modelName)")
                    print("📊 Model description: \(model.modelDescription)")
                } catch {
                    print("❌ Failed to load \(modelName): \(error)")
                }
            } else if let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodel") {
                print("✅ Found \(modelName) (source) at: \(modelURL)")
                
                do {
                    let model = try MLModel(contentsOf: modelURL)
                    print("✅ Successfully loaded \(modelName)")
                    print("📊 Model description: \(model.modelDescription)")
                } catch {
                    print("❌ Failed to load \(modelName): \(error)")
                }
            } else {
                print("❌ \(modelName) not found in bundle")
            }
        }
    }
}
