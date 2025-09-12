import UIKit
import CoreML
import Vision

class SignLanguageModel {
    
    private var model: VNCoreMLModel?
    private let imageSize = CGSize(width: 224, height: 224)
    
    // ASL Alphabet classes (A-Z + space, del, nothing)
    private let classNames = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "space", "del", "nothing"
    ]
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        // Try to load the Core ML model
        guard let modelURL = Bundle.main.url(forResource: "SignLanguageModel", withExtension: "mlmodelc") else {
            print("Core ML model not found. Please add the converted model to the app bundle.")
            return
        }
        
        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            model = try VNCoreMLModel(for: mlModel)
            print("Core ML model loaded successfully")
        } catch {
            print("Failed to load Core ML model: \(error)")
        }
    }
    
    func predictSign(from image: UIImage, completion: @escaping (String?, Float) -> Void) {
        guard let model = model else {
            completion(nil, 0.0)
            return
        }
        
        // Preprocess the image
        guard let processedImage = preprocessImage(image) else {
            completion(nil, 0.0)
            return
        }
        
        // Create request
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self = self else { return }
            
            if let error = error {
                print("Prediction error: \(error)")
                completion(nil, 0.0)
                return
            }
            
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                completion(nil, 0.0)
                return
            }
            
            let prediction = self.classNames[topResult.identifier] ?? "Unknown"
            let confidence = topResult.confidence
            
            completion(prediction, confidence)
        }
        
        // Perform prediction
        let handler = VNImageRequestHandler(cgImage: processedImage.cgImage!, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform prediction: \(error)")
            completion(nil, 0.0)
        }
    }
    
    private func preprocessImage(_ image: UIImage) -> UIImage? {
        // Resize image to model input size
        let renderer = UIGraphicsImageRenderer(size: imageSize)
        let resizedImage = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: imageSize))
        }
        
        return resizedImage
    }
    
    // MARK: - Model Conversion Helper
    static func convertPyTorchToCoreML(pytorchModelPath: String, outputPath: String) {
        // This is a placeholder for the actual conversion process
        // In practice, you would use the conversion script provided
        print("Use the provided conversion script to convert your PyTorch model to Core ML format")
        print("PyTorch model path: \(pytorchModelPath)")
        print("Output path: \(outputPath)")
    }
}
