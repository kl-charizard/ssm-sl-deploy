import UIKit
import Vision
import CoreImage

class HandDetectionHelper {
    
    private let handPoseRequest: VNDetectHumanHandPoseRequest
    private let imageRequestHandler: VNImageRequestHandler
    
    init() {
        handPoseRequest = VNDetectHumanHandPoseRequest()
        handPoseRequest.maximumHandCount = 2
        imageRequestHandler = VNImageRequestHandler()
    }
    
    func detectHands(in pixelBuffer: CVPixelBuffer, completion: @escaping ([VNHumanHandPoseObservation]?) -> Void) {
        let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        
        do {
            try requestHandler.perform([handPoseRequest])
            completion(handPoseRequest.results)
        } catch {
            print("Hand detection failed: \(error)")
            completion(nil)
        }
    }
    
    func cropHandRegion(from pixelBuffer: CVPixelBuffer, handObservation: VNHumanHandPoseObservation) -> UIImage? {
        // Get hand bounding box
        guard let boundingBox = try? handObservation.boundingBox(for: .all) else {
            return nil
        }
        
        // Convert Vision coordinates to Core Image coordinates
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let imageSize = ciImage.extent.size
        
        // Calculate crop rectangle with padding
        let padding: CGFloat = 0.2
        let width = boundingBox.width * imageSize.width
        let height = boundingBox.height * imageSize.height
        let x = (boundingBox.origin.x * imageSize.width) - (width * padding)
        let y = (1 - boundingBox.origin.y - boundingBox.height) * imageSize.height - (height * padding)
        
        let paddedWidth = width * (1 + 2 * padding)
        let paddedHeight = height * (1 + 2 * padding)
        
        let cropRect = CGRect(
            x: max(0, x),
            y: max(0, y),
            width: min(paddedWidth, imageSize.width - max(0, x)),
            height: min(paddedHeight, imageSize.height - max(0, y))
        )
        
        // Crop the image
        let croppedImage = ciImage.cropped(to: cropRect)
        
        // Convert to UIImage
        let context = CIContext()
        guard let cgImage = context.createCGImage(croppedImage, from: croppedImage.extent) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
    
    func drawHandLandmarks(on image: UIImage, handObservations: [VNHumanHandPoseObservation]) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        guard let context = UIGraphicsGetCurrentContext() else { return nil }
        
        image.draw(in: CGRect(origin: .zero, size: image.size))
        
        context.setStrokeColor(UIColor.green.cgColor)
        context.setLineWidth(3.0)
        
        for handObservation in handObservations {
            guard let allPoints = try? handObservation.availableJointsGroupNames.compactMap({ groupName in
                try? handObservation.recognizedPoints(groupName)
            }).flatMap({ $0 }) else { continue }
            
            for (_, point) in allPoints {
                let x = point.location.x * image.size.width
                let y = (1 - point.location.y) * image.size.height
                
                context.addEllipse(in: CGRect(x: x - 3, y: y - 3, width: 6, height: 6))
                context.strokePath()
            }
        }
        
        let result = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return result
    }
}
