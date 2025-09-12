//
//  HandDetectionHelper.swift
//  demo
//
//  Created by Kenny Lam on 12/9/25.
//

import Foundation
import Vision
import CoreML
import UIKit

class HandDetectionHelper {
    private let handPoseRequest = VNDetectHumanHandPoseRequest()
    
    init() {
        handPoseRequest.maximumHandCount = 1
    }
    
    func detectHands(in pixelBuffer: CVPixelBuffer, completion: @escaping (UIImage?) -> Void) {
        let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        
        do {
            try requestHandler.perform([handPoseRequest])
            
            guard let observation = handPoseRequest.results?.first else {
                print("❌ No hand pose detected in frame")
                completion(nil)
                return
            }
            
            print("✋ Hand pose detected, extracting bounding box...")
            
            // Extract hand bounding box from keypoints
            let boundingBox = getHandBoundingBox(from: observation, in: pixelBuffer)
            let image = UIImage(ciImage: CIImage(cvPixelBuffer: pixelBuffer))
            
            // Crop hand region
            let croppedImage = cropHandRegion(from: image, boundingBox: boundingBox)
            
            if croppedImage != nil {
                print("✅ Hand region cropped successfully")
            } else {
                print("❌ Failed to crop hand region")
            }
            
            completion(croppedImage)
            
        } catch {
            print("❌ Hand detection error: \(error)")
            completion(nil)
        }
    }
    
    private func getHandBoundingBox(from observation: VNHumanHandPoseObservation, in pixelBuffer: CVPixelBuffer) -> CGRect {
        let imageSize = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        
        // Get all available keypoints
        var allPoints: [CGPoint] = []
        
        // Try to get keypoints for different hand parts
        if let wrist = try? observation.recognizedPoint(.wrist) {
            allPoints.append(CGPoint(x: wrist.location.x * imageSize.width, y: wrist.location.y * imageSize.height))
        }
        
        // Add finger keypoints
        let fingerJoints: [VNHumanHandPoseObservation.JointName] = [
            .thumbTip, .thumbIP, .thumbMP, .thumbCMC,
            .indexTip, .indexDIP, .indexPIP, .indexMCP,
            .middleTip, .middleDIP, .middlePIP, .middleMCP,
            .ringTip, .ringDIP, .ringPIP, .ringMCP,
            .littleTip, .littleDIP, .littlePIP, .littleMCP
        ]
        
        for joint in fingerJoints {
            if let point = try? observation.recognizedPoint(joint) {
                allPoints.append(CGPoint(x: point.location.x * imageSize.width, y: point.location.y * imageSize.height))
            }
        }
        
        guard !allPoints.isEmpty else {
            // Fallback to center region if no keypoints found
            return CGRect(x: imageSize.width * 0.25, y: imageSize.height * 0.25, width: imageSize.width * 0.5, height: imageSize.height * 0.5)
        }
        
        // Calculate bounding box from keypoints
        let minX = allPoints.map { $0.x }.min() ?? 0
        let maxX = allPoints.map { $0.x }.max() ?? imageSize.width
        let minY = allPoints.map { $0.y }.min() ?? 0
        let maxY = allPoints.map { $0.y }.max() ?? imageSize.height
        
        let width = maxX - minX
        let height = maxY - minY
        
        return CGRect(x: minX, y: minY, width: width, height: height)
    }
    
    private func cropHandRegion(from image: UIImage, boundingBox: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        // Convert Vision coordinates to Core Graphics coordinates
        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)
        
        let x = boundingBox.origin.x * width
        let y = (1 - boundingBox.origin.y - boundingBox.height) * height
        let cropWidth = boundingBox.width * width
        let cropHeight = boundingBox.height * height
        
        // Add padding around the hand
        let padding: CGFloat = 0.2
        let paddedX = max(0, x - cropWidth * padding)
        let paddedY = max(0, y - cropHeight * padding)
        let paddedWidth = min(width - paddedX, cropWidth * (1 + 2 * padding))
        let paddedHeight = min(height - paddedY, cropHeight * (1 + 2 * padding))
        
        let cropRect = CGRect(x: paddedX, y: paddedY, width: paddedWidth, height: paddedHeight)
        
        guard let croppedCGImage = cgImage.cropping(to: cropRect) else { return nil }
        
        return UIImage(cgImage: croppedCGImage)
    }
}
