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
            
            // Convert CVPixelBuffer to UIImage more reliably
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            print("🖼️ Created CIImage with extent: \(ciImage.extent)")
            
            let context = CIContext()
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                print("❌ Failed to create CGImage from CIImage")
                completion(nil)
                return
            }
            print("✅ Successfully created CGImage: \(cgImage.width)x\(cgImage.height)")
            
            let image = UIImage(cgImage: cgImage)
            print("✅ Successfully created UIImage from CGImage")
            
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
        print("📐 Image size: \(imageSize)")
        
        // Get all available keypoints
        var allPoints: [CGPoint] = []
        
        // Try to get keypoints for different hand parts
        if let wrist = try? observation.recognizedPoint(.wrist) {
            let point = CGPoint(x: wrist.location.x * imageSize.width, y: wrist.location.y * imageSize.height)
            allPoints.append(point)
            print("✋ Wrist point: \(point)")
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
                let cgPoint = CGPoint(x: point.location.x * imageSize.width, y: point.location.y * imageSize.height)
                allPoints.append(cgPoint)
            }
        }
        
        print("📊 Total keypoints found: \(allPoints.count)")
        
        guard !allPoints.isEmpty else {
            print("⚠️ No keypoints found, using fallback center region")
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
        
        let boundingBox = CGRect(x: minX, y: minY, width: width, height: height)
        print("📦 Calculated bounding box: \(boundingBox)")
        
        return boundingBox
    }
    
    private func cropHandRegion(from image: UIImage, boundingBox: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage else { 
            print("❌ Failed to get CGImage from UIImage")
            return nil 
        }
        
        print("🖼️ Original image size: \(cgImage.width)x\(cgImage.height)")
        print("📦 Bounding box to crop: \(boundingBox)")
        
        // The bounding box is already in pixel coordinates, so we can use it directly
        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        
        // Ensure bounding box is within image bounds
        let clampedX = max(0, min(boundingBox.origin.x, imageWidth - boundingBox.width))
        let clampedY = max(0, min(boundingBox.origin.y, imageHeight - boundingBox.height))
        let clampedWidth = min(boundingBox.width, imageWidth - clampedX)
        let clampedHeight = min(boundingBox.height, imageHeight - clampedY)
        
        let clampedRect = CGRect(x: clampedX, y: clampedY, width: clampedWidth, height: clampedHeight)
        print("🔧 Clamped rect: \(clampedRect)")
        
        // Ensure the rect has valid dimensions
        guard clampedRect.width > 0 && clampedRect.height > 0 else {
            print("❌ Invalid rect dimensions: width=\(clampedRect.width), height=\(clampedRect.height)")
            return nil
        }
        
        // Add some padding around the hand
        let padding: CGFloat = 20
        let paddedX = max(0, clampedX - padding)
        let paddedY = max(0, clampedY - padding)
        let paddedWidth = min(imageWidth - paddedX, clampedWidth + 2 * padding)
        let paddedHeight = min(imageHeight - paddedY, clampedHeight + 2 * padding)
        
        let paddedRect = CGRect(x: paddedX, y: paddedY, width: paddedWidth, height: paddedHeight)
        print("🔧 Padded rect: \(paddedRect)")
        
        guard let croppedCgImage = cgImage.cropping(to: paddedRect) else { 
            print("❌ Failed to crop CGImage")
            return nil 
        }
        
        print("✅ Successfully cropped image to: \(croppedCgImage.width)x\(croppedCgImage.height)")
        return UIImage(cgImage: croppedCgImage)
    }
}
