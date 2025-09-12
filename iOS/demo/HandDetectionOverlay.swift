//
//  HandDetectionOverlay.swift
//  demo
//
//  Created by Kenny Lam on 12/9/25.
//

import SwiftUI
import Vision
import CoreMedia

struct HandDetectionOverlay: View {
    let handObservations: [VNHumanHandPoseObservation]
    let imageSize: CGSize
    
    var body: some View {
        ZStack {
            // Test overlay to verify drawing works
            Rectangle()
                .stroke(Color.red, lineWidth: 5)
                .frame(width: 100, height: 100)
                .position(x: 100, y: 100)
            
            ForEach(Array(handObservations.enumerated()), id: \.offset) { index, observation in
                // Draw bounding box
                HandBoundingBoxView(observation: observation, imageSize: imageSize)
                
                // Draw keypoints
                HandKeypointsView(observation: observation, imageSize: imageSize)
            }
        }
        .onAppear {
            print("🎨 HandDetectionOverlay appeared with \(handObservations.count) observations")
            print("🎨 Image size: \(imageSize)")
        }
        .onChange(of: handObservations.count) { count in
            print("🎨 Hand observations changed: \(count)")
        }
    }
}

struct HandBoundingBoxView: View {
    let observation: VNHumanHandPoseObservation
    let imageSize: CGSize
    
    var body: some View {
        let boundingBox = getHandBoundingBox(from: observation, in: imageSize)
        let screenSize = UIScreen.main.bounds.size
        
        // Scale coordinates from image size to screen size
        let scaleX = screenSize.width / imageSize.width
        let scaleY = screenSize.height / imageSize.height
        
        let scaledBox = CGRect(
            x: boundingBox.origin.x * scaleX,
            y: boundingBox.origin.y * scaleY,
            width: boundingBox.width * scaleX,
            height: boundingBox.height * scaleY
        )
        
        Rectangle()
            .stroke(Color.green, lineWidth: 3)
            .frame(width: scaledBox.width, height: scaledBox.height)
            .position(x: scaledBox.midX, y: scaledBox.midY)
            .onAppear {
                print("🎨 Drawing bounding box: \(scaledBox)")
                print("🎨 Screen size: \(screenSize), Image size: \(imageSize)")
                print("🎨 Scale: \(scaleX), \(scaleY)")
            }
    }
    
    private func getHandBoundingBox(from observation: VNHumanHandPoseObservation, in imageSize: CGSize) -> CGRect {
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
}

struct HandKeypointsView: View {
    let observation: VNHumanHandPoseObservation
    let imageSize: CGSize
    
    var body: some View {
        let screenSize = UIScreen.main.bounds.size
        let scaleX = screenSize.width / imageSize.width
        let scaleY = screenSize.height / imageSize.height
        
        ZStack {
            // Draw wrist
            if let wrist = try? observation.recognizedPoint(.wrist) {
                let x = wrist.location.x * imageSize.width * scaleX
                let y = wrist.location.y * imageSize.height * scaleY
                
                Circle()
                    .fill(Color.red)
                    .frame(width: 12, height: 12)
                    .position(x: x, y: y)
                    .onAppear {
                        print("🎨 Drawing wrist at: \(x), \(y)")
                    }
            }
            
            // Draw finger tips
            let fingerTips: [VNHumanHandPoseObservation.JointName] = [
                .thumbTip, .indexTip, .middleTip, .ringTip, .littleTip
            ]
            
            ForEach(Array(fingerTips.enumerated()), id: \.offset) { index, joint in
                if let point = try? observation.recognizedPoint(joint) {
                    let x = point.location.x * imageSize.width * scaleX
                    let y = point.location.y * imageSize.height * scaleY
                    
                    Circle()
                        .fill(Color.blue)
                        .frame(width: 8, height: 8)
                        .position(x: x, y: y)
                        .onAppear {
                            print("🎨 Drawing fingertip \(index) at: \(x), \(y)")
                        }
                }
            }
        }
    }
}

#Preview {
    HandDetectionOverlay(handObservations: [], imageSize: CGSize(width: 1920, height: 1080))
}
