//
//  CameraManager.swift
//  demo
//
//  Created by Kenny Lam on 12/9/25.
//

import Foundation
import AVFoundation
import Vision
import CoreML
import UIKit
import SwiftUI
import Combine

class CameraManager: NSObject, ObservableObject {
    @Published var isAuthorized = false
    @Published var error: CameraError?
    @Published var detectedSign = "No sign detected"
    @Published var confidence = 0.0
    
    var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var handDetectionHelper: HandDetectionHelper?
    private var signLanguageModel: SignLanguageModel?
    
    enum CameraError: Error {
        case unauthorized
        case unavailable
        case setupFailed
    }
    
    override init() {
        super.init()
        checkAuthorization()
        setupHandDetection()
        setupSignLanguageModel()
    }
    
    private func checkAuthorization() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isAuthorized = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.isAuthorized = granted
                    if !granted {
                        self?.error = .unauthorized
                    }
                }
            }
        case .denied, .restricted:
            error = .unauthorized
        @unknown default:
            error = .unavailable
        }
    }
    
    private func setupHandDetection() {
        handDetectionHelper = HandDetectionHelper()
    }
    
    private func setupSignLanguageModel() {
        signLanguageModel = SignLanguageModel()
    }
    
    func startSession() {
        guard isAuthorized else { return }
        
        captureSession = AVCaptureSession()
        guard let captureSession = captureSession else { return }
        
        captureSession.beginConfiguration()
        
        // Add video input
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else {
            self.error = .setupFailed
            return
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }
        
        // Add video output
        videoOutput = AVCaptureVideoDataOutput()
        guard let videoOutput = videoOutput else { return }
        
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        captureSession.commitConfiguration()
        captureSession.startRunning()
    }
    
    func stopSession() {
        captureSession?.stopRunning()
        captureSession = nil
        videoOutput = nil
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { 
            print("❌ Failed to get pixel buffer from sample buffer")
            return 
        }
        
        print("📸 Camera frame captured, processing...")
        
        // Detect hands and crop
        handDetectionHelper?.detectHands(in: pixelBuffer) { [weak self] croppedImage in
            guard let self = self else { return }
            
            if let croppedImage = croppedImage {
                print("✋ Hand detected, using cropped image for detection")
                // Run sign language detection
                self.signLanguageModel?.predict(image: croppedImage) { [weak self] prediction, confidence in
                    print("🎯 Prediction: \(prediction), Confidence: \(confidence)")
                    DispatchQueue.main.async {
                        self?.detectedSign = prediction
                        self?.confidence = confidence
                    }
                }
            } else {
                print("❌ No hand detected, skipping detection")
            }
        }
    }
}

// Camera Preview View
struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager
    
    func makeUIView(context: UIViewRepresentableContext<CameraPreviewView>) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        
        guard let captureSession = cameraManager.captureSession else { return view }
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.frame
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: UIViewRepresentableContext<CameraPreviewView>) {
        // Update if needed
    }
}
