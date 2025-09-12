//
//  ContentView.swift
//  demo
//
//  Created by Kenny Lam on 12/9/25.
//

import SwiftUI
import AVFoundation
import Vision
import CoreML

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var detectedSign = "No sign detected"
    @State private var confidence = 0.0
    @State private var showingCamera = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 10) {
                    Image(systemName: "hand.raised.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.blue)
                    
                    Text("Sign Language Detector")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Real-time ASL alphabet recognition")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 20)
                
                // Detection Display
                VStack(spacing: 15) {
                    Text("Detected Sign")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Text(detectedSign)
                        .font(.system(size: 80, weight: .bold, design: .rounded))
                        .foregroundColor(.blue)
                        .frame(height: 100)
                    
                    Text("Confidence: \(Int(confidence * 100))%")
                        .font(.title2)
                        .foregroundColor(confidence > 0.7 ? .green : .orange)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 8)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(20)
                }
                .padding()
                .background(Color.gray.opacity(0.05))
                .cornerRadius(20)
                
                // Camera Button
                Button(action: {
                    showingCamera = true
                }) {
                    HStack {
                        Image(systemName: "camera.fill")
                        Text("Start Camera Detection")
                    }
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .padding(.horizontal, 30)
                    .padding(.vertical, 15)
                    .background(
                        LinearGradient(
                            gradient: Gradient(colors: [.blue, .purple]),
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(25)
                    .shadow(color: .blue.opacity(0.3), radius: 10, x: 0, y: 5)
                }
                
                // Info Section
                VStack(spacing: 10) {
                    Text("Supported Signs")
                        .font(.headline)
                    
                    Text("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z")
                        .font(.title3)
                        .fontWeight(.medium)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding()
                .background(Color.gray.opacity(0.05))
                .cornerRadius(15)
                
                Spacer()
            }
            .padding()
            .onAppear {
                ModelTest.testModelLoading()
            }
            .sheet(isPresented: $showingCamera) {
                CameraView(
                    detectedSign: $detectedSign,
                    confidence: $confidence
                )
            }
        }
    }
}

struct CameraView: View {
    @Binding var detectedSign: String
    @Binding var confidence: Double
    @Environment(\.dismiss) var dismiss
    @StateObject private var cameraManager = CameraManager()
    
    var body: some View {
        NavigationView {
            ZStack {
                // Camera Preview
                CameraPreviewView(cameraManager: cameraManager)
                    .ignoresSafeArea()
                
                // Overlay UI
                VStack {
                    // Top Bar
                    HStack {
                        Button("Close") {
                            dismiss()
                        }
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(10)
                        
                        Spacer()
                        
                        Text("Sign Detection")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(10)
                    }
                    .padding()
                    
                    Spacer()
                    
                    // Detection Results
                    VStack(spacing: 15) {
                        Text(detectedSign)
                            .font(.system(size: 60, weight: .bold, design: .rounded))
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(20)
                        
                        Text("Confidence: \(Int(confidence * 100))%")
                            .font(.title2)
                            .foregroundColor(.white)
                            .padding(.horizontal, 20)
                            .padding(.vertical, 8)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(15)
                    }
                    .padding()
                }
            }
            .navigationBarHidden(true)
        }
        .onAppear {
            cameraManager.startSession()
        }
        .onDisappear {
            cameraManager.stopSession()
        }
    }
}

#Preview {
    ContentView()
}