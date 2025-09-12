import UIKit
import AVFoundation
import Vision
import CoreML

class CameraViewController: UIViewController {
    
    // MARK: - IBOutlets
    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var predictionLabel: UILabel!
    @IBOutlet weak var confidenceLabel: UILabel!
    @IBOutlet weak var handDetectionLabel: UILabel!
    @IBOutlet weak var closeButton: UIButton!
    
    // MARK: - Properties
    private var captureSession: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var videoOutput: AVCaptureVideoDataOutput!
    private var handDetectionHelper: HandDetectionHelper!
    private var signLanguageModel: SignLanguageModel!
    
    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupCamera()
        setupHandDetection()
        setupSignLanguageModel()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        startCamera()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        stopCamera()
    }
    
    // MARK: - Setup Methods
    private func setupUI() {
        predictionLabel.text = "Point your hand at the camera"
        predictionLabel.font = UIFont.boldSystemFont(ofSize: 24)
        predictionLabel.textAlignment = .center
        predictionLabel.textColor = .white
        predictionLabel.layer.shadowColor = UIColor.black.cgColor
        predictionLabel.layer.shadowOffset = CGSize(width: 1, height: 1)
        predictionLabel.layer.shadowOpacity = 0.8
        predictionLabel.layer.shadowRadius = 2
        
        confidenceLabel.text = "Confidence: --"
        confidenceLabel.font = UIFont.systemFont(ofSize: 16)
        confidenceLabel.textAlignment = .center
        confidenceLabel.textColor = .white
        confidenceLabel.layer.shadowColor = UIColor.black.cgColor
        confidenceLabel.layer.shadowOffset = CGSize(width: 1, height: 1)
        confidenceLabel.layer.shadowOpacity = 0.8
        confidenceLabel.layer.shadowRadius = 2
        
        handDetectionLabel.text = "Hand Detection: --"
        handDetectionLabel.font = UIFont.systemFont(ofSize: 14)
        handDetectionLabel.textAlignment = .center
        handDetectionLabel.textColor = .yellow
        handDetectionLabel.layer.shadowColor = UIColor.black.cgColor
        handDetectionLabel.layer.shadowOffset = CGSize(width: 1, height: 1)
        handDetectionLabel.layer.shadowOpacity = 0.8
        handDetectionLabel.layer.shadowRadius = 2
        
        closeButton.setTitle("✕", for: .normal)
        closeButton.titleLabel?.font = UIFont.systemFont(ofSize: 24)
        closeButton.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        closeButton.layer.cornerRadius = 20
        closeButton.setTitleColor(.white, for: .normal)
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high
        
        guard let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("Unable to access back camera")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: backCamera)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
        } catch {
            print("Error setting up camera input: \(error)")
            return
        }
        
        videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewView.layer.addSublayer(previewLayer)
    }
    
    private func setupHandDetection() {
        handDetectionHelper = HandDetectionHelper()
    }
    
    private func setupSignLanguageModel() {
        signLanguageModel = SignLanguageModel()
    }
    
    private func startCamera() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
    }
    
    private func stopCamera() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.stopRunning()
        }
    }
    
    // MARK: - Actions
    @IBAction func closeButtonTapped(_ sender: UIButton) {
        dismiss(animated: true)
    }
    
    // MARK: - Layout
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = previewView.bounds
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Perform hand detection
        handDetectionHelper.detectHands(in: pixelBuffer) { [weak self] handObservations in
            DispatchQueue.main.async {
                if let handObservations = handObservations, !handObservations.isEmpty {
                    self?.handDetectionLabel.text = "Hand Detection: ✓ (\(handObservations.count) hand(s))"
                    
                    // Get the first hand and crop the region
                    if let firstHand = handObservations.first {
                        self?.processHandRegion(pixelBuffer: pixelBuffer, handObservation: firstHand)
                    }
                } else {
                    self?.handDetectionLabel.text = "Hand Detection: ✗"
                    self?.predictionLabel.text = "No hand detected"
                    self?.confidenceLabel.text = "Confidence: --"
                }
            }
        }
    }
    
    private func processHandRegion(pixelBuffer: CVPixelBuffer, handObservation: VNHumanHandPoseObservation) {
        // Convert hand observation to image
        guard let handImage = handDetectionHelper.cropHandRegion(from: pixelBuffer, handObservation: handObservation) else {
            return
        }
        
        // Perform sign language prediction
        signLanguageModel.predictSign(from: handImage) { [weak self] prediction, confidence in
            DispatchQueue.main.async {
                if let prediction = prediction {
                    self?.predictionLabel.text = prediction
                    self?.confidenceLabel.text = String(format: "Confidence: %.1f%%", confidence * 100)
                } else {
                    self?.predictionLabel.text = "Unknown sign"
                    self?.confidenceLabel.text = "Confidence: --"
                }
            }
        }
    }
}
