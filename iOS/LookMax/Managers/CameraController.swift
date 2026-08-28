import Foundation
import AVFoundation
import UIKit
import Combine
import Vision

class CameraController: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var session = AVCaptureSession()
    @Published var isSessionRunning = false
    @Published var isCountdownActive = false
    @Published var countdownRemaining = 0
    @Published var selectedTimerDuration: Int = 0
    @Published var flashMode: AVCaptureDevice.FlashMode = .auto
    @Published var cameraPosition: AVCaptureDevice.Position = .front
    @Published var isGridVisible = false
    @Published var currentZoomFactor: CGFloat = 1.0
    @Published var focusPoint: CGPoint?
    @Published var isFocusing = false

    // Real-time AR Biometric Overlays
    @Published var spineStartPoint: CGPoint? = nil
    @Published var spineEndPoint: CGPoint? = nil
    @Published var faceLandmarkPoints: [CGPoint] = []
    @Published var isPostureAligned: Bool = true
    @Published var biometricStatusText: String = "Biometrics Tracking"

    private var photoOutput = AVCapturePhotoOutput()
    private var videoDataOutput = AVCaptureVideoDataOutput()
    private var currentDeviceInput: AVCaptureDeviceInput?
    private var countdownTimer: Timer?
    private let visionQueue = DispatchQueue(label: "com.lookmax.visionQueue", qos: .userInteractive)
    private var isProcessingFrame = false

    var onPhotoCaptured: ((UIImage) -> Void)?

    var isUltraWideAvailable: Bool { minZoomFactor < 0.95 }
    var isTelephotoAvailable: Bool { maxZoomFactor >= 2.0 }

    func setupCamera() {
        guard !session.isRunning else { return }
        session.beginConfiguration()
        session.sessionPreset = .high
        guard let device = getCameraDevice(for: cameraPosition) else {
            session.commitConfiguration(); return
        }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) { session.addInput(input); currentDeviceInput = input }
            if session.canAddOutput(photoOutput) { session.addOutput(photoOutput) }
            
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.setSampleBufferDelegate(self, queue: visionQueue)
            if session.canAddOutput(videoDataOutput) {
                session.addOutput(videoDataOutput)
            }
        } catch { print("Camera error: \(error)") }
        session.commitConfiguration()
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async { self?.isSessionRunning = self?.session.isRunning ?? false }
        }
    }

    func stopCamera() {
        cancelCountdown()
        guard session.isRunning else { return }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.stopRunning()
            DispatchQueue.main.async { self?.isSessionRunning = false }
        }
    }

    func switchCamera() {
        session.beginConfiguration()
        if let cur = currentDeviceInput { session.removeInput(cur) }
        let newPos: AVCaptureDevice.Position = cameraPosition == .back ? .front : .back
        guard let dev = getCameraDevice(for: newPos) else {
            if let cur = currentDeviceInput { session.addInput(cur) }
            session.commitConfiguration(); return
        }
        do {
            let inp = try AVCaptureDeviceInput(device: dev)
            if session.canAddInput(inp) {
                session.addInput(inp)
                currentDeviceInput = inp
                cameraPosition = newPos
                currentZoomFactor = 1.0
            }
        } catch { print("Switch camera error: \(error)") }
        session.commitConfiguration()
    }

    private func getCameraDevice(for pos: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let types: [AVCaptureDevice.DeviceType] = pos == .back
            ? [.builtInTripleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInUltraWideCamera, .builtInWideAngleCamera]
            : [.builtInTrueDepthCamera, .builtInWideAngleCamera]
        return AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .video, position: pos).devices.first
    }

    var minZoomFactor: CGFloat { currentDeviceInput?.device.minAvailableVideoZoomFactor ?? 1.0 }
    var maxZoomFactor: CGFloat { min(currentDeviceInput?.device.activeFormat.videoMaxZoomFactor ?? 5.0, 10.0) }

    func toggleFlash() {
        switch flashMode {
        case .auto:  flashMode = .on
        case .on:    flashMode = .off
        case .off:   flashMode = .auto
        @unknown default: flashMode = .auto
        }
    }

    func setZoom(_ factor: CGFloat) {
        guard let device = currentDeviceInput?.device else { return }
        do {
            try device.lockForConfiguration()
            let clamp = max(device.minAvailableVideoZoomFactor,
                            min(factor, min(device.activeFormat.videoMaxZoomFactor, 10.0)))
            device.videoZoomFactor = clamp
            currentZoomFactor = clamp
            device.unlockForConfiguration()
        } catch { print("Zoom error: \(error)") }
    }

    func focus(at point: CGPoint, viewSize: CGSize) {
        guard let device = currentDeviceInput?.device else { return }
        let fp = CGPoint(x: point.y / viewSize.height, y: 1.0 - (point.x / viewSize.width))
        do {
            try device.lockForConfiguration()
            if device.isFocusPointOfInterestSupported && device.isFocusModeSupported(.autoFocus) {
                device.focusPointOfInterest = fp
                device.focusMode = .autoFocus
            }
            if device.isExposurePointOfInterestSupported && device.isExposureModeSupported(.autoExpose) {
                device.exposurePointOfInterest = fp
                device.exposureMode = .autoExpose
            }
            device.unlockForConfiguration()
            focusPoint = point
            isFocusing = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { self.isFocusing = false }
        } catch { print("Focus error: \(error)") }
    }

    func triggerCapture() {
        selectedTimerDuration > 0 ? startCountdown(seconds: selectedTimerDuration) : performCapture()
    }

    private func startCountdown(seconds: Int) {
        countdownRemaining = seconds
        isCountdownActive = true
        HapticManager.medium()
        countdownTimer?.invalidate()
        countdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] t in
            guard let self else { return }
            self.countdownRemaining -= 1
            HapticManager.light()
            if self.countdownRemaining <= 0 {
                t.invalidate()
                self.isCountdownActive = false
                self.performCapture()
            }
        }
    }

    func cancelCountdown() {
        countdownTimer?.invalidate()
        countdownTimer = nil
        isCountdownActive = false
        countdownRemaining = 0
    }

    private func performCapture() {
        guard session.isRunning else { return }
        let settings = AVCapturePhotoSettings()
        if let device = currentDeviceInput?.device, device.hasFlash {
            settings.flashMode = flashMode
        }
        photoOutput.capturePhoto(with: settings, delegate: self)
        HapticManager.heavy()
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let data = photo.fileDataRepresentation(), let image = UIImage(data: data) else { return }
        let final: UIImage
        if cameraPosition == .front, let cg = image.cgImage {
            final = UIImage(cgImage: cg, scale: image.scale, orientation: .leftMirrored)
        } else {
            final = image
        }
        DispatchQueue.main.async { self.onPhotoCaptured?(final) }
    }

    // MARK: - Real-time Vision Frame Processing
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !isProcessingFrame, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        isProcessingFrame = true

        let isFront = cameraPosition == .front
        let orientation: CGImagePropertyOrientation = isFront ? .leftMirrored : .right

        let bodyRequest = VNDetectHumanBodyPoseRequest()
        let faceRequest = VNDetectFaceLandmarksRequest()

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
        do {
            try handler.perform([bodyRequest, faceRequest])

            var startPt: CGPoint? = nil
            var endPt: CGPoint? = nil
            var aligned = true
            var facePoints: [CGPoint] = []

            // Extract Posture Axis (Neck to Root / Spine)
            if let body = bodyRequest.results?.first {
                if let neck = try? body.recognizedPoint(.neck),
                   let root = try? body.recognizedPoint(.root),
                   neck.confidence > 0.25, root.confidence > 0.25 {
                    // Convert Vision (bottom-left 0,0) to SwiftUI (top-left 0,0)
                    startPt = CGPoint(x: neck.location.x, y: 1.0 - neck.location.y)
                    endPt = CGPoint(x: root.location.x, y: 1.0 - root.location.y)
                    let dx = abs(neck.location.x - root.location.x)
                    aligned = dx < 0.05
                }
            }

            // Extract Face Contour
            if let face = faceRequest.results?.first, let landmarks = face.landmarks {
                let bbox = face.boundingBox
                if let contour = landmarks.faceContour {
                    facePoints = contour.normalizedPoints.map { pt in
                        let x = bbox.origin.x + (pt.x * bbox.size.width)
                        let y = 1.0 - (bbox.origin.y + (pt.y * bbox.size.height))
                        return CGPoint(x: x, y: y)
                    }
                }
            }

            DispatchQueue.main.async {
                self.spineStartPoint = startPt
                self.spineEndPoint = endPt
                self.faceLandmarkPoints = facePoints
                self.isPostureAligned = aligned
                self.isProcessingFrame = false
            }
        } catch {
            DispatchQueue.main.async { self.isProcessingFrame = false }
        }
    }
}
