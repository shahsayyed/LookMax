import Foundation
import AVFoundation
import UIKit
import Combine
import Vision

enum ScanMode: String, CaseIterable, Identifiable {
    case grooming = "Grooming"
    case outfit = "Full Outfit"
    var id: String { rawValue }

    var icon: String {
        switch self {
        case .grooming: return "face.dashed"
        case .outfit:   return "figure.stand"
        }
    }
}

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

    // Real-time AR Biometric Overlays & Guidance
    @Published var scanMode: ScanMode = .grooming
    @Published var isAutoCaptureEnabled: Bool = false
    @Published var isFramingMatched: Bool = false
    @Published var isLightingAdequate: Bool = true
    @Published var autoCaptureProgress: CGFloat = 0.0
    @Published var guidanceMessage: String = "Align within guide"
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
    private var autoCaptureHoldStart: CFTimeInterval? = nil
    private var hasFiredAutoCapture: Bool = false

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

    func toggleAutoCapture() {
        isAutoCaptureEnabled.toggle()
        autoCaptureProgress = 0.0
        autoCaptureHoldStart = nil
        hasFiredAutoCapture = false
        HapticManager.selection()
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let data = photo.fileDataRepresentation(), let image = UIImage(data: data) else { return }
        let final: UIImage
        if cameraPosition == .front, let cg = image.cgImage {
            final = UIImage(cgImage: cg, scale: image.scale, orientation: .leftMirrored)
        } else {
            final = image
        }
        hasFiredAutoCapture = false
        autoCaptureProgress = 0.0
        autoCaptureHoldStart = nil
        DispatchQueue.main.async { self.onPhotoCaptured?(final) }
    }

    // MARK: - Real-time Vision Frame Processing & Smart Quality Gates
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !isProcessingFrame, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        isProcessingFrame = true

        let isFront = cameraPosition == .front
        let orientation: CGImagePropertyOrientation = isFront ? .leftMirrored : .right

        let bodyRequest = VNDetectHumanBodyPoseRequest()
        let faceRequest = VNDetectFaceLandmarksRequest()
        let faceQualityRequest = VNDetectFaceCaptureQualityRequest()

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
        do {
            try handler.perform([bodyRequest, faceRequest, faceQualityRequest])

            var startPt: CGPoint? = nil
            var endPt: CGPoint? = nil
            var aligned = true
            var facePoints: [CGPoint] = []
            var framingMatched = false
            var lightingAdequate = true
            var message = "Position inside guide"

            if scanMode == .grooming {
                // ─── Grooming Quality Gate: Face Alignment & Lighting ───
                if let face = faceRequest.results?.first {
                    let bbox = face.boundingBox
                    let centerX = bbox.midX
                    let centerY = 1.0 - bbox.midY
                    let faceHeight = bbox.height

                    // Face contour for AR overlay
                    if let landmarks = face.landmarks, let contour = landmarks.faceContour {
                        facePoints = contour.normalizedPoints.map { pt in
                            let x = bbox.origin.x + (pt.x * bbox.size.width)
                            let y = 1.0 - (bbox.origin.y + (pt.y * bbox.size.height))
                            return CGPoint(x: x, y: y)
                        }
                    }

                    // Lighting Quality
                    let quality = Double(faceQualityRequest.results?.first?.faceCaptureQuality ?? 0.6)
                    if quality < 0.35 {
                        lightingAdequate = false
                        message = "Lighting too dark — face light source"
                    } else if faceHeight < 0.22 {
                        message = "Move closer to fill oval"
                    } else if faceHeight > 0.68 {
                        message = "Step back slightly"
                    } else if abs(centerX - 0.5) > 0.14 || abs(centerY - 0.44) > 0.16 {
                        message = "Center face inside oval"
                    } else {
                        framingMatched = true
                        message = isAutoCaptureEnabled ? "Hold steady..." : "Perfect alignment"
                    }
                } else {
                    message = "Position face inside oval"
                }
            } else {
                // ─── Outfit Quality Gate: Full Body (Head to Shoes) ───
                if let body = bodyRequest.results?.first {
                    let neck = try? body.recognizedPoint(.neck)
                    let root = try? body.recognizedPoint(.root)
                    let leftAnkle = try? body.recognizedPoint(.leftAnkle)
                    let rightAnkle = try? body.recognizedPoint(.rightAnkle)

                    if let n = neck, let r = root, n.confidence > 0.25, r.confidence > 0.25 {
                        startPt = CGPoint(x: n.location.x, y: 1.0 - n.location.y)
                        endPt = CGPoint(x: r.location.x, y: 1.0 - r.location.y)
                        let dx = abs(n.location.x - r.location.x)
                        aligned = dx < 0.05
                    }

                    let hasAnkles = (leftAnkle?.confidence ?? 0 > 0.2) || (rightAnkle?.confidence ?? 0 > 0.2)
                    if let n = neck, n.confidence > 0.25, n.location.y > 0.90 {
                        message = "Step back to show head"
                    } else if !hasAnkles {
                        message = "Step back to show shoes"
                    } else if !aligned {
                        message = "Align upright in silhouette"
                    } else {
                        framingMatched = true
                        message = isAutoCaptureEnabled ? "Hold steady..." : "Full body aligned"
                    }
                } else {
                    message = "Step into silhouette guide"
                }
            }

            // ─── Auto-Capture Engine ───
            var progress: CGFloat = 0.0
            var shouldTriggerPhoto = false

            if self.isAutoCaptureEnabled && framingMatched && lightingAdequate {
                let now = CACurrentMediaTime()
                let start = self.autoCaptureHoldStart ?? now
                self.autoCaptureHoldStart = start
                let elapsed = now - start
                progress = min(1.0, CGFloat(elapsed / 1.2))

                if progress >= 1.0 && !self.hasFiredAutoCapture {
                    self.hasFiredAutoCapture = true
                    shouldTriggerPhoto = true
                }
            } else {
                self.autoCaptureHoldStart = nil
                self.autoCaptureProgress = 0.0
                if !self.isAutoCaptureEnabled {
                    self.hasFiredAutoCapture = false
                }
            }

            DispatchQueue.main.async {
                self.spineStartPoint = startPt
                self.spineEndPoint = endPt
                self.faceLandmarkPoints = facePoints
                self.isPostureAligned = aligned
                self.isFramingMatched = framingMatched
                self.isLightingAdequate = lightingAdequate
                self.guidanceMessage = message
                self.autoCaptureProgress = progress
                self.isProcessingFrame = false

                if shouldTriggerPhoto {
                    HapticManager.success()
                    self.performCapture()
                }
            }
        } catch {
            DispatchQueue.main.async { self.isProcessingFrame = false }
        }
    }
}
