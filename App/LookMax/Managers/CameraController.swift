import Foundation
import AVFoundation
import UIKit
import Combine

class CameraController: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
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

    private var photoOutput = AVCapturePhotoOutput()
    private var currentDeviceInput: AVCaptureDeviceInput?
    private var countdownTimer: Timer?
    var onPhotoCaptured: ((UIImage) -> Void)?

    var isUltraWideAvailable: Bool { minZoomFactor < 0.95 }
    var isTelephotoAvailable: Bool { maxZoomFactor >= 2.0 }

    func setupCamera() {
        guard !session.isRunning else { return }
        session.beginConfiguration()
        session.sessionPreset = .photo
        guard let device = getCameraDevice(for: cameraPosition) else {
            session.commitConfiguration(); return
        }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) { session.addInput(input); currentDeviceInput = input }
            if session.canAddOutput(photoOutput) { session.addOutput(photoOutput) }
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
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        countdownTimer?.invalidate()
        countdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] t in
            guard let self else { return }
            self.countdownRemaining -= 1
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
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
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
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
}
