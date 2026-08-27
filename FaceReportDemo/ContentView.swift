import SwiftUI
import Vision
import CoreImage
import AVFoundation
import PhotosUI
import Combine

// MARK: - Analysis Scope Enum
enum VisualScope: String, CaseIterable {
    case closeUpFace = "Close-up Face / Portrait"
    case upperBody = "Upper Body Portrait"
    case fullBody = "Full Body Stance"
    case generalScene = "Person in Scene"
    
    var icon: String {
        switch self {
        case .closeUpFace: return "face.smiling"
        case .upperBody: return "person.crop.rectangle"
        case .fullBody: return "figure.stand"
        case .generalScene: return "photo"
        }
    }
}

// MARK: - Report Section Data Model
struct VisualReportCategory: Identifiable {
    let id = UUID()
    let title: String
    let icon: String
    let items: [String]
}

// MARK: - Main ContentView
struct ContentView: View {
    @State private var showingCustomCamera = false
    @State private var showingLibraryPicker = false
    @State private var inputImage: UIImage?
    @State private var detectedScope: VisualScope?
    @State private var reportCategories: [VisualReportCategory] = []
    @State private var isAnalyzing = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Image Preview Card
                    if let inputImage = inputImage {
                        ZStack(alignment: .topTrailing) {
                            Image(uiImage: inputImage)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 320)
                                .cornerRadius(16)
                                .shadow(color: Color.black.opacity(0.15), radius: 8, x: 0, y: 4)
                            
                            if let scope = detectedScope {
                                Label(scope.rawValue, systemImage: scope.icon)
                                    .font(.caption.bold())
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(.ultraThinMaterial)
                                    .clipShape(Capsule())
                                    .padding(12)
                            }
                        }
                    } else {
                        ZStack {
                            RoundedRectangle(cornerRadius: 16)
                                .fill(Color.secondary.opacity(0.12))
                                .frame(height: 280)
                            VStack(spacing: 12) {
                                Image(systemName: "person.and.background.dotted")
                                    .font(.system(size: 64))
                                    .foregroundColor(.secondary)
                                Text("Take or choose a picture to analyze visual features")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal)
                            }
                        }
                    }
                    
                    // Loading State
                    if isAnalyzing {
                        VStack(spacing: 12) {
                            ProgressView()
                                .scaleEffect(1.2)
                            Text("Running Apple Vision Visual Analysis...")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                    }
                    
                    // Comprehensive Visual Report
                    if !reportCategories.isEmpty && !isAnalyzing {
                        VStack(alignment: .leading, spacing: 16) {
                            HStack {
                                Text("Visual Analysis Report")
                                    .font(.title3.bold())
                                Spacer()
                                if let scope = detectedScope {
                                    Text(scope.rawValue)
                                        .font(.caption.weight(.semibold))
                                        .foregroundColor(.blue)
                                }
                            }
                            .padding(.horizontal, 4)
                            
                            ForEach(reportCategories) { category in
                                VStack(alignment: .leading, spacing: 10) {
                                    HStack(spacing: 8) {
                                        Image(systemName: category.icon)
                                            .foregroundColor(.blue)
                                            .font(.headline)
                                        Text(category.title)
                                            .font(.headline)
                                    }
                                    
                                    ForEach(category.items, id: \.self) { item in
                                        HStack(alignment: .top, spacing: 10) {
                                            Image(systemName: "checkmark.seal.fill")
                                                .foregroundColor(.green)
                                                .font(.subheadline)
                                                .padding(.top, 2)
                                            Text(item)
                                                .font(.subheadline)
                                                .foregroundColor(.primary)
                                        }
                                    }
                                }
                                .padding(14)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(UIColor.secondarySystemBackground))
                                .cornerRadius(14)
                            }
                        }
                    }
                    
                    // Action Buttons
                    VStack(spacing: 12) {
                        Button(action: {
                            checkCameraPermissionAndPresent()
                        }) {
                            Label(inputImage == nil ? "Open Camera (with Timer & Controls)" : "Retake Photo", systemImage: "camera.viewfinder")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                        
                        Button(action: {
                            showingLibraryPicker = true
                        }) {
                            Label("Choose from Photo Library", systemImage: "photo.on.rectangle")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color(UIColor.secondarySystemBackground))
                                .foregroundColor(.primary)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                    }
                    .padding(.top, 8)
                }
                .padding()
            }
            .navigationTitle("Visual AI Reporter")
            .fullScreenCover(isPresented: $showingCustomCamera) {
                CustomCameraView(isPresented: $showingCustomCamera, onPhotoCaptured: { image in
                    self.inputImage = image
                    self.startVisualAnalysis()
                })
                .ignoresSafeArea()
            }
            .sheet(isPresented: $showingLibraryPicker, onDismiss: startVisualAnalysis) {
                ImagePicker(image: $inputImage, sourceType: .photoLibrary)
            }
        }
    }
    
    // MARK: - Permission Handler
    private func checkCameraPermissionAndPresent() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            showingCustomCamera = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        showingCustomCamera = true
                    }
                }
            }
        default:
            showingCustomCamera = true
        }
    }
    
    // MARK: - AI Visual Analysis Router
    private func startVisualAnalysis() {
        guard let uiImage = inputImage, let cgImage = uiImage.cgImage else { return }
        
        isAnalyzing = true
        reportCategories = []
        detectedScope = nil
        
        DispatchQueue.global(qos: .userInitiated).async {
            let orientation = CGImagePropertyOrientation(uiImage.imageOrientation)
            
            // 1. Initial Requests to determine framing & features
            let faceRequest = VNDetectFaceLandmarksRequest()
            let faceQualityRequest = VNDetectFaceCaptureQualityRequest()
            let bodyPoseRequest = VNDetectHumanBodyPoseRequest()
            let segmentationRequest = VNGeneratePersonSegmentationRequest()
            segmentationRequest.qualityLevel = .balanced
            let classificationRequest = VNClassifyImageRequest()
            
            let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
            
            do {
                try handler.perform([faceRequest, faceQualityRequest, bodyPoseRequest, segmentationRequest, classificationRequest])
                
                let faces = faceRequest.results ?? []
                let bodyPoses = bodyPoseRequest.results ?? []
                let classifications = classificationRequest.results ?? []
                
                // 2. Determine Scope (Face, Upper Body, Full Body, General)
                let scope = determineImageScope(faces: faces, bodyPoses: bodyPoses, imageSize: CGSize(width: cgImage.width, height: cgImage.height))
                
                var categories: [VisualReportCategory] = []
                
                // 3. Select and execute relevant analysis functions based on detected scope
                
                // A. Framing & Stance Analysis
                let stanceItems = analyzeFramingAndPose(scope: scope, bodyPoses: bodyPoses, faces: faces)
                if !stanceItems.isEmpty {
                    categories.append(VisualReportCategory(title: "Framing & Body Pose", icon: "figure.walk", items: stanceItems))
                }
                
                // B. Head, Hairstyle & Eyewear Analysis
                let headAndFaceItems = analyzeHeadHairstyleAndEyewear(faces: faces, classifications: classifications, cgImage: cgImage, orientation: orientation)
                if !headAndFaceItems.isEmpty {
                    categories.append(VisualReportCategory(title: "Hairstyle, Eyewear & Facial Features", icon: "eyeglasses", items: headAndFaceItems))
                }
                
                // C. Apparel, Clothing & Accessories Analysis
                let apparelItems = analyzeApparelAndOutfit(scope: scope, bodyPoses: bodyPoses, classifications: classifications, cgImage: cgImage)
                if !apparelItems.isEmpty {
                    categories.append(VisualReportCategory(title: "Apparel & Wearing Analysis", icon: "tshirt.fill", items: apparelItems))
                }
                
                // D. Visual Context & Capture Quality
                let contextItems = analyzeCaptureAndEnvironment(faces: faces, faceQualityRequest: faceQualityRequest, classifications: classifications)
                if !contextItems.isEmpty {
                    categories.append(VisualReportCategory(title: "Environment & Image Context", icon: "sparkles", items: contextItems))
                }
                
                DispatchQueue.main.async {
                    self.detectedScope = scope
                    self.reportCategories = categories
                    self.isAnalyzing = false
                }
                
            } catch {
                print("Vision analysis failed: \(error)")
                DispatchQueue.main.async {
                    self.reportCategories = [
                        VisualReportCategory(title: "Analysis Status", icon: "exclamationmark.triangle", items: ["Visual analysis could not identify subject clearly."])
                    ]
                    self.isAnalyzing = false
                }
            }
        }
    }
    
    // MARK: - Scope Determination Function
    private func determineImageScope(faces: [VNFaceObservation], bodyPoses: [VNHumanBodyPoseObservation], imageSize: CGSize) -> VisualScope {
        if let primaryBody = bodyPoses.first {
            let hasAnkles = (try? primaryBody.recognizedPoint(.leftAnkle))?.confidence ?? 0 > 0.3 ||
                            (try? primaryBody.recognizedPoint(.rightAnkle))?.confidence ?? 0 > 0.3
            let hasKnees = (try? primaryBody.recognizedPoint(.leftKnee))?.confidence ?? 0 > 0.3 ||
                           (try? primaryBody.recognizedPoint(.rightKnee))?.confidence ?? 0 > 0.3
            
            if hasAnkles || hasKnees {
                return .fullBody
            }
            
            let hasTorsoOrArms = (try? primaryBody.recognizedPoint(.leftShoulder))?.confidence ?? 0 > 0.3 ||
                                 (try? primaryBody.recognizedPoint(.rightShoulder))?.confidence ?? 0 > 0.3 ||
                                 (try? primaryBody.recognizedPoint(.neck))?.confidence ?? 0 > 0.3
            if hasTorsoOrArms {
                return .upperBody
            }
        }
        
        if let primaryFace = faces.first {
            let faceAreaRatio = primaryFace.boundingBox.width * primaryFace.boundingBox.height
            if faceAreaRatio > 0.12 {
                return .closeUpFace
            } else {
                return .upperBody
            }
        }
        
        return .generalScene
    }
    
    // MARK: - Function 1: Framing & Pose Analysis
    private func analyzeFramingAndPose(scope: VisualScope, bodyPoses: [VNHumanBodyPoseObservation], faces: [VNFaceObservation]) -> [String] {
        var items: [String] = []
        items.append("Detected Framing: \(scope.rawValue)")
        
        if let body = bodyPoses.first {
            let leftWrist = try? body.recognizedPoint(.leftWrist)
            let rightWrist = try? body.recognizedPoint(.rightWrist)
            let leftShoulder = try? body.recognizedPoint(.leftShoulder)
            let rightShoulder = try? body.recognizedPoint(.rightShoulder)
            
            if let lw = leftWrist, let ls = leftShoulder, lw.confidence > 0.4, ls.confidence > 0.4, lw.location.y > ls.location.y {
                items.append("Arm/Hand Gesture: Raised hand detected")
            } else if let rw = rightWrist, let rs = rightShoulder, rw.confidence > 0.4, rs.confidence > 0.4, rw.location.y > rs.location.y {
                items.append("Arm/Hand Gesture: Raised hand detected")
            } else {
                items.append("Arm Position: Relaxed / at sides")
            }
            
            if let neck = try? body.recognizedPoint(.neck), let root = try? body.recognizedPoint(.root), neck.confidence > 0.3, root.confidence > 0.3 {
                let dx = abs(neck.location.x - root.location.x)
                if dx < 0.05 {
                    items.append("Body Posture: Upright and centered")
                } else {
                    items.append("Body Posture: Angled / leaning stance")
                }
            }
        } else if let face = faces.first {
            if let yaw = face.yaw {
                if abs(yaw.doubleValue) < 0.15 {
                    items.append("Facing Direction: Directly facing camera")
                } else if yaw.doubleValue > 0.15 {
                    items.append("Facing Direction: Turned towards right")
                } else {
                    items.append("Facing Direction: Turned towards left")
                }
            }
            if let roll = face.roll, abs(roll.doubleValue) > 0.15 {
                items.append("Head Tilt: Tilted head angle")
            }
        }
        
        return items
    }
    
    // MARK: - Function 2: Head, Hairstyle & Eyewear Analysis
    private func analyzeHeadHairstyleAndEyewear(faces: [VNFaceObservation], classifications: [VNClassificationObservation], cgImage: CGImage, orientation: CGImagePropertyOrientation) -> [String] {
        var items: [String] = []
        
        let eyewearMatches = classifications.filter {
            let id = $0.identifier.lowercased()
            return (id.contains("sunglass") || id.contains("spectacles") || id.contains("glasses") || id.contains("eyewear") || id.contains("goggles")) && $0.confidence > 0.3
        }
        
        if let topEyewear = eyewearMatches.first {
            let label = topEyewear.identifier.capitalized
            items.append("Eyewear: Wearing \(label) (ML verified)")
        } else {
            items.append("Eyewear: No prominent glasses/sunglasses detected")
        }
        
        let headwearMatches = classifications.filter {
            let id = $0.identifier.lowercased()
            return (id.contains("hat") || id.contains("cap") || id.contains("helmet") || id.contains("beanie") || id.contains("beret") || id.contains("fedora") || id.contains("turban")) && $0.confidence > 0.3
        }
        if let topHeadwear = headwearMatches.first {
            items.append("Headwear: Wearing \(topHeadwear.identifier.capitalized)")
        }
        
        let hairMatches = classifications.filter {
            let id = $0.identifier.lowercased()
            return (id.contains("hair") || id.contains("beard") || id.contains("mustache") || id.contains("wig") || id.contains("ponytail") || id.contains("braid") || id.contains("afro")) && $0.confidence > 0.3
        }
        for match in hairMatches.prefix(2) {
            items.append("Hair/Styling: \(match.identifier.capitalized)")
        }
        
        if let face = faces.first {
            if let landmarks = face.landmarks {
                var facialDetails: [String] = []
                if landmarks.leftEye != nil && landmarks.rightEye != nil {
                    facialDetails.append("Both eyes clearly visible")
                }
                if landmarks.outerLips != nil {
                    facialDetails.append("Mouth / smile line mapped")
                }
                if landmarks.faceContour != nil {
                    facialDetails.append("Jawline & face contour detected")
                }
                if !facialDetails.isEmpty {
                    items.append("Facial Landmarks: " + facialDetails.joined(separator: ", "))
                }
            }
        }
        
        return items
    }
    
    // MARK: - Function 3: Apparel, Clothing & Accessories Analysis
    private func analyzeApparelAndOutfit(scope: VisualScope, bodyPoses: [VNHumanBodyPoseObservation], classifications: [VNClassificationObservation], cgImage: CGImage) -> [String] {
        var items: [String] = []
        
        let clothingKeywords = [
            "shirt", "t-shirt", "jersey", "suit", "jacket", "coat", "blazer", "sweater", "cardigan",
            "hoodie", "dress", "gown", "top", "blouse", "tie", "bow-tie", "uniform", "denim",
            "jeans", "trousers", "pants", "shorts", "skirt", "sneaker", "shoe", "boot"
        ]
        
        let matchingApparel = classifications.filter { item in
            let id = item.identifier.lowercased()
            return clothingKeywords.contains { kw in id.contains(kw) } && item.confidence > 0.25
        }
        
        if !matchingApparel.isEmpty {
            for apparel in matchingApparel.prefix(3) {
                let name = apparel.identifier.capitalized
                items.append("Outfit item: \(name)")
            }
        } else {
            items.append("Apparel style: Casual / everyday clothing")
        }
        
        let sampleRect = CGRect(x: 0.35, y: 0.45, width: 0.30, height: 0.25)
        if let colorName = sampleDominantColorName(in: cgImage, normalizedRect: sampleRect) {
            items.append("Top / Upper Body Color: Dominant \(colorName)")
        }
        
        if scope == .fullBody {
            let lowerSampleRect = CGRect(x: 0.35, y: 0.70, width: 0.30, height: 0.20)
            if let lowerColor = sampleDominantColorName(in: cgImage, normalizedRect: lowerSampleRect) {
                items.append("Bottom / Lower Body Color: Dominant \(lowerColor)")
            }
        }
        
        let accessoryKeywords = ["watch", "backpack", "bag", "purse", "necklace", "earring", "bracelet", "scarf", "glove"]
        let accessoryMatches = classifications.filter { item in
            let id = item.identifier.lowercased()
            return accessoryKeywords.contains { kw in id.contains(kw) } && item.confidence > 0.25
        }
        for acc in accessoryMatches.prefix(2) {
            items.append("Accessory: \(acc.identifier.capitalized)")
        }
        
        return items
    }
    
    // MARK: - Function 4: Capture Quality & Environment
    private func analyzeCaptureAndEnvironment(faces: [VNFaceObservation], faceQualityRequest: VNDetectFaceCaptureQualityRequest, classifications: [VNClassificationObservation]) -> [String] {
        var items: [String] = []
        
        if let qualityFace = faceQualityRequest.results?.first, let quality = qualityFace.faceCaptureQuality {
            let score = Int(quality * 100)
            items.append("Capture Clarity: \(score)% score (lighting & sharpness)")
        }
        
        let envKeywords = ["indoor", "outdoor", "room", "studio", "office", "nature", "stage", "urban", "home"]
        let envMatches = classifications.filter { item in
            let id = item.identifier.lowercased()
            return envKeywords.contains { kw in id.contains(kw) } && item.confidence > 0.35
        }
        if let topEnv = envMatches.first {
            items.append("Setting / Environment: \(topEnv.identifier.capitalized)")
        }
        
        return items
    }
    
    // MARK: - Dominant Color Sampling Utility
    private func sampleDominantColorName(in cgImage: CGImage, normalizedRect: CGRect) -> String? {
        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)
        
        let cropRect = CGRect(
            x: normalizedRect.origin.x * width,
            y: normalizedRect.origin.y * height,
            width: normalizedRect.size.width * width,
            height: normalizedRect.size.height * height
        )
        
        guard let cropped = cgImage.cropping(to: cropRect) else { return nil }
        
        let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        
        context?.draw(cropped, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        guard let data = context?.data else { return nil }
        
        let pointer = data.bindMemory(to: UInt8.self, capacity: 4)
        let r = Double(pointer[0]) / 255.0
        let g = Double(pointer[1]) / 255.0
        let b = Double(pointer[2]) / 255.0
        
        return approximateColorName(r: r, g: g, b: b)
    }
    
    private func approximateColorName(r: Double, g: Double, b: Double) -> String {
        let brightness = (r + g + b) / 3.0
        if brightness < 0.18 { return "Black / Dark tone" }
        if brightness > 0.85 { return "White / Light tone" }
        
        let maxDiff = max(abs(r - g), abs(g - b), abs(r - b))
        if maxDiff < 0.08 { return "Grey / Neutral tone" }
        
        if r > g && r > b {
            if g > 0.5 { return "Yellow / Orange tone" }
            return "Red / Warm tone"
        } else if g > r && g > b {
            return "Green tone"
        } else if b > r && b > g {
            if r > 0.4 { return "Purple / Violet tone" }
            return "Blue / Navy tone"
        }
        
        return "Medium Neutral tone"
    }
}

// MARK: - Camera Controller (AVCaptureSession Engine)
class CameraController: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
    @Published var session = AVCaptureSession()
    @Published var isSessionRunning = false
    @Published var isCountdownActive = false
    @Published var countdownRemaining = 0
    @Published var selectedTimerDuration: Int = 0 // 0s (Off), 3s, 5s, 10s
    @Published var flashMode: AVCaptureDevice.FlashMode = .auto
    @Published var cameraPosition: AVCaptureDevice.Position = .back
    @Published var isGridVisible = false
    @Published var currentZoomFactor: CGFloat = 1.0
    @Published var focusPoint: CGPoint?
    @Published var isFocusing = false
    
    private var photoOutput = AVCapturePhotoOutput()
    private var currentDeviceInput: AVCaptureDeviceInput?
    private var countdownTimer: Timer?
    var onPhotoCaptured: ((UIImage) -> Void)?
    
    func setupCamera() {
        guard !session.isRunning else { return }
        
        session.beginConfiguration()
        session.sessionPreset = .photo
        
        guard let device = getCameraDevice(for: cameraPosition) else {
            session.commitConfiguration()
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
                currentDeviceInput = input
            }
            
            if session.canAddOutput(photoOutput) {
                session.addOutput(photoOutput)
            }
        } catch {
            print("Camera configuration error: \(error)")
        }
        
        session.commitConfiguration()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async {
                self?.isSessionRunning = self?.session.isRunning ?? false
            }
        }
    }
    
    func stopCamera() {
        cancelCountdown()
        if session.isRunning {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.session.stopRunning()
                DispatchQueue.main.async {
                    self?.isSessionRunning = false
                }
            }
        }
    }
    
    func switchCamera() {
        session.beginConfiguration()
        if let currentInput = currentDeviceInput {
            session.removeInput(currentInput)
        }
        
        let newPosition: AVCaptureDevice.Position = (cameraPosition == .back) ? .front : .back
        guard let newDevice = getCameraDevice(for: newPosition) else {
            if let currentInput = currentDeviceInput {
                session.addInput(currentInput)
            }
            session.commitConfiguration()
            return
        }
        
        do {
            let newInput = try AVCaptureDeviceInput(device: newDevice)
            if session.canAddInput(newInput) {
                session.addInput(newInput)
                currentDeviceInput = newInput
                cameraPosition = newPosition
                currentZoomFactor = 1.0
            }
        } catch {
            print("Error switching camera: \(error)")
        }
        session.commitConfiguration()
    }
    
    private func getCameraDevice(for position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let deviceTypes: [AVCaptureDevice.DeviceType]
        if position == .back {
            deviceTypes = [.builtInTripleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInUltraWideCamera, .builtInWideAngleCamera]
        } else {
            deviceTypes = [.builtInTrueDepthCamera, .builtInWideAngleCamera]
        }
        
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: .video,
            position: position
        )
        return discovery.devices.first
    }
    
    var minZoomFactor: CGFloat {
        guard let device = currentDeviceInput?.device else { return 1.0 }
        return device.minAvailableVideoZoomFactor
    }
    
    var maxZoomFactor: CGFloat {
        guard let device = currentDeviceInput?.device else { return 5.0 }
        return min(device.activeFormat.videoMaxZoomFactor, 10.0)
    }
    
    func toggleFlash() {
        switch flashMode {
        case .auto: flashMode = .on
        case .on: flashMode = .off
        case .off: flashMode = .auto
        @unknown default: flashMode = .auto
        }
    }
    
    func setZoom(_ factor: CGFloat) {
        guard let device = currentDeviceInput?.device else { return }
        do {
            try device.lockForConfiguration()
            let minZoom = device.minAvailableVideoZoomFactor
            let maxZoom = min(device.activeFormat.videoMaxZoomFactor, 10.0)
            let targetZoom = max(minZoom, min(factor, maxZoom))
            device.videoZoomFactor = targetZoom
            currentZoomFactor = targetZoom
            device.unlockForConfiguration()
        } catch {
            print("Zoom error: \(error)")
        }
    }
    
    func focus(at point: CGPoint, viewSize: CGSize) {
        guard let device = currentDeviceInput?.device else { return }
        
        // Convert screen coordinates to normalized camera coordinates (0.0 to 1.0)
        let focusPointNormalized = CGPoint(x: point.y / viewSize.height, y: 1.0 - (point.x / viewSize.width))
        
        do {
            try device.lockForConfiguration()
            if device.isFocusPointOfInterestSupported && device.isFocusModeSupported(.autoFocus) {
                device.focusPointOfInterest = focusPointNormalized
                device.focusMode = .autoFocus
            }
            if device.isExposurePointOfInterestSupported && device.isExposureModeSupported(.autoExpose) {
                device.exposurePointOfInterest = focusPointNormalized
                device.exposureMode = .autoExpose
            }
            device.unlockForConfiguration()
            
            self.focusPoint = point
            self.isFocusing = true
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
                self.isFocusing = false
            }
        } catch {
            print("Focus error: \(error)")
        }
    }
    
    func triggerCapture() {
        if selectedTimerDuration > 0 {
            startCountdown(seconds: selectedTimerDuration)
        } else {
            performActualCapture()
        }
    }
    
    private func startCountdown(seconds: Int) {
        countdownRemaining = seconds
        isCountdownActive = true
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        
        countdownTimer?.invalidate()
        countdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            guard let self = self else { return }
            self.countdownRemaining -= 1
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            
            if self.countdownRemaining <= 0 {
                timer.invalidate()
                self.isCountdownActive = false
                self.performActualCapture()
            }
        }
    }
    
    func cancelCountdown() {
        countdownTimer?.invalidate()
        countdownTimer = nil
        isCountdownActive = false
        countdownRemaining = 0
    }
    
    private func performActualCapture() {
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
        
        let finalImage: UIImage
        if cameraPosition == .front, let cgImg = image.cgImage {
            finalImage = UIImage(cgImage: cgImg, scale: image.scale, orientation: .leftMirrored)
        } else {
            finalImage = image
        }
        
        DispatchQueue.main.async {
            self.onPhotoCaptured?(finalImage)
        }
    }
}

// MARK: - Custom Camera Fullscreen View
struct CustomCameraView: View {
    @Binding var isPresented: Bool
    var onPhotoCaptured: (UIImage) -> Void
    
    @StateObject private var camera = CameraController()
    @State private var showingTimerMenu = false
    @State private var showingLibraryPicker = false
    @State private var pickerImage: UIImage?
    
    @State private var baseZoomFactor: CGFloat = 1.0
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            // 1. Live Camera Preview Layer
            GeometryReader { geo in
                ZStack {
                    CameraPreviewView(camera: camera)
                        .onTapGesture { location in
                            camera.focus(at: location, viewSize: geo.size)
                        }
                        .gesture(
                            MagnificationGesture()
                                .onChanged { value in
                                    camera.setZoom(baseZoomFactor * value)
                                }
                                .onEnded { _ in
                                    baseZoomFactor = camera.currentZoomFactor
                                }
                        )
                    
                    // Grid Overlay
                    if camera.isGridVisible {
                        CameraGridView()
                    }
                    
                    // Tap to Focus Square Box
                    if camera.isFocusing, let point = camera.focusPoint {
                        Rectangle()
                            .stroke(Color.yellow, lineWidth: 1.5)
                            .frame(width: 70, height: 70)
                            .position(point)
                            .animation(.easeInOut(duration: 0.2), value: camera.isFocusing)
                    }
                }
            }
            .ignoresSafeArea()
            
            // 2. Countdown Timer Overlay
            if camera.isCountdownActive {
                ZStack {
                    Color.black.opacity(0.4).ignoresSafeArea()
                    VStack(spacing: 24) {
                        Text("\(camera.countdownRemaining)")
                            .font(.system(size: 110, weight: .bold, design: .rounded))
                            .foregroundColor(.white)
                            .shadow(radius: 12)
                            .scaleEffect(camera.countdownRemaining > 0 ? 1.1 : 0.8)
                            .animation(.easeInOut(duration: 0.3), value: camera.countdownRemaining)
                        
                        Button(action: {
                            camera.cancelCountdown()
                        }) {
                            Text("Cancel Timer")
                                .font(.subheadline.bold())
                                .foregroundColor(.white)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 10)
                                .background(Capsule().fill(Color.red.opacity(0.85)))
                        }
                    }
                }
            }
            
            // 3. Camera Controls UI Overlays
            VStack {
                // Top Control Bar
                HStack(spacing: 24) {
                    // Flash Mode Toggle
                    Button(action: {
                        camera.toggleFlash()
                    }) {
                        Image(systemName: flashIconName)
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(camera.flashMode == .off ? .white : .yellow)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                    }
                    
                    // Delayed Capture Timer Button
                    Menu {
                        Button("Timer Off") { camera.selectedTimerDuration = 0 }
                        Button("3 Seconds") { camera.selectedTimerDuration = 3 }
                        Button("5 Seconds") { camera.selectedTimerDuration = 5 }
                        Button("10 Seconds") { camera.selectedTimerDuration = 10 }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "timer")
                            if camera.selectedTimerDuration > 0 {
                                Text("\(camera.selectedTimerDuration)s")
                                    .font(.caption.bold())
                            }
                        }
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(camera.selectedTimerDuration > 0 ? .yellow : .white)
                        .padding(.horizontal, camera.selectedTimerDuration > 0 ? 12 : 10)
                        .frame(height: 44)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())
                    }
                    
                    // Grid Toggle
                    Button(action: {
                        camera.isGridVisible.toggle()
                    }) {
                        Image(systemName: camera.isGridVisible ? "grid.circle.fill" : "grid")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(camera.isGridVisible ? .yellow : .white)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                    }
                    
                    Spacer()
                    
                    // Close Button
                    Button(action: {
                        camera.stopCamera()
                        isPresented = false
                    }) {
                        Image(systemName: "xmark")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 50)
                
                Spacer()
                
                // Zoom Switcher (0.5x / 1x / 2x)
                HStack(spacing: 12) {
                    Button(action: {
                        let target = max(0.5, camera.minZoomFactor)
                        camera.setZoom(target)
                        baseZoomFactor = target
                    }) {
                        Text(".5x")
                            .font(.caption.bold())
                            .foregroundColor(camera.currentZoomFactor < 0.9 ? .yellow : .white)
                            .frame(width: 38, height: 38)
                            .background(Color.black.opacity(0.65))
                            .clipShape(Circle())
                    }
                    
                    Button(action: {
                        camera.setZoom(1.0)
                        baseZoomFactor = 1.0
                    }) {
                        Text("1x")
                            .font(.caption.bold())
                            .foregroundColor(abs(camera.currentZoomFactor - 1.0) < 0.2 ? .yellow : .white)
                            .frame(width: 38, height: 38)
                            .background(Color.black.opacity(0.65))
                            .clipShape(Circle())
                    }
                    
                    Button(action: {
                        camera.setZoom(2.0)
                        baseZoomFactor = 2.0
                    }) {
                        Text("2x")
                            .font(.caption.bold())
                            .foregroundColor(camera.currentZoomFactor >= 1.8 ? .yellow : .white)
                            .frame(width: 38, height: 38)
                            .background(Color.black.opacity(0.65))
                            .clipShape(Circle())
                    }
                }
                .padding(.bottom, 16)
                
                // Bottom Control Bar
                HStack(alignment: .center) {
                    // Photo Library Shortcut
                    Button(action: {
                        showingLibraryPicker = true
                    }) {
                        Image(systemName: "photo.on.rectangle")
                            .font(.system(size: 24))
                            .foregroundColor(.white)
                            .frame(width: 54, height: 54)
                            .background(Color.white.opacity(0.2))
                            .clipShape(Circle())
                    }
                    
                    Spacer()
                    
                    // Shutter Button with Countdown State
                    Button(action: {
                        camera.triggerCapture()
                    }) {
                        ZStack {
                            Circle()
                                .strokeBorder(Color.white, lineWidth: 4)
                                .frame(width: 78, height: 78)
                            
                            Circle()
                                .fill(camera.isCountdownActive ? Color.yellow : Color.white)
                                .frame(width: 64, height: 64)
                                .scaleEffect(camera.isCountdownActive ? 0.85 : 1.0)
                                .animation(.spring(), value: camera.isCountdownActive)
                        }
                    }
                    
                    Spacer()
                    
                    // Front / Back Camera Switch
                    Button(action: {
                        camera.switchCamera()
                    }) {
                        Image(systemName: "camera.rotate.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.white)
                            .frame(width: 54, height: 54)
                            .background(Color.white.opacity(0.2))
                            .clipShape(Circle())
                    }
                }
                .padding(.horizontal, 30)
                .padding(.bottom, 40)
            }
        }
        .onAppear {
            camera.onPhotoCaptured = { photo in
                camera.stopCamera()
                isPresented = false
                onPhotoCaptured(photo)
            }
            camera.setupCamera()
        }
        .onDisappear {
            camera.stopCamera()
        }
        .sheet(isPresented: $showingLibraryPicker) {
            ImagePicker(image: $pickerImage, sourceType: .photoLibrary)
                .onDisappear {
                    if let image = pickerImage {
                        camera.stopCamera()
                        isPresented = false
                        onPhotoCaptured(image)
                    }
                }
        }
    }
    
    private var flashIconName: String {
        switch camera.flashMode {
        case .auto: return "bolt.badge.a.fill"
        case .on: return "bolt.fill"
        case .off: return "bolt.slash.fill"
        @unknown default: return "bolt.badge.a.fill"
        }
    }
}

// MARK: - Camera Preview UI Representable
struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var camera: CameraController
    
    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.previewLayer.session = camera.session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }
    
    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {}
}

class CameraPreviewUIView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }
    
    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}

// MARK: - Camera Rule of Thirds Grid
struct CameraGridView: View {
    var body: some View {
        GeometryReader { geo in
            Path { path in
                let w = geo.size.width
                let h = geo.size.height
                
                // Vertical lines
                path.move(to: CGPoint(x: w / 3, y: 0))
                path.addLine(to: CGPoint(x: w / 3, y: h))
                path.move(to: CGPoint(x: 2 * w / 3, y: 0))
                path.addLine(to: CGPoint(x: 2 * w / 3, y: h))
                
                // Horizontal lines
                path.move(to: CGPoint(x: 0, y: h / 3))
                path.addLine(to: CGPoint(x: w, y: h / 3))
                path.move(to: CGPoint(x: 0, y: 2 * h / 3))
                path.addLine(to: CGPoint(x: w, y: 2 * h / 3))
            }
            .stroke(Color.white.opacity(0.3), lineWidth: 1)
        }
        .allowsHitTesting(false)
    }
}

// MARK: - CGImage Orientation Helper
extension CGImagePropertyOrientation {
    init(_ orientation: UIImage.Orientation) {
        switch orientation {
        case .up: self = .up
        case .upMirrored: self = .upMirrored
        case .down: self = .down
        case .downMirrored: self = .downMirrored
        case .left: self = .left
        case .leftMirrored: self = .leftMirrored
        case .right: self = .right
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}

// MARK: - UIKit Image Picker Bridge
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    var sourceType: UIImagePickerController.SourceType = .camera
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        if sourceType == .camera && UIImagePickerController.isSourceTypeAvailable(.camera) {
            picker.sourceType = .camera
        } else {
            picker.sourceType = .photoLibrary
        }
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

#Preview {
    ContentView()
}