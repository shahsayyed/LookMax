import SwiftUI
import Vision
import CoreImage
import AVFoundation
import PhotosUI
import Combine

// MARK: - Biometric Face Signature Model
struct FaceBiometricSignature: Codable, Equatable {
    let eyeToNoseRatio: Double
    let eyeToMouthRatio: Double
    let noseToChinRatio: Double
    let mouthWidthRatio: Double
    let jawWidthRatio: Double
    let faceAspectRatio: Double
    
    // Compute similarity score between 0.0 (completely different) and 1.0 (identical)
    func similarity(to other: FaceBiometricSignature) -> Double {
        let diffs = [
            abs(eyeToNoseRatio - other.eyeToNoseRatio) * 1.5,
            abs(eyeToMouthRatio - other.eyeToMouthRatio) * 1.5,
            abs(noseToChinRatio - other.noseToChinRatio) * 1.2,
            abs(mouthWidthRatio - other.mouthWidthRatio) * 1.0,
            abs(jawWidthRatio - other.jawWidthRatio) * 1.0,
            abs(faceAspectRatio - other.faceAspectRatio) * 1.2
        ]
        
        let avgDiff = diffs.reduce(0, +) / Double(diffs.count)
        let similarity = max(0.0, min(1.0, 1.0 - (avgDiff * 2.8)))
        return similarity
    }
}

// MARK: - User Profile Model
struct UserProfile: Codable {
    var name: String
    var photoDataList: [Data]
    var signatures: [FaceBiometricSignature]
    var dateCreated: Date
    
    static let storageKey = "FaceReportDemo_UserProfile"
    
    static func load() -> UserProfile? {
        guard let data = UserDefaults.standard.data(forKey: storageKey) else { return nil }
        return try? JSONDecoder().decode(UserProfile.self, from: data)
    }
    
    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: UserProfile.storageKey)
        }
    }
    
    static func clear() {
        UserDefaults.standard.removeObject(forKey: storageKey)
    }
}

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

// MARK: - Verification Result Model
struct VerificationResult {
    let isMatched: Bool
    let confidence: Double // 0.0 to 1.0
    let profileName: String
    let statusMessage: String
}

// MARK: - Style Rating Model
struct StyleRating {
    let overallScore: Double // 0.0 to 10.0
    let title: String
    let styleVerdict: String
    let colorHarmony: String
    let groomingRating: String
    let lightingClarity: String
}

// MARK: - Report Category Model
struct VisualReportCategory: Identifiable {
    let id = UUID()
    let title: String
    let icon: String
    let color: Color
    let items: [String]
}

// MARK: - Main ContentView
struct ContentView: View {
    @State private var profile: UserProfile? = UserProfile.load()
    @State private var showingOnboarding = false
    @State private var showingCustomCamera = false
    @State private var showingLibraryPicker = false
    @State private var inputImage: UIImage?
    
    @State private var verificationResult: VerificationResult?
    @State private var styleRating: StyleRating?
    @State private var detectedScope: VisualScope?
    @State private var reportCategories: [VisualReportCategory] = []
    @State private var isAnalyzing = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Profile Header Card
                    if let user = profile {
                        HStack(spacing: 14) {
                            if let firstData = user.photoDataList.first, let avatar = UIImage(data: firstData) {
                                Image(uiImage: avatar)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 52, height: 52)
                                    .clipShape(Circle())
                                    .overlay(Circle().stroke(Color.blue, lineWidth: 2))
                            }
                            
                            VStack(alignment: .leading, spacing: 3) {
                                Text(user.name)
                                    .font(.headline)
                                Text("\(user.photoDataList.count) enrolled reference photo\(user.photoDataList.count > 1 ? "s" : "")")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            Button("Edit") {
                                showingOnboarding = true
                            }
                            .font(.subheadline.bold())
                            .padding(.horizontal, 14)
                            .padding(.vertical, 6)
                            .background(Color.blue.opacity(0.12))
                            .foregroundColor(.blue)
                            .clipShape(Capsule())
                        }
                        .padding(14)
                        .background(Color(UIColor.secondarySystemBackground))
                        .cornerRadius(14)
                    } else {
                        // Prompt to create profile
                        VStack(spacing: 12) {
                            Image(systemName: "person.badge.plus")
                                .font(.system(size: 38))
                                .foregroundColor(.blue)
                            Text("No Profile Enrolled")
                                .font(.headline)
                            Text("Create a profile with 1 to 3 reference photos to enable identity verification and personalized style ratings.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                            
                            Button(action: { showingOnboarding = true }) {
                                Text("Create Profile Now")
                                    .font(.subheadline.bold())
                                    .padding(.horizontal, 20)
                                    .padding(.vertical, 10)
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .clipShape(Capsule())
                            }
                        }
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color(UIColor.secondarySystemBackground))
                        .cornerRadius(16)
                    }
                    
                    // Image Preview Card
                    if let inputImage = inputImage {
                        ZStack(alignment: .topTrailing) {
                            Image(uiImage: inputImage)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 300)
                                .cornerRadius(16)
                                .shadow(color: Color.black.opacity(0.12), radius: 8, x: 0, y: 4)
                            
                            if let scope = detectedScope {
                                Label(scope.rawValue, systemImage: scope.icon)
                                    .font(.caption.bold())
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 5)
                                    .background(.ultraThinMaterial)
                                    .clipShape(Capsule())
                                    .padding(10)
                            }
                        }
                    } else {
                        ZStack {
                            RoundedRectangle(cornerRadius: 16)
                                .fill(Color.secondary.opacity(0.10))
                                .frame(height: 240)
                            VStack(spacing: 10) {
                                Image(systemName: "person.crop.rectangle.badge.plus")
                                    .font(.system(size: 52))
                                    .foregroundColor(.secondary)
                                Text("Take or choose a photo to verify identity & analyze style")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal)
                            }
                        }
                    }
                    
                    // Loading Spinner
                    if isAnalyzing {
                        VStack(spacing: 12) {
                            ProgressView()
                                .scaleEffect(1.2)
                            Text("Running Apple Vision Verification & Visual Analysis...")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                    }
                    
                    // Identity Verification Banner
                    if let verif = verificationResult, !isAnalyzing {
                        HStack(spacing: 14) {
                            Image(systemName: verif.isMatched ? "checkmark.seal.fill" : "person.fill.xmark")
                                .font(.system(size: 32))
                                .foregroundColor(verif.isMatched ? .green : .orange)
                            
                            VStack(alignment: .leading, spacing: 2) {
                                Text(verif.isMatched ? "Identity Verified" : "Identity Mismatch")
                                    .font(.headline.bold())
                                    .foregroundColor(verif.isMatched ? .green : .orange)
                                Text(verif.statusMessage)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            VStack(alignment: .trailing) {
                                Text("\(Int(verif.confidence * 100))%")
                                    .font(.title2.bold())
                                    .foregroundColor(verif.isMatched ? .green : .orange)
                                Text("Match")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(14)
                        .background(verif.isMatched ? Color.green.opacity(0.12) : Color.orange.opacity(0.12))
                        .cornerRadius(14)
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(verif.isMatched ? Color.green.opacity(0.3) : Color.orange.opacity(0.3), lineWidth: 1.5)
                        )
                    }
                    
                    // Style & Look Rating Card (Displayed when verified)
                    if let rating = styleRating, !isAnalyzing {
                        VStack(spacing: 14) {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Style & Aesthetic Rating")
                                        .font(.headline)
                                    Text(rating.styleVerdict)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                
                                Spacer()
                                
                                HStack(alignment: .firstTextBaseline, spacing: 2) {
                                    Text(String(format: "%.1f", rating.overallScore))
                                        .font(.system(size: 34, weight: .heavy, design: .rounded))
                                        .foregroundColor(.purple)
                                    Text("/10")
                                        .font(.subheadline.bold())
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            Divider()
                            
                            // Rating Metrics Grid
                            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                                MetricBadge(title: "Overall Look", value: rating.title, icon: "sparkles", color: .purple)
                                MetricBadge(title: "Color Harmony", value: rating.colorHarmony, icon: "paintpalette.fill", color: .blue)
                                MetricBadge(title: "Grooming & Hair", value: rating.groomingRating, icon: "comb.fill", color: .orange)
                                MetricBadge(title: "Lighting & Clarity", value: rating.lightingClarity, icon: "sun.max.fill", color: .green)
                            }
                        }
                        .padding(16)
                        .background(Color(UIColor.secondarySystemBackground))
                        .cornerRadius(16)
                    }
                    
                    // Detailed Visual Report Cards
                    if !reportCategories.isEmpty && !isAnalyzing {
                        VStack(alignment: .leading, spacing: 14) {
                            Text("Detailed Visual Analysis")
                                .font(.headline)
                                .padding(.horizontal, 4)
                            
                            ForEach(reportCategories) { category in
                                VStack(alignment: .leading, spacing: 10) {
                                    HStack(spacing: 8) {
                                        Image(systemName: category.icon)
                                            .foregroundColor(category.color)
                                            .font(.headline)
                                        Text(category.title)
                                            .font(.headline)
                                    }
                                    
                                    ForEach(category.items, id: \.self) { item in
                                        HStack(alignment: .top, spacing: 10) {
                                            Image(systemName: "checkmark.circle.fill")
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
                            Label(inputImage == nil ? "Scan / Take Picture" : "Scan Another Photo", systemImage: "camera.viewfinder")
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
            .navigationTitle("Face & Style AI")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingOnboarding = true }) {
                        Image(systemName: profile == nil ? "person.crop.circle.badge.plus" : "person.crop.circle")
                            .font(.system(size: 20))
                    }
                }
            }
            .sheet(isPresented: $showingOnboarding) {
                ProfileOnboardingView(profile: $profile)
            }
            .fullScreenCover(isPresented: $showingCustomCamera) {
                CustomCameraView(isPresented: $showingCustomCamera, onPhotoCaptured: { image in
                    self.inputImage = image
                    self.startVisualAndVerificationPipeline()
                })
                .ignoresSafeArea()
            }
            .sheet(isPresented: $showingLibraryPicker, onDismiss: startVisualAndVerificationPipeline) {
                ImagePicker(image: $inputImage, sourceType: .photoLibrary)
            }
        }
        .onAppear {
            if profile == nil {
                showingOnboarding = true
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
    
    // MARK: - Main AI Pipeline: Verification, Style & Visual Analysis
    private func startVisualAndVerificationPipeline() {
        guard let uiImage = inputImage, let cgImage = uiImage.cgImage else { return }
        
        isAnalyzing = true
        verificationResult = nil
        styleRating = nil
        reportCategories = []
        detectedScope = nil
        
        DispatchQueue.global(qos: .userInitiated).async {
            let orientation = CGImagePropertyOrientation(uiImage.imageOrientation)
            
            // 1. Setup Apple Vision Requests
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
                
                // 2. Extract Biometrics for Face Matching
                var currentSignature: FaceBiometricSignature?
                if let firstFace = faces.first, let landmarks = firstFace.landmarks {
                    currentSignature = extractFaceSignature(from: landmarks, boundingBox: firstFace.boundingBox)
                }
                
                // 3. Perform Identity Verification against Enrolled Profile
                var matchResult = VerificationResult(
                    isMatched: false,
                    confidence: 0.0,
                    profileName: profile?.name ?? "Unknown",
                    statusMessage: "No face detected to verify."
                )
                
                if let enrolledProfile = profile, let currentSig = currentSignature, !enrolledProfile.signatures.isEmpty {
                    // Compare against all enrolled reference signatures
                    var highestSimilarity = 0.0
                    for enrolledSig in enrolledProfile.signatures {
                        let sim = currentSig.similarity(to: enrolledSig)
                        if sim > highestSimilarity {
                            highestSimilarity = sim
                        }
                    }
                    
                    let isMatched = highestSimilarity >= 0.72
                    let confidencePct = Int(highestSimilarity * 100)
                    
                    matchResult = VerificationResult(
                        isMatched: isMatched,
                        confidence: highestSimilarity,
                        profileName: enrolledProfile.name,
                        statusMessage: isMatched ?
                            "Matched with enrolled profile \(enrolledProfile.name) (\(confidencePct)% similarity)" :
                            "Subject does not match enrolled profile for \(enrolledProfile.name)."
                    )
                } else if profile == nil {
                    matchResult = VerificationResult(
                        isMatched: true,
                        confidence: 1.0,
                        profileName: "Guest",
                        statusMessage: "Guest mode (No profile enrolled)."
                    )
                }
                
                // 4. Determine Image Scope
                let scope = determineImageScope(faces: faces, bodyPoses: bodyPoses, imageSize: CGSize(width: cgImage.width, height: cgImage.height))
                
                // 5. Compute Style & Aesthetic Rating
                var computedRating: StyleRating?
                if matchResult.isMatched || profile == nil {
                    computedRating = computeStyleRating(faces: faces, faceQualityRequest: faceQualityRequest, bodyPoses: bodyPoses, classifications: classifications, cgImage: cgImage)
                }
                
                // 6. Generate Category Reports
                var categories: [VisualReportCategory] = []
                
                // Framing & Pose
                let stanceItems = analyzeFramingAndPose(scope: scope, bodyPoses: bodyPoses, faces: faces)
                if !stanceItems.isEmpty {
                    categories.append(VisualReportCategory(title: "Framing & Stance", icon: "figure.walk", color: .blue, items: stanceItems))
                }
                
                // Hairstyle, Eyewear & Grooming
                let headItems = analyzeHeadHairstyleAndEyewear(faces: faces, classifications: classifications, cgImage: cgImage, orientation: orientation)
                if !headItems.isEmpty {
                    categories.append(VisualReportCategory(title: "Hair, Eyewear & Grooming", icon: "eyeglasses", color: .orange, items: headItems))
                }
                
                // Apparel & Outfits
                let apparelItems = analyzeApparelAndOutfit(scope: scope, bodyPoses: bodyPoses, classifications: classifications, cgImage: cgImage)
                if !apparelItems.isEmpty {
                    categories.append(VisualReportCategory(title: "Apparel & Outfit Analysis", icon: "tshirt.fill", color: .purple, items: apparelItems))
                }
                
                // Capture Quality & Environment
                let contextItems = analyzeCaptureAndEnvironment(faces: faces, faceQualityRequest: faceQualityRequest, classifications: classifications)
                if !contextItems.isEmpty {
                    categories.append(VisualReportCategory(title: "Lighting & Setting", icon: "sun.max.fill", color: .green, items: contextItems))
                }
                
                DispatchQueue.main.async {
                    self.verificationResult = matchResult
                    self.styleRating = computedRating
                    self.detectedScope = scope
                    self.reportCategories = categories
                    self.isAnalyzing = false
                }
                
            } catch {
                print("Pipeline error: \(error)")
                DispatchQueue.main.async {
                    self.isAnalyzing = false
                }
            }
        }
    }
    
    // MARK: - Style & Aesthetic Scoring Engine
    private func computeStyleRating(faces: [VNFaceObservation], faceQualityRequest: VNDetectFaceCaptureQualityRequest, bodyPoses: [VNHumanBodyPoseObservation], classifications: [VNClassificationObservation], cgImage: CGImage) -> StyleRating {
        var baseScore = 7.5
        
        // Lighting & Quality Factor
        var qualityScore = 0.7
        if let qualityFace = faceQualityRequest.results?.first, let q = qualityFace.faceCaptureQuality {
            qualityScore = Double(q)
            baseScore += (qualityScore - 0.5) * 2.0 // +/- 1.0
        }
        
        // Posture Factor
        if let body = bodyPoses.first {
            if let neck = try? body.recognizedPoint(.neck), let root = try? body.recognizedPoint(.root), neck.confidence > 0.3, root.confidence > 0.3 {
                let dx = abs(neck.location.x - root.location.x)
                if dx < 0.04 {
                    baseScore += 0.6 // Good upright posture
                }
            }
        }
        
        // Clothing & Styling Factors
        let formalOrSharpKeywords = ["suit", "blazer", "tie", "jacket", "shirt", "dress", "uniform"]
        let casualKeywords = ["t-shirt", "sweater", "hoodie", "denim", "jersey"]
        
        let isFormal = classifications.contains { item in
            formalOrSharpKeywords.contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.3
        }
        let isCasual = classifications.contains { item in
            casualKeywords.contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.3
        }
        
        if isFormal { baseScore += 0.8 }
        else if isCasual { baseScore += 0.4 }
        
        // Eyewear / Accessory bonus
        let hasAccessories = classifications.contains { item in
            ["sunglass", "glasses", "watch", "necklace", "jewelry"].contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.3
        }
        if hasAccessories { baseScore += 0.4 }
        
        let finalScore = max(6.0, min(9.9, baseScore))
        
        let verdict: String
        let title: String
        if finalScore >= 9.0 {
            title = "Superb & Sharp"
            verdict = "Exceptional presentation with high visual balance and crisp coordination."
        } else if finalScore >= 8.2 {
            title = "Well-Coordinated"
            verdict = "Polished look with harmonious tones, confident posture, and clear framing."
        } else if finalScore >= 7.2 {
            title = "Clean & Casual"
            verdict = "Comfortable, natural everyday style with well-balanced lighting."
        } else {
            title = "Relaxed Look"
            verdict = "Casual appearance with soft lighting and natural framing."
        }
        
        let colorVerdict = isFormal ? "Sharp & High Contrast" : "Harmonious & Balanced"
        let groomingVerdict = hasAccessories ? "Accessorized & Styled" : "Clean & Natural"
        let lightingVerdict = qualityScore > 0.65 ? "Bright & Crisp (Studio/Daylight)" : "Soft Ambient Lighting"
        
        return StyleRating(
            overallScore: finalScore,
            title: title,
            styleVerdict: verdict,
            colorHarmony: colorVerdict,
            groomingRating: groomingVerdict,
            lightingClarity: lightingVerdict
        )
    }
    
    // MARK: - Biometric Extraction Helper
    private func extractFaceSignature(from landmarks: VNFaceLandmarks2D, boundingBox: CGRect) -> FaceBiometricSignature? {
        guard let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let nose = landmarks.nose ?? landmarks.noseCrest,
              let lips = landmarks.outerLips else { return nil }
        
        let leftEyeCenter = calculateCenter(points: leftEye.normalizedPoints)
        let rightEyeCenter = calculateCenter(points: rightEye.normalizedPoints)
        let noseTip = calculateCenter(points: nose.normalizedPoints)
        let mouthCenter = calculateCenter(points: lips.normalizedPoints)
        
        let eyeDistance = distance(from: leftEyeCenter, to: rightEyeCenter)
        guard eyeDistance > 0.01 else { return nil }
        
        let eyeMidpoint = CGPoint(x: (leftEyeCenter.x + rightEyeCenter.x)/2, y: (leftEyeCenter.y + rightEyeCenter.y)/2)
        
        let eyeToNose = distance(from: eyeMidpoint, to: noseTip) / eyeDistance
        let eyeToMouth = distance(from: eyeMidpoint, to: mouthCenter) / eyeDistance
        
        var noseToChinRatio = 1.2
        if let contour = landmarks.faceContour, let chinPoint = contour.normalizedPoints.last {
            noseToChinRatio = distance(from: noseTip, to: chinPoint) / eyeDistance
        }
        
        let mouthWidth = lips.normalizedPoints.map { $0.x }.max() ?? 0 - (lips.normalizedPoints.map { $0.x }.min() ?? 0)
        let mouthWidthRatio = Double(mouthWidth) / Double(eyeDistance)
        
        var jawWidthRatio = 2.0
        if let contour = landmarks.faceContour {
            let jawPoints = contour.normalizedPoints
            if let first = jawPoints.first, let last = jawPoints.last {
                jawWidthRatio = distance(from: first, to: last) / eyeDistance
            }
        }
        
        let aspectRatio = Double(boundingBox.height / max(0.01, boundingBox.width))
        
        return FaceBiometricSignature(
            eyeToNoseRatio: eyeToNose,
            eyeToMouthRatio: eyeToMouth,
            noseToChinRatio: noseToChinRatio,
            mouthWidthRatio: mouthWidthRatio,
            jawWidthRatio: jawWidthRatio,
            faceAspectRatio: aspectRatio
        )
    }
    
    private func calculateCenter(points: [CGPoint]) -> CGPoint {
        guard !points.isEmpty else { return .zero }
        let totalX = points.reduce(0) { $0 + $1.x }
        let totalY = points.reduce(0) { $0 + $1.y }
        return CGPoint(x: totalX / CGFloat(points.count), y: totalY / CGFloat(points.count))
    }
    
    private func distance(from: CGPoint, to: CGPoint) -> Double {
        let dx = Double(from.x - to.x)
        let dy = Double(from.y - to.y)
        return sqrt(dx*dx + dy*dy)
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
    
    // MARK: - Framing & Pose Analysis
    private func analyzeFramingAndPose(scope: VisualScope, bodyPoses: [VNHumanBodyPoseObservation], faces: [VNFaceObservation]) -> [String] {
        var items: [String] = []
        items.append("Framing Category: \(scope.rawValue)")
        
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
                items.append("Arm Position: Relaxed posture at sides")
            }
            
            if let neck = try? body.recognizedPoint(.neck), let root = try? body.recognizedPoint(.root), neck.confidence > 0.3, root.confidence > 0.3 {
                let dx = abs(neck.location.x - root.location.x)
                if dx < 0.05 {
                    items.append("Body Posture: Upright and centered")
                } else {
                    items.append("Body Posture: Angled / dynamic stance")
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
                items.append("Head Tilt: Tilted head orientation")
            }
        }
        
        return items
    }
    
    // MARK: - Hairstyle, Eyewear & Grooming Analysis
    private func analyzeHeadHairstyleAndEyewear(faces: [VNFaceObservation], classifications: [VNClassificationObservation], cgImage: CGImage, orientation: CGImagePropertyOrientation) -> [String] {
        var items: [String] = []
        
        let eyewearMatches = classifications.filter {
            let id = $0.identifier.lowercased()
            return (id.contains("sunglass") || id.contains("spectacles") || id.contains("glasses") || id.contains("eyewear") || id.contains("goggles")) && $0.confidence > 0.3
        }
        
        if let topEyewear = eyewearMatches.first {
            items.append("Eyewear: Wearing \(topEyewear.identifier.capitalized)")
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
            items.append("Hair & Facial Styling: \(match.identifier.capitalized)")
        }
        
        if let face = faces.first, let landmarks = face.landmarks {
            var facialDetails: [String] = []
            if landmarks.leftEye != nil && landmarks.rightEye != nil {
                facialDetails.append("Eyes open & visible")
            }
            if landmarks.outerLips != nil {
                facialDetails.append("Mouth / smile line tracked")
            }
            if landmarks.faceContour != nil {
                facialDetails.append("Defined jawline contour")
            }
            if !facialDetails.isEmpty {
                items.append("Facial Biometrics: " + facialDetails.joined(separator: ", "))
            }
        }
        
        return items
    }
    
    // MARK: - Apparel & Outfit Analysis
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
                items.append("Outfit Category: \(apparel.identifier.capitalized)")
            }
        } else {
            items.append("Apparel Style: Casual / daily wear")
        }
        
        let sampleRect = CGRect(x: 0.35, y: 0.45, width: 0.30, height: 0.25)
        if let colorName = sampleDominantColorName(in: cgImage, normalizedRect: sampleRect) {
            items.append("Upper Body Color Tone: Dominant \(colorName)")
        }
        
        if scope == .fullBody {
            let lowerSampleRect = CGRect(x: 0.35, y: 0.70, width: 0.30, height: 0.20)
            if let lowerColor = sampleDominantColorName(in: cgImage, normalizedRect: lowerSampleRect) {
                items.append("Lower Body Color Tone: Dominant \(lowerColor)")
            }
        }
        
        let accessoryKeywords = ["watch", "backpack", "bag", "purse", "necklace", "earring", "bracelet", "scarf", "glove"]
        let accessoryMatches = classifications.filter { item in
            let id = item.identifier.lowercased()
            return accessoryKeywords.contains { kw in id.contains(kw) } && item.confidence > 0.25
        }
        for acc in accessoryMatches.prefix(2) {
            items.append("Accessory Detected: \(acc.identifier.capitalized)")
        }
        
        return items
    }
    
    // MARK: - Capture Quality & Environment
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
            items.append("Environment Setting: \(topEnv.identifier.capitalized)")
        }
        
        return items
    }
    
    // MARK: - Color Sampling Utility
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
            if g > 0.5 { return "Yellow / Warm tone" }
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

// MARK: - Metric Badge Component
struct MetricBadge: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.subheadline)
            
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(value)
                    .font(.caption.bold())
                    .lineLimit(1)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(UIColor.tertiarySystemBackground))
        .cornerRadius(10)
    }
}

// MARK: - Profile Onboarding / Enrollment View
struct ProfileOnboardingView: View {
    @Binding var profile: UserProfile?
    @Environment(\.presentationMode) var presentationMode
    
    @State private var name: String = ""
    @State private var selectedImages: [UIImage] = []
    @State private var isProcessing = false
    @State private var errorMessage: String?
    @State private var showingImagePicker = false
    @State private var pickerImage: UIImage?
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    VStack(spacing: 8) {
                        Image(systemName: "person.text.rectangle.fill")
                            .font(.system(size: 48))
                            .foregroundColor(.blue)
                        Text(profile == nil ? "Create Your Profile" : "Edit Profile")
                            .font(.title2.bold())
                        Text("Add your name and 1 to 3 clear reference photos of your face for accurate biometric matching.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .padding(.top, 10)
                    
                    // Name Field
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Your Name")
                            .font(.subheadline.bold())
                        TextField("Enter full name", text: $name)
                            .padding()
                            .background(Color(UIColor.secondarySystemBackground))
                            .cornerRadius(12)
                    }
                    
                    // Reference Photos Grid
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Reference Photos (\(selectedImages.count)/3)")
                                .font(.subheadline.bold())
                            Spacer()
                            if selectedImages.count < 3 {
                                Button(action: { showingImagePicker = true }) {
                                    Label("Add Photo", systemImage: "plus.circle.fill")
                                        .font(.subheadline.bold())
                                }
                            }
                        }
                        
                        HStack(spacing: 12) {
                            ForEach(Array(selectedImages.enumerated()), id: \.offset) { index, img in
                                ZStack(alignment: .topTrailing) {
                                    Image(uiImage: img)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(width: 96, height: 96)
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                    
                                    Button(action: {
                                        selectedImages.remove(at: index)
                                    }) {
                                        Image(systemName: "xmark.circle.fill")
                                            .foregroundColor(.white)
                                            .background(Circle().fill(Color.black.opacity(0.6)))
                                    }
                                    .padding(4)
                                }
                            }
                            
                            if selectedImages.count < 3 {
                                Button(action: { showingImagePicker = true }) {
                                    VStack(spacing: 6) {
                                        Image(systemName: "camera.fill")
                                            .font(.title3)
                                        Text("Add #\(selectedImages.count + 1)")
                                            .font(.caption2.bold())
                                    }
                                    .foregroundColor(.blue)
                                    .frame(width: 96, height: 96)
                                    .background(Color.blue.opacity(0.08))
                                    .cornerRadius(12)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [4]))
                                            .foregroundColor(.blue.opacity(0.5))
                                    )
                                }
                            }
                        }
                    }
                    
                    if let err = errorMessage {
                        Text(err)
                            .font(.caption)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                    }
                    
                    // Save Button
                    Button(action: enrollProfile) {
                        if isProcessing {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                        } else {
                            Text("Save Profile")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(canSave ? Color.blue : Color.gray.opacity(0.4))
                                .foregroundColor(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                    }
                    .disabled(!canSave || isProcessing)
                    
                    if profile != nil {
                        Button("Delete Current Profile", role: .destructive) {
                            UserProfile.clear()
                            profile = nil
                            presentationMode.wrappedValue.dismiss()
                        }
                        .font(.subheadline)
                        .padding(.top, 4)
                    }
                }
                .padding()
            }
            .navigationBarItems(trailing: Button("Cancel") {
                presentationMode.wrappedValue.dismiss()
            })
            .sheet(isPresented: $showingImagePicker) {
                ImagePicker(image: $pickerImage, sourceType: .photoLibrary)
                    .onDisappear {
                        if let img = pickerImage {
                            selectedImages.append(img)
                            pickerImage = nil
                        }
                    }
            }
            .onAppear {
                if let current = profile {
                    name = current.name
                    selectedImages = current.photoDataList.compactMap { UIImage(data: $0) }
                }
            }
        }
    }
    
    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespaces).isEmpty && !selectedImages.isEmpty
    }
    
    private func enrollProfile() {
        isProcessing = true
        errorMessage = nil
        
        DispatchQueue.global(qos: .userInitiated).async {
            var extractedSignatures: [FaceBiometricSignature] = []
            var validPhotoDataList: [Data] = []
            
            let request = VNDetectFaceLandmarksRequest()
            
            for image in selectedImages {
                guard let cgImage = image.cgImage else { continue }
                let orientation = CGImagePropertyOrientation(image.imageOrientation)
                let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
                
                try? handler.perform([request])
                if let face = request.results?.first, let landmarks = face.landmarks {
                    if let sig = extractSignature(from: landmarks, boundingBox: face.boundingBox) {
                        extractedSignatures.append(sig)
                        if let data = image.jpegData(compressionQuality: 0.7) {
                            validPhotoDataList.append(data)
                        }
                    }
                }
            }
            
            DispatchQueue.main.async {
                self.isProcessing = false
                if extractedSignatures.isEmpty {
                    self.errorMessage = "No clear face could be detected in the chosen photos. Please select clearer front-facing photos."
                } else {
                    let newProfile = UserProfile(
                        name: self.name.trimmingCharacters(in: .whitespaces),
                        photoDataList: validPhotoDataList,
                        signatures: extractedSignatures,
                        dateCreated: Date()
                    )
                    newProfile.save()
                    self.profile = newProfile
                    self.presentationMode.wrappedValue.dismiss()
                }
            }
        }
    }
    
    private func extractSignature(from landmarks: VNFaceLandmarks2D, boundingBox: CGRect) -> FaceBiometricSignature? {
        guard let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let nose = landmarks.nose ?? landmarks.noseCrest,
              let lips = landmarks.outerLips else { return nil }
        
        let leftEyeCenter = calculateCenter(points: leftEye.normalizedPoints)
        let rightEyeCenter = calculateCenter(points: rightEye.normalizedPoints)
        let noseTip = calculateCenter(points: nose.normalizedPoints)
        let mouthCenter = calculateCenter(points: lips.normalizedPoints)
        
        let eyeDistance = distance(from: leftEyeCenter, to: rightEyeCenter)
        guard eyeDistance > 0.01 else { return nil }
        
        let eyeMidpoint = CGPoint(x: (leftEyeCenter.x + rightEyeCenter.x)/2, y: (leftEyeCenter.y + rightEyeCenter.y)/2)
        
        let eyeToNose = distance(from: eyeMidpoint, to: noseTip) / eyeDistance
        let eyeToMouth = distance(from: eyeMidpoint, to: mouthCenter) / eyeDistance
        
        var noseToChinRatio = 1.2
        if let contour = landmarks.faceContour, let chinPoint = contour.normalizedPoints.last {
            noseToChinRatio = distance(from: noseTip, to: chinPoint) / eyeDistance
        }
        
        let mouthWidth = lips.normalizedPoints.map { $0.x }.max() ?? 0 - (lips.normalizedPoints.map { $0.x }.min() ?? 0)
        let mouthWidthRatio = Double(mouthWidth) / Double(eyeDistance)
        
        var jawWidthRatio = 2.0
        if let contour = landmarks.faceContour {
            let jawPoints = contour.normalizedPoints
            if let first = jawPoints.first, let last = jawPoints.last {
                jawWidthRatio = distance(from: first, to: last) / eyeDistance
            }
        }
        
        let aspectRatio = Double(boundingBox.height / max(0.01, boundingBox.width))
        
        return FaceBiometricSignature(
            eyeToNoseRatio: eyeToNose,
            eyeToMouthRatio: eyeToMouth,
            noseToChinRatio: noseToChinRatio,
            mouthWidthRatio: mouthWidthRatio,
            jawWidthRatio: jawWidthRatio,
            faceAspectRatio: aspectRatio
        )
    }
    
    private func calculateCenter(points: [CGPoint]) -> CGPoint {
        guard !points.isEmpty else { return .zero }
        let totalX = points.reduce(0) { $0 + $1.x }
        let totalY = points.reduce(0) { $0 + $1.y }
        return CGPoint(x: totalX / CGFloat(points.count), y: totalY / CGFloat(points.count))
    }
    
    private func distance(from: CGPoint, to: CGPoint) -> Double {
        let dx = Double(from.x - to.x)
        let dy = Double(from.y - to.y)
        return sqrt(dx*dx + dy*dy)
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
    @Published var cameraPosition: AVCaptureDevice.Position = .front
    @Published var isGridVisible = false
    @Published var currentZoomFactor: CGFloat = 1.0
    @Published var focusPoint: CGPoint?
    @Published var isFocusing = false
    
    private var photoOutput = AVCapturePhotoOutput()
    private var currentDeviceInput: AVCaptureDeviceInput?
    private var countdownTimer: Timer?
    var onPhotoCaptured: ((UIImage) -> Void)?
    
    var isUltraWideAvailable: Bool {
        return minZoomFactor < 0.95
    }
    
    var isTelephotoAvailable: Bool {
        return maxZoomFactor >= 2.0
    }
    
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
                    
                    if camera.isGridVisible {
                        CameraGridView()
                    }
                    
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
                HStack(spacing: 20) {
                    // Flash Mode
                    Button(action: { camera.toggleFlash() }) {
                        Image(systemName: flashIconName)
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(camera.flashMode == .off ? .white : .yellow)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                    }
                    
                    // Timer Menu
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
                    Button(action: { camera.isGridVisible.toggle() }) {
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
                
                // Zoom Switcher (Only display supported zoom options)
                if camera.isUltraWideAvailable || camera.isTelephotoAvailable {
                    HStack(spacing: 12) {
                        if camera.isUltraWideAvailable {
                            Button(action: {
                                let target = camera.minZoomFactor
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
                        
                        if camera.isTelephotoAvailable {
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
                    }
                    .padding(.bottom, 16)
                }
                
                // Bottom Control Bar
                HStack(alignment: .center) {
                    Button(action: { showingLibraryPicker = true }) {
                        Image(systemName: "photo.on.rectangle")
                            .font(.system(size: 24))
                            .foregroundColor(.white)
                            .frame(width: 54, height: 54)
                            .background(Color.white.opacity(0.2))
                            .clipShape(Circle())
                    }
                    
                    Spacer()
                    
                    Button(action: { camera.triggerCapture() }) {
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
                    
                    Button(action: { camera.switchCamera() }) {
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
                
                path.move(to: CGPoint(x: w / 3, y: 0))
                path.addLine(to: CGPoint(x: w / 3, y: h))
                path.move(to: CGPoint(x: 2 * w / 3, y: 0))
                path.addLine(to: CGPoint(x: 2 * w / 3, y: h))
                
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