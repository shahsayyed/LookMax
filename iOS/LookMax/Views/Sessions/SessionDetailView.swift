import SwiftUI
import Vision
import AVFoundation

struct SessionDetailView: View {
    @Binding var session: LookSession
    let profile: UserProfile?

    @Environment(\.presentationMode) var presentationMode
    @State private var selectedLookId: UUID?
    @State private var showingDeleteLookConfirm = false
    @State private var lookToDelete: LookItem?
    @State private var isDeepAnalyzing = false      // Gemini VLM call in flight
    @State private var deepAnalysisError: String?   // Shown as a dismissable banner
    @State private var showingCustomCamera = false
    @State private var showingLibraryPicker = false
    @State private var showingComparison = false
    @State private var incomingImage: UIImage?
    @State private var isAnalyzing = false

    private var selectedLook: LookItem? {
        guard let id = selectedLookId else { return session.looks.first }
        return session.looks.first(where: { $0.id == id })
    }

    var body: some View {
        NavigationView {
            ZStack {
                Theme.oledBlack.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 18) {
                        if session.looks.isEmpty {
                            VStack(spacing: 16) {
                                Image(systemName: "camera.viewfinder")
                                    .font(.system(size: 60))
                                    .foregroundColor(Theme.neonCyan)
                                    .neonGlow(color: Theme.neonCyan, radius: 12)

                                Text("Add Your First Look")
                                    .font(.title3.bold())
                                    .foregroundColor(.white)

                                Text("Capture a photo to get instant AI biometric feedback and 5-min tweaks for \"\(session.title)\".")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 28)
                            }
                            .padding(.vertical, 40)
                        } else {
                            // Look Carousel with glowing border
                            LookCarouselView(
                                looks: session.looks,
                                selectedId: $selectedLookId,
                                onDelete: { look in
                                    lookToDelete = look
                                    showingDeleteLookConfirm = true
                                }
                            )

                            // Side-by-side bar chart comparison
                            if session.looks.count > 1 {
                                ScoreComparisonBar(
                                    looks: session.looks,
                                    selectedId: selectedLookId,
                                    onCompareTapped: { showingComparison = true }
                                )
                                .padding(.horizontal)
                            }

                            // Active Look Detail Card
                            if let look = selectedLook {
                                LookDetailCard(
                                    look: look,
                                    isBestLook: look.id == session.bestLook?.id,
                                    onCompareTapped: { showingComparison = true }
                                )
                                .padding(.horizontal)
                            }
                        }

                        // Capture CTAs
                        VStack(spacing: 12) {
                            Button(action: { checkCameraPermission() }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "camera.fill")
                                    Text(session.looks.isEmpty ? "Take Photo (Biometric Camera)" : "Add Another Look (Camera)")
                                }
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Theme.neonGradient)
                                .foregroundColor(.black)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                                .shadow(color: Theme.neonCyan.opacity(0.4), radius: 8)
                            }

                            Button(action: { showingLibraryPicker = true }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "photo.on.rectangle")
                                    Text("Choose from Library")
                                }
                                .font(.subheadline.bold())
                                .frame(maxWidth: .infinity)
                                .padding(14)
                                .glassCard(cornerRadius: 14)
                                .foregroundColor(.white)
                            }
                        }
                        .padding(.horizontal)

                        // Phase 1: Vision (instant, on-device)
                        if isAnalyzing {
                            HStack(spacing: 12) {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: Theme.neonCyan))
                                Text("Biometric scan in progress…")
                                    .font(.subheadline.bold())
                                    .foregroundColor(Theme.neonCyan)
                            }
                            .padding()
                            .glassCard(cornerRadius: 14)
                            .padding(.horizontal)
                        }

                        // Phase 2: Gemini VLM deep analysis
                        if isDeepAnalyzing {
                            HStack(spacing: 12) {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: Theme.emerald))

                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Deep AI Style Analysis")
                                        .font(.subheadline.bold())
                                        .foregroundColor(.white)
                                    Text("Gemini is evaluating formality, fit & occasion match…")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                            }
                            .padding()
                            .background(
                                RoundedRectangle(cornerRadius: 14)
                                    .fill(Theme.emerald.opacity(0.1))
                                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(Theme.emerald.opacity(0.3), lineWidth: 1))
                            )
                            .padding(.horizontal)
                        }

                        // VLM error banner (dismissable)
                        if let err = deepAnalysisError {
                            HStack(spacing: 8) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundColor(Theme.warmAmber)
                                Text(err)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Spacer()
                                Button("Dismiss") { deepAnalysisError = nil }
                                    .font(.caption.bold())
                                    .foregroundColor(Theme.warmAmber)
                            }
                            .padding(12)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Theme.warmAmber.opacity(0.08))
                                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.warmAmber.opacity(0.3), lineWidth: 1))
                            )
                            .padding(.horizontal)
                        }
                    }
                    .padding(.top, 12)
                    .padding(.bottom, 40)
                }
            }
            .navigationTitle(session.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: { presentationMode.wrappedValue.dismiss() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 18))
                            .foregroundColor(.secondary)
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    HStack(spacing: 14) {
                        if session.looks.count >= 2 {
                            Button(action: { showingComparison = true }) {
                                Image(systemName: "slider.horizontal.below.rectangle")
                                    .foregroundColor(Theme.neonCyan)
                            }
                        }
                    }
                }
            }
            .fullScreenCover(isPresented: $showingCustomCamera) {
                CustomCameraView(isPresented: $showingCustomCamera, occasion: session.occasion) { image in
                    incomingImage = image
                    analyzeAndAddLook()
                }
                .ignoresSafeArea()
            }
            .sheet(isPresented: $showingLibraryPicker) {
                ImagePicker(image: $incomingImage, sourceType: .photoLibrary)
                    .onDisappear { if incomingImage != nil { analyzeAndAddLook() } }
            }
            .confirmationDialog(
                "Delete this look?",
                isPresented: $showingDeleteLookConfirm,
                titleVisibility: .visible
            ) {
                Button("Delete Look", role: .destructive) {
                    if let look = lookToDelete {
                        withAnimation {
                            SessionStorageManager.shared.deleteLook(look, from: &session)
                            if selectedLookId == look.id {
                                selectedLookId = session.looks.first?.id
                            }
                            HapticManager.medium()
                        }
                    }
                }
                Button("Cancel", role: .cancel) {}
            }
            .sheet(isPresented: $showingComparison) {
                if session.looks.count >= 2 {
                    BeforeAfterComparisonView(
                        look1: session.looks.first!,
                        look2: session.looks.last!,
                        occasion: session.occasion
                    )
                }
            }
        }
    }

    private func checkCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: showingCustomCamera = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { if granted { showingCustomCamera = true } }
            }
        default: showingCustomCamera = true
        }
    }

    private func analyzeAndAddLook() {
        guard let uiImage = incomingImage else { return }
        incomingImage = nil
        isAnalyzing = true

        // ─── Phase 1: On-Device Vision Analysis (instant) ───
        DispatchQueue.global(qos: .userInitiated).async {
            guard let cgImage = uiImage.cgImage else {
                DispatchQueue.main.async { isAnalyzing = false }
                return
            }

            let orientation = CGImagePropertyOrientation(uiImage.imageOrientation)
            let faceRequest = VNDetectFaceLandmarksRequest()
            let faceQualityRequest = VNDetectFaceCaptureQualityRequest()
            let bodyPoseRequest = VNDetectHumanBodyPoseRequest()
            let classificationRequest = VNClassifyImageRequest()

            let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
            try? handler.perform([faceRequest, faceQualityRequest, bodyPoseRequest, classificationRequest])

            let visionAnalysis = LookAnalysisEngine.analyze(
                faces: faceRequest.results ?? [],
                faceQualityRequest: faceQualityRequest,
                bodyPoses: bodyPoseRequest.results ?? [],
                classifications: classificationRequest.results ?? [],
                cgImage: cgImage,
                occasion: session.occasion
            )

            let imagePath = SessionStorageManager.shared.saveImage(uiImage)

            // Build initial look from Vision scores (shown immediately to user)
            let initialLook = LookItem(
                imagePath: imagePath,
                score: visionAnalysis.score,
                headlineBadge: visionAnalysis.headlineBadge,
                goodPoints: visionAnalysis.goodPoints,
                badPoints: visionAnalysis.badPoints,
                suggestions: visionAnalysis.suggestions,
                detectedOutfitColor: visionAnalysis.detectedOutfitColor,
                detectedFaceShape: visionAnalysis.detectedFaceShape,
                lightingScore: visionAnalysis.lightingScore,
                postureScore: visionAnalysis.postureScore,
                fitScore: visionAnalysis.fitScore,
                groomingScore: visionAnalysis.groomingScore,
                postureNote: visionAnalysis.postureNote,
                fitNote: visionAnalysis.fitNote,
                styleNote: visionAnalysis.styleNote
            )

            DispatchQueue.main.async {
                session.looks.append(initialLook)
                SessionStorageManager.shared.updateSession(session)
                selectedLookId = initialLook.id
                isAnalyzing = false
                HapticManager.success()

                // ─── Phase 2: Gemini VLM Deep Analysis (async, enriches the look) ───
                isDeepAnalyzing = true
                deepAnalysisError = nil

                Task {
                    defer { isDeepAnalyzing = false }
                    do {
                        let geminiResult = try await GeminiVisionService.shared.evaluate(
                            image: uiImage,
                            occasion: session.occasion
                        )

                        // Merge Gemini scores with Vision result
                        let merged = LookAnalysisEngine.merge(
                            gemini: geminiResult,
                            visionResult: visionAnalysis
                        )

                        // Find and update the look we just added
                        if let idx = session.looks.firstIndex(where: { $0.id == initialLook.id }) {
                            session.looks[idx] = LookItem(
                                id: initialLook.id,
                                imagePath: imagePath,
                                score: merged.score,
                                headlineBadge: merged.headlineBadge,
                                goodPoints: merged.goodPoints,
                                badPoints: merged.badPoints,
                                suggestions: merged.suggestions,
                                detectedOutfitColor: merged.detectedOutfitColor,
                                detectedFaceShape: merged.detectedFaceShape,
                                lightingScore: merged.lightingScore,
                                postureScore: merged.postureScore,
                                fitScore: merged.fitScore,
                                groomingScore: merged.groomingScore,
                                postureNote: merged.postureNote,
                                fitNote: merged.fitNote,
                                styleNote: merged.styleNote
                            )
                            SessionStorageManager.shared.updateSession(session)
                            HapticManager.medium()  // Subtle confirmation that deep analysis landed
                        }
                    } catch GeminiServiceError.missingAPIKey {
                        // Silent fail when no API key — Vision scores are still shown
                        print("[GeminiVisionService] No API key configured. Using on-device Vision scores only.")
                    } catch {
                        deepAnalysisError = error.localizedDescription
                    }
                }
            }
        }
    }
}
