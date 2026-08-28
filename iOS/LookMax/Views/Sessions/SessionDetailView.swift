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

                        if isAnalyzing {
                            HStack(spacing: 12) {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: Theme.neonCyan))
                                Text("AI Consultant analyzing pose & style…")
                                    .font(.subheadline)
                                    .foregroundColor(Theme.neonCyan)
                            }
                            .padding()
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

            let analysis = LookAnalysisEngine.analyze(
                faces: faceRequest.results ?? [],
                faceQualityRequest: faceQualityRequest,
                bodyPoses: bodyPoseRequest.results ?? [],
                classifications: classificationRequest.results ?? [],
                cgImage: cgImage,
                occasion: session.occasion
            )

            let imagePath = SessionStorageManager.shared.saveImage(uiImage)

            let lookItem = LookItem(
                imagePath: imagePath,
                score: analysis.score,
                headlineBadge: analysis.headlineBadge,
                goodPoints: analysis.goodPoints,
                badPoints: analysis.badPoints,
                suggestions: analysis.suggestions,
                detectedOutfitColor: analysis.detectedOutfitColor,
                detectedFaceShape: analysis.detectedFaceShape,
                lightingScore: analysis.lightingScore,
                postureScore: analysis.postureScore,
                fitScore: analysis.fitScore,
                groomingScore: analysis.groomingScore,
                postureNote: analysis.postureNote,
                fitNote: analysis.fitNote,
                styleNote: analysis.styleNote
            )

            DispatchQueue.main.async {
                session.looks.append(lookItem)
                SessionStorageManager.shared.updateSession(session)
                selectedLookId = lookItem.id
                isAnalyzing = false
                HapticManager.success()
            }
        }
    }
}
