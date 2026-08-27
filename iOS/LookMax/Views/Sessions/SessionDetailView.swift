import SwiftUI
import Vision
import AVFoundation

struct SessionDetailView: View {
    @Binding var session: LookSession
    let profile: UserProfile?

    @State private var selectedLookId: UUID?
    @State private var showingCustomCamera = false
    @State private var showingLibraryPicker = false
    @State private var incomingImage: UIImage?
    @State private var isAnalyzing = false

    private var selectedLook: LookItem? {
        guard let id = selectedLookId else { return session.looks.first }
        return session.looks.first(where: { $0.id == id })
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 18) {
                    if session.looks.isEmpty {
                        VStack(spacing: 16) {
                            Image(systemName: "camera.viewfinder")
                                .font(.system(size: 54)).foregroundColor(.purple.opacity(0.6))
                            Text("Add Your First Look").font(.title3.bold())
                            Text("Take or choose a photo to get instant AI styling feedback for \"\(session.title)\".")
                                .font(.subheadline).foregroundColor(.secondary)
                                .multilineTextAlignment(.center).padding(.horizontal, 28)
                        }
                        .padding(.vertical, 40)
                    } else {
                        LookCarouselView(looks: session.looks, selectedId: $selectedLookId)

                        if session.looks.count > 1 {
                            ScoreComparisonBar(looks: session.looks, selectedId: selectedLookId)
                                .padding(.horizontal)
                        }

                        if let look = selectedLook {
                            LookDetailCard(look: look, isBestLook: look.id == session.bestLook?.id)
                                .padding(.horizontal)
                        }
                    }

                    // Action Buttons
                    VStack(spacing: 10) {
                        Button(action: { checkCameraPermission() }) {
                            Label(
                                session.looks.isEmpty ? "Add Look with Camera" : "Add Another Look (Camera)",
                                systemImage: "camera.fill"
                            )
                            .font(.headline)
                            .frame(maxWidth: .infinity).padding()
                            .background(Color.blue).foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                        Button(action: { showingLibraryPicker = true }) {
                            Label("Choose from Photo Library", systemImage: "photo.on.rectangle")
                                .font(.headline)
                                .frame(maxWidth: .infinity).padding()
                                .background(Color(UIColor.secondarySystemBackground)).foregroundColor(.primary)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                    }
                    .padding(.horizontal)

                    if isAnalyzing {
                        HStack(spacing: 12) {
                            ProgressView()
                            Text("AI Consultant Analyzing your look…")
                                .font(.subheadline).foregroundColor(.secondary)
                        }
                        .padding()
                    }
                }
                .padding(.top, 12).padding(.bottom, 40)
            }
            .navigationTitle(session.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if let best = session.bestLook {
                        HStack(spacing: 4) {
                            Image(systemName: session.occasion.icon).foregroundColor(session.occasion.color)
                            Text(String(format: "%.1f", best.score)).font(.subheadline.bold()).foregroundColor(.purple)
                        }
                    }
                }
            }
            .fullScreenCover(isPresented: $showingCustomCamera) {
                CustomCameraView(isPresented: $showingCustomCamera) { image in
                    incomingImage = image
                    analyzeAndAddLook()
                }
                .ignoresSafeArea()
            }
            .sheet(isPresented: $showingLibraryPicker) {
                ImagePicker(image: $incomingImage, sourceType: .photoLibrary)
                    .onDisappear { if incomingImage != nil { analyzeAndAddLook() } }
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
                lightingScore: analysis.lightingScore
            )

            DispatchQueue.main.async {
                session.looks.append(lookItem)
                SessionStorageManager.shared.updateSession(session)
                selectedLookId = lookItem.id
                isAnalyzing = false
            }
        }
    }
}
