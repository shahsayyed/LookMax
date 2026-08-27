import SwiftUI
import Vision

struct ProfileOnboardingView: View {
    @Binding var profile: UserProfile?
    @Environment(\.presentationMode) var presentationMode

    @State private var name = ""
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
                            .font(.system(size: 48)).foregroundColor(.blue)
                        Text(profile == nil ? "Create Your Profile" : "Edit Profile")
                            .font(.title2.bold())
                        Text("Add your name and 1–3 reference photos of your face for personalized identity tracking and style history.")
                            .font(.subheadline).foregroundColor(.secondary)
                            .multilineTextAlignment(.center).padding(.horizontal)
                    }
                    .padding(.top, 10)

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Your Name").font(.subheadline.bold())
                        TextField("Enter full name", text: $name)
                            .padding()
                            .background(Color(UIColor.secondarySystemBackground))
                            .cornerRadius(12)
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Reference Photos (\(selectedImages.count)/3)").font(.subheadline.bold())
                            Spacer()
                            if selectedImages.count < 3 {
                                Button(action: { showingImagePicker = true }) {
                                    Label("Add Photo", systemImage: "plus.circle.fill").font(.subheadline.bold())
                                }
                            }
                        }

                        HStack(spacing: 12) {
                            ForEach(Array(selectedImages.enumerated()), id: \.offset) { index, img in
                                ZStack(alignment: .topTrailing) {
                                    Image(uiImage: img).resizable().scaledToFill()
                                        .frame(width: 96, height: 96)
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                    Button(action: { selectedImages.remove(at: index) }) {
                                        Image(systemName: "xmark.circle.fill")
                                            .foregroundColor(.white)
                                            .background(Circle().fill(Color.black.opacity(0.6)))
                                    }.padding(4)
                                }
                            }
                            if selectedImages.count < 3 {
                                Button(action: { showingImagePicker = true }) {
                                    VStack(spacing: 6) {
                                        Image(systemName: "camera.fill").font(.title3)
                                        Text("Add #\(selectedImages.count + 1)").font(.caption2.bold())
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
                        Text(err).font(.caption).foregroundColor(.red).multilineTextAlignment(.center)
                    }

                    Button(action: enrollProfile) {
                        if isProcessing {
                            ProgressView().progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .frame(maxWidth: .infinity).padding()
                                .background(Color.blue).clipShape(RoundedRectangle(cornerRadius: 14))
                        } else {
                            Text("Save Profile").font(.headline)
                                .frame(maxWidth: .infinity).padding()
                                .background(canSave ? Color.blue : Color.gray.opacity(0.4))
                                .foregroundColor(.white).clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                    }
                    .disabled(!canSave || isProcessing)

                    if profile != nil {
                        Button("Delete Current Profile", role: .destructive) {
                            UserProfile.clear()
                            profile = nil
                            presentationMode.wrappedValue.dismiss()
                        }
                        .font(.subheadline).padding(.top, 4)
                    }
                }
                .padding()
            }
            .navigationBarItems(trailing: Button("Cancel") { presentationMode.wrappedValue.dismiss() })
            .sheet(isPresented: $showingImagePicker) {
                ImagePicker(image: $pickerImage, sourceType: .photoLibrary)
                    .onDisappear {
                        if let img = pickerImage { selectedImages.append(img); pickerImage = nil }
                    }
            }
            .onAppear {
                if let p = profile {
                    name = p.name
                    selectedImages = p.photoDataList.compactMap { UIImage(data: $0) }
                }
            }
        }
    }

    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespaces).isEmpty && !selectedImages.isEmpty
    }

    private func enrollProfile() {
        isProcessing = true; errorMessage = nil
        DispatchQueue.global(qos: .userInitiated).async {
            var sigs: [FaceBiometricSignature] = []
            var photos: [Data] = []
            let req = VNDetectFaceLandmarksRequest()
            for img in selectedImages {
                guard let cg = img.cgImage else { continue }
                let h = VNImageRequestHandler(
                    cgImage: cg,
                    orientation: CGImagePropertyOrientation(img.imageOrientation),
                    options: [:]
                )
                try? h.perform([req])
                if let face = req.results?.first,
                   let lm = face.landmarks,
                   let sig = extractSig(from: lm, bbox: face.boundingBox) {
                    sigs.append(sig)
                    if let d = img.jpegData(compressionQuality: 0.7) { photos.append(d) }
                }
            }
            DispatchQueue.main.async {
                isProcessing = false
                if sigs.isEmpty {
                    errorMessage = "No clear face detected. Please use clearer front-facing photos."
                } else {
                    let p = UserProfile(
                        name: name.trimmingCharacters(in: .whitespaces),
                        photoDataList: photos,
                        signatures: sigs,
                        dateCreated: Date()
                    )
                    p.save(); profile = p; presentationMode.wrappedValue.dismiss()
                }
            }
        }
    }

    private func extractSig(from lm: VNFaceLandmarks2D, bbox: CGRect) -> FaceBiometricSignature? {
        guard let le = lm.leftEye, let re = lm.rightEye,
              let n = lm.nose ?? lm.noseCrest,
              let lips = lm.outerLips else { return nil }

        let lec = center(le.normalizedPoints)
        let rec = center(re.normalizedPoints)
        let nt  = center(n.normalizedPoints)
        let mc  = center(lips.normalizedPoints)
        let ed  = dist(lec, rec)
        guard ed > 0.01 else { return nil }

        let mid = CGPoint(x: (lec.x + rec.x) / 2, y: (lec.y + rec.y) / 2)
        let noseToChin: Double = (lm.faceContour?.normalizedPoints.last).map { dist(nt, $0) / ed } ?? 1.2
        let mw = (lips.normalizedPoints.map(\.x).max() ?? 0) - (lips.normalizedPoints.map(\.x).min() ?? 0)
        let jw: Double = (lm.faceContour?.normalizedPoints).flatMap { pts -> Double? in
            guard let f = pts.first, let l = pts.last else { return nil }
            return dist(f, l) / ed
        } ?? 2.0

        return FaceBiometricSignature(
            eyeToNoseRatio: dist(mid, nt) / ed,
            eyeToMouthRatio: dist(mid, mc) / ed,
            noseToChinRatio: noseToChin,
            mouthWidthRatio: Double(mw) / ed,
            jawWidthRatio: jw,
            faceAspectRatio: Double(bbox.height / max(0.01, bbox.width))
        )
    }

    private func center(_ pts: [CGPoint]) -> CGPoint {
        guard !pts.isEmpty else { return .zero }
        return CGPoint(
            x: pts.map(\.x).reduce(0, +) / CGFloat(pts.count),
            y: pts.map(\.y).reduce(0, +) / CGFloat(pts.count)
        )
    }

    private func dist(_ a: CGPoint, _ b: CGPoint) -> Double {
        sqrt(pow(Double(a.x - b.x), 2) + pow(Double(a.y - b.y), 2))
    }
}
