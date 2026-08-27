import SwiftUI
import Vision
import CoreImage
import AVFoundation
import PhotosUI
import Combine

// MARK: - Occasion Category
enum OccasionCategory: String, Codable, CaseIterable {
    case businessMeeting = "Business Meeting"
    case dateNight = "Date Night"
    case casualEveryday = "Casual Everyday"
    case formalEvent = "Formal Event"
    case custom = "Custom"

    var icon: String {
        switch self {
        case .businessMeeting: return "briefcase.fill"
        case .dateNight: return "heart.fill"
        case .casualEveryday: return "sun.max.fill"
        case .formalEvent: return "star.fill"
        case .custom: return "tag.fill"
        }
    }

    var color: Color {
        switch self {
        case .businessMeeting: return .blue
        case .dateNight: return .pink
        case .casualEveryday: return .orange
        case .formalEvent: return .purple
        case .custom: return .gray
        }
    }
}

// MARK: - Style Suggestion Model
struct StyleSuggestion: Identifiable, Codable {
    let id: UUID
    let category: String
    let icon: String
    let iconColorHex: String
    let title: String
    let recommendation: String
    let effortTime: String
    var isDone: Bool

    init(id: UUID = UUID(), category: String, icon: String, iconColor: Color, title: String, recommendation: String, effortTime: String, isDone: Bool = false) {
        self.id = id
        self.category = category
        self.icon = icon
        self.iconColorHex = iconColor.toHex()
        self.title = title
        self.recommendation = recommendation
        self.effortTime = effortTime
        self.isDone = isDone
    }

    var iconColor: Color { Color(hex: iconColorHex) ?? .blue }
}

// MARK: - Look Item (Single Photo Analysis)
struct LookItem: Identifiable, Codable {
    let id: UUID
    let imagePath: String           // stored in documents dir
    let timestamp: Date
    let score: Double
    let headlineBadge: String
    let goodPoints: [String]
    let badPoints: [String]
    var suggestions: [StyleSuggestion]
    let detectedOutfitColor: String
    let detectedFaceShape: String
    let lightingScore: Int

    init(id: UUID = UUID(), imagePath: String, timestamp: Date = Date(), score: Double, headlineBadge: String, goodPoints: [String], badPoints: [String], suggestions: [StyleSuggestion], detectedOutfitColor: String, detectedFaceShape: String, lightingScore: Int) {
        self.id = id
        self.imagePath = imagePath
        self.timestamp = timestamp
        self.score = score
        self.headlineBadge = headlineBadge
        self.goodPoints = goodPoints
        self.badPoints = badPoints
        self.suggestions = suggestions
        self.detectedOutfitColor = detectedOutfitColor
        self.detectedFaceShape = detectedFaceShape
        self.lightingScore = lightingScore
    }

    var image: UIImage? { UIImage(contentsOfFile: imagePath) }
    var formattedTime: String {
        let f = DateFormatter()
        f.timeStyle = .short
        return f.string(from: timestamp)
    }
    var formattedDate: String {
        let f = DateFormatter()
        f.dateStyle = .medium
        return f.string(from: timestamp)
    }
}

// MARK: - Look Session Model
struct LookSession: Identifiable, Codable {
    let id: UUID
    var title: String
    var occasion: OccasionCategory
    let createdAt: Date
    var looks: [LookItem]

    init(id: UUID = UUID(), title: String, occasion: OccasionCategory, createdAt: Date = Date(), looks: [LookItem] = []) {
        self.id = id
        self.title = title
        self.occasion = occasion
        self.createdAt = createdAt
        self.looks = looks
    }

    var bestLook: LookItem? { looks.max(by: { $0.score < $1.score }) }
    var averageScore: Double {
        guard !looks.isEmpty else { return 0 }
        return looks.reduce(0) { $0 + $1.score } / Double(looks.count)
    }
    var formattedDate: String {
        let f = DateFormatter()
        f.dateStyle = .medium
        return f.string(from: createdAt)
    }
}

// MARK: - Session Storage Manager
class SessionStorageManager: ObservableObject {
    static let shared = SessionStorageManager()
    static let sessionsKey = "FaceReport_Sessions_v2"

    @Published var sessions: [LookSession] = []

    private var documentsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    init() { load() }

    func load() {
        guard let data = UserDefaults.standard.data(forKey: Self.sessionsKey),
              let decoded = try? JSONDecoder().decode([LookSession].self, from: data) else { return }
        sessions = decoded
    }

    func save() {
        if let data = try? JSONEncoder().encode(sessions) {
            UserDefaults.standard.set(data, forKey: Self.sessionsKey)
        }
    }

    func saveImage(_ image: UIImage) -> String {
        let filename = UUID().uuidString + ".jpg"
        let url = documentsDir.appendingPathComponent(filename)
        if let data = image.jpegData(compressionQuality: 0.75) {
            try? data.write(to: url)
        }
        return url.path
    }

    func addSession(_ session: LookSession) {
        sessions.insert(session, at: 0)
        save()
    }

    func updateSession(_ session: LookSession) {
        if let idx = sessions.firstIndex(where: { $0.id == session.id }) {
            sessions[idx] = session
            save()
        }
    }

    func deleteSession(_ session: LookSession) {
        // Clean up stored images
        for look in session.looks {
            try? FileManager.default.removeItem(atPath: look.imagePath)
        }
        sessions.removeAll { $0.id == session.id }
        save()
    }

    func deleteLook(_ look: LookItem, from session: inout LookSession) {
        try? FileManager.default.removeItem(atPath: look.imagePath)
        session.looks.removeAll { $0.id == look.id }
        updateSession(session)
    }
}

// MARK: - Biometric Face Signature Model
struct FaceBiometricSignature: Codable, Equatable {
    let eyeToNoseRatio: Double
    let eyeToMouthRatio: Double
    let noseToChinRatio: Double
    let mouthWidthRatio: Double
    let jawWidthRatio: Double
    let faceAspectRatio: Double

    func similarity(to other: FaceBiometricSignature) -> Double {
        let diffs = [
            abs(eyeToNoseRatio - other.eyeToNoseRatio) * 1.4,
            abs(eyeToMouthRatio - other.eyeToMouthRatio) * 1.4,
            abs(noseToChinRatio - other.noseToChinRatio) * 1.1,
            abs(mouthWidthRatio - other.mouthWidthRatio) * 1.0,
            abs(jawWidthRatio - other.jawWidthRatio) * 1.0,
            abs(faceAspectRatio - other.faceAspectRatio) * 1.1
        ]
        let avgDiff = diffs.reduce(0, +) / Double(diffs.count)
        return max(0.0, min(1.0, 1.0 - (avgDiff * 2.5)))
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

    static func clear() { UserDefaults.standard.removeObject(forKey: storageKey) }
}

// MARK: - AI Consultant Result (Transient)
struct LookAnalysisResult {
    let score: Double
    let headlineBadge: String
    let goodPoints: [String]
    let badPoints: [String]
    let suggestions: [StyleSuggestion]
    let detectedOutfitColor: String
    let detectedFaceShape: String
    let lightingScore: Int
}

// MARK: - Color Hex Utilities
extension Color {
    func toHex() -> String {
        let uiColor = UIColor(self)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        uiColor.getRed(&r, green: &g, blue: &b, alpha: &a)
        return String(format: "#%02X%02X%02X", Int(r * 255), Int(g * 255), Int(b * 255))
    }

    init?(hex: String) {
        var str = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        if str.hasPrefix("#") { str.removeFirst() }
        guard str.count == 6, let value = UInt64(str, radix: 16) else { return nil }
        self.init(red: Double((value >> 16) & 0xFF) / 255,
                  green: Double((value >> 8) & 0xFF) / 255,
                  blue: Double(value & 0xFF) / 255)
    }
}

// MARK: - CGImage Orientation Helper
extension CGImagePropertyOrientation {
    init(_ o: UIImage.Orientation) {
        switch o {
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

// =============================================================================
// MARK: - MAIN CONTENT VIEW (Sessions Dashboard)
// =============================================================================
struct ContentView: View {
    @State private var profile: UserProfile? = UserProfile.load()
    @StateObject private var storage = SessionStorageManager.shared
    @State private var showingOnboarding = false
    @State private var showingCreateSession = false
    @State private var selectedSession: LookSession?

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Profile Header
                ProfileHeaderBanner(profile: $profile, showingOnboarding: $showingOnboarding)
                    .padding(.horizontal)
                    .padding(.top, 12)
                    .padding(.bottom, 8)

                if storage.sessions.isEmpty {
                    EmptySessionsView(onNew: { showingCreateSession = true })
                } else {
                    ScrollView {
                        LazyVStack(spacing: 14) {
                            Button(action: { showingCreateSession = true }) {
                                Label("Start New Style Session", systemImage: "plus.circle.fill")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 14))
                            }
                            .padding(.horizontal)
                            .padding(.top, 8)

                            ForEach(storage.sessions) { session in
                                SessionCardView(session: session)
                                    .onTapGesture { selectedSession = session }
                                    .padding(.horizontal)
                            }
                            .onDelete { indexSet in
                                indexSet.forEach { storage.deleteSession(storage.sessions[$0]) }
                            }
                        }
                        .padding(.bottom, 30)
                    }
                }
            }
            .navigationTitle("Style Sessions")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingOnboarding = true }) {
                        Image(systemName: profile == nil ? "person.crop.circle.badge.plus" : "person.crop.circle.fill")
                    }
                }
            }
            .sheet(isPresented: $showingOnboarding) {
                ProfileOnboardingView(profile: $profile)
            }
            .sheet(isPresented: $showingCreateSession) {
                CreateSessionSheet { newSession in
                    storage.addSession(newSession)
                    selectedSession = newSession
                }
            }
            .sheet(item: $selectedSession) { session in
                SessionDetailView(session: binding(for: session), profile: profile)
            }
        }
        .onAppear {
            if profile == nil { showingOnboarding = true }
        }
    }

    private func binding(for session: LookSession) -> Binding<LookSession> {
        guard let idx = storage.sessions.firstIndex(where: { $0.id == session.id }) else {
            return .constant(session)
        }
        return $storage.sessions[idx]
    }
}

// MARK: - Profile Header Banner
struct ProfileHeaderBanner: View {
    @Binding var profile: UserProfile?
    @Binding var showingOnboarding: Bool

    var body: some View {
        HStack(spacing: 12) {
            if let p = profile {
                if let data = p.photoDataList.first, let img = UIImage(data: data) {
                    Image(uiImage: img)
                        .resizable().scaledToFill()
                        .frame(width: 40, height: 40)
                        .clipShape(Circle())
                        .overlay(Circle().stroke(Color.blue, lineWidth: 2))
                } else {
                    Image(systemName: "person.circle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.blue)
                }

                VStack(alignment: .leading, spacing: 1) {
                    Text(p.name).font(.subheadline.bold())
                    Text("Personal Style Client").font(.caption2).foregroundColor(.secondary)
                }
            } else {
                Image(systemName: "person.crop.circle.badge.plus")
                    .font(.system(size: 36))
                    .foregroundColor(.secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Setup Profile").font(.subheadline.bold())
                    Text("For identity tracking & style history").font(.caption2).foregroundColor(.secondary)
                }
            }

            Spacer()

            Button(profile == nil ? "Enroll" : "Edit") {
                showingOnboarding = true
            }
            .font(.caption.bold())
            .padding(.horizontal, 12).padding(.vertical, 5)
            .background(Color.blue.opacity(0.12))
            .foregroundColor(.blue)
            .clipShape(Capsule())
        }
        .padding(12)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(14)
    }
}

// MARK: - Empty Sessions State
struct EmptySessionsView: View {
    let onNew: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "sparkles.rectangle.stack.fill")
                .font(.system(size: 64))
                .foregroundColor(.purple.opacity(0.8))
            Text("No Style Sessions Yet")
                .font(.title2.bold())
            Text("Create your first session to start building your personal style history. Try different outfits and let the AI tell you what works.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 36)

            Button(action: onNew) {
                Label("Start Your First Session", systemImage: "plus.circle.fill")
                    .font(.headline)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .padding(.horizontal, 30)
            Spacer()
        }
    }
}

// MARK: - Session Card View
struct SessionCardView: View {
    let session: LookSession

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(session.occasion.rawValue, systemImage: session.occasion.icon)
                    .font(.caption.bold())
                    .foregroundColor(session.occasion.color)
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(session.occasion.color.opacity(0.12))
                    .clipShape(Capsule())

                Spacer()

                Text(session.formattedDate)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Text(session.title)
                .font(.headline)

            if !session.looks.isEmpty {
                HStack(spacing: 6) {
                    // Thumbnail grid (up to 4)
                    ForEach(session.looks.prefix(4)) { look in
                        if let img = look.image {
                            Image(uiImage: img)
                                .resizable().scaledToFill()
                                .frame(width: 56, height: 56)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    if session.looks.count > 4 {
                        ZStack {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.secondary.opacity(0.15))
                                .frame(width: 56, height: 56)
                            Text("+\(session.looks.count - 4)")
                                .font(.subheadline.bold())
                                .foregroundColor(.secondary)
                        }
                    }
                    Spacer()
                    // Best score badge
                    if let best = session.bestLook {
                        VStack(alignment: .trailing, spacing: 2) {
                            HStack(alignment: .firstTextBaseline, spacing: 2) {
                                Text(String(format: "%.1f", best.score))
                                    .font(.title3.bold())
                                    .foregroundColor(.purple)
                                Text("/10")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Text("Best Look").font(.caption2).foregroundColor(.secondary)
                        }
                    }
                }
            } else {
                Text("No looks added yet — tap to open and start scanning!")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .italic()
            }
        }
        .padding(14)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
        .contentShape(Rectangle())
    }
}

// MARK: - Create Session Sheet
struct CreateSessionSheet: View {
    let onCreated: (LookSession) -> Void
    @Environment(\.presentationMode) var presentationMode

    @State private var title = ""
    @State private var occasion: OccasionCategory = .casualEveryday

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Session Name")) {
                    TextField("e.g. Board Meeting, Date Night, Job Interview", text: $title)
                }

                Section(header: Text("Occasion")) {
                    ForEach(OccasionCategory.allCases, id: \.self) { cat in
                        HStack {
                            Label(cat.rawValue, systemImage: cat.icon)
                                .foregroundColor(cat.color)
                            Spacer()
                            if occasion == cat {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.blue)
                            }
                        }
                        .contentShape(Rectangle())
                        .onTapGesture { occasion = cat }
                    }
                }
            }
            .navigationTitle("New Session")
            .navigationBarItems(
                leading: Button("Cancel") { presentationMode.wrappedValue.dismiss() },
                trailing: Button("Create") {
                    let trimmed = title.trimmingCharacters(in: .whitespaces)
                    let name = trimmed.isEmpty ? occasion.rawValue + " Session" : trimmed
                    let session = LookSession(title: name, occasion: occasion)
                    onCreated(session)
                    presentationMode.wrappedValue.dismiss()
                }
                .bold()
            )
        }
    }
}

// =============================================================================
// MARK: - SESSION DETAIL VIEW
// =============================================================================
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
                                .font(.system(size: 54))
                                .foregroundColor(.purple.opacity(0.6))
                            Text("Add Your First Look")
                                .font(.title3.bold())
                            Text("Take or choose a photo to get instant AI styling feedback for \"\(session.title)\".")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 28)
                        }
                        .padding(.vertical, 40)
                    } else {
                        // Photo Carousel
                        LookCarouselView(looks: session.looks, selectedId: $selectedLookId)

                        // Comparison Bar
                        if session.looks.count > 1 {
                            ScoreComparisonBar(looks: session.looks, selectedId: selectedLookId)
                                .padding(.horizontal)
                        }

                        // Detailed AI Feedback for Selected Look
                        if let look = selectedLook {
                            LookDetailCard(look: look, isBestLook: look.id == session.bestLook?.id)
                                .padding(.horizontal)
                        }
                    }

                    // Action Buttons
                    VStack(spacing: 10) {
                        Button(action: { checkCameraPermission() }) {
                            Label(session.looks.isEmpty ? "Add Look with Camera" : "Add Another Look (Camera)", systemImage: "camera.fill")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                        Button(action: { showingLibraryPicker = true }) {
                            Label("Choose from Photo Library", systemImage: "photo.on.rectangle")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color(UIColor.secondarySystemBackground))
                                .foregroundColor(.primary)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                    }
                    .padding(.horizontal)

                    if isAnalyzing {
                        HStack(spacing: 12) {
                            ProgressView()
                            Text("AI Consultant Analyzing your look…")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                    }
                }
                .padding(.top, 12)
                .padding(.bottom, 40)
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
                    .onDisappear {
                        if incomingImage != nil { analyzeAndAddLook() }
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

            let faces = faceRequest.results ?? []
            let bodyPoses = bodyPoseRequest.results ?? []
            let classifications = classificationRequest.results ?? []

            let analysis = LookAnalysisEngine.analyze(
                faces: faces,
                faceQualityRequest: faceQualityRequest,
                bodyPoses: bodyPoses,
                classifications: classifications,
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

// MARK: - Look Carousel
struct LookCarouselView: View {
    let looks: [LookItem]
    @Binding var selectedId: UUID?

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(looks) { look in
                    let isSelected = (selectedId ?? looks.first?.id) == look.id
                    VStack(spacing: 4) {
                        ZStack(alignment: .topTrailing) {
                            if let img = look.image {
                                Image(uiImage: img)
                                    .resizable().scaledToFill()
                                    .frame(width: 100, height: 130)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(isSelected ? Color.purple : Color.clear, lineWidth: 2.5)
                                    )
                            }
                            Text(String(format: "%.1f", look.score))
                                .font(.caption2.bold())
                                .foregroundColor(.white)
                                .padding(.horizontal, 5).padding(.vertical, 2)
                                .background(scoreColor(look.score))
                                .clipShape(Capsule())
                                .padding(5)
                        }
                        Text(look.formattedTime)
                            .font(.caption2)
                            .foregroundColor(isSelected ? .purple : .secondary)
                    }
                    .onTapGesture { selectedId = look.id }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
        }
    }

    private func scoreColor(_ score: Double) -> Color {
        if score >= 8.5 { return .green }
        if score >= 7.5 { return .blue }
        return .orange
    }
}

// MARK: - Score Comparison Bar
struct ScoreComparisonBar: View {
    let looks: [LookItem]
    let selectedId: UUID?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("LOOK COMPARISON").font(.caption2.bold()).foregroundColor(.secondary).tracking(0.5)

            HStack(alignment: .bottom, spacing: 6) {
                ForEach(Array(looks.enumerated()), id: \.element.id) { idx, look in
                    let isSelected = (selectedId ?? looks.first?.id) == look.id
                    VStack(spacing: 4) {
                        Text(String(format: "%.1f", look.score))
                            .font(.caption2.bold())
                            .foregroundColor(isSelected ? .white : .primary)
                        GeometryReader { geo in
                            VStack {
                                Spacer()
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(isSelected ? Color.purple : Color.blue.opacity(0.4))
                                    .frame(height: barHeight(look.score, in: geo.size.height))
                            }
                        }
                        .frame(height: 50)
                        Text("Look \(idx + 1)").font(.caption2).foregroundColor(.secondary)
                    }
                }
            }
            .frame(height: 80)
        }
        .padding(12)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
    }

    private func barHeight(_ score: Double, in total: CGFloat) -> CGFloat {
        let norm = (score - 6.0) / 4.0
        return max(8, CGFloat(norm) * total)
    }
}

// MARK: - Look Detail Card (Good / Bad / Tweaks)
struct LookDetailCard: View {
    let look: LookItem
    let isBestLook: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {

            // Score Header
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 4) {
                    if isBestLook {
                        Label("Top Pick in Session", systemImage: "trophy.fill")
                            .font(.caption2.bold())
                            .foregroundColor(.yellow)
                    }
                    Text(look.headlineBadge).font(.title3.bold())
                    Text("\(look.detectedFaceShape) Face • \(look.detectedOutfitColor)")
                        .font(.caption).foregroundColor(.secondary)
                }
                Spacer()
                HStack(alignment: .firstTextBaseline, spacing: 2) {
                    Text(String(format: "%.1f", look.score))
                        .font(.system(size: 40, weight: .heavy, design: .rounded))
                        .foregroundColor(.purple)
                    Text("/10").font(.subheadline.bold()).foregroundColor(.secondary)
                }
            }

            Divider()

            // What Looked Good
            VStack(alignment: .leading, spacing: 6) {
                Label("WHAT LOOKED GOOD", systemImage: "hand.thumbsup.fill")
                    .font(.caption.bold())
                    .foregroundColor(.green)

                ForEach(look.goodPoints, id: \.self) { point in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "checkmark.circle.fill").foregroundColor(.green).font(.caption)
                            .padding(.top, 1)
                        Text(point).font(.subheadline)
                    }
                }
            }

            Divider()

            // What Needs Work
            VStack(alignment: .leading, spacing: 6) {
                Label("WHAT NEEDS IMPROVEMENT", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption.bold())
                    .foregroundColor(.orange)

                ForEach(look.badPoints, id: \.self) { point in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "arrow.up.circle.fill").foregroundColor(.orange).font(.caption)
                            .padding(.top, 1)
                        Text(point).font(.subheadline)
                    }
                }
            }

            if !look.suggestions.isEmpty {
                Divider()

                // 5-Min Tweaks Checklist
                VStack(alignment: .leading, spacing: 10) {
                    Label("QUICK 5-MIN TWEAKS", systemImage: "clock.arrow.circlepath")
                        .font(.caption.bold())
                        .foregroundColor(.blue)

                    ForEach(look.suggestions) { sug in
                        SuggestionRow(suggestion: sug)
                    }
                }
            }
        }
        .padding(16)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
    }
}

// MARK: - Suggestion Row
struct SuggestionRow: View {
    let suggestion: StyleSuggestion
    @State private var isDone: Bool

    init(suggestion: StyleSuggestion) {
        self.suggestion = suggestion
        _isDone = State(initialValue: suggestion.isDone)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Button(action: {
                isDone.toggle()
                UIImpactFeedbackGenerator(style: .light).impactOccurred()
            }) {
                Image(systemName: isDone ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isDone ? .green : .secondary)
                    .font(.title3)
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Label(suggestion.category, systemImage: suggestion.icon)
                        .font(.caption2.bold())
                        .foregroundColor(suggestion.iconColor)
                    Spacer()
                    Text(suggestion.effortTime)
                        .font(.caption2)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.blue.opacity(0.10))
                        .foregroundColor(.blue)
                        .clipShape(Capsule())
                }
                Text(suggestion.title)
                    .font(.subheadline.bold())
                    .strikethrough(isDone, color: .secondary)
                    .foregroundColor(isDone ? .secondary : .primary)
                Text(suggestion.recommendation)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .background(isDone ? Color.green.opacity(0.06) : Color(UIColor.tertiarySystemBackground))
        .cornerRadius(10)
    }
}

// =============================================================================
// MARK: - LOOK ANALYSIS ENGINE
// =============================================================================
enum LookAnalysisEngine {
    static func analyze(
        faces: [VNFaceObservation],
        faceQualityRequest: VNDetectFaceCaptureQualityRequest,
        bodyPoses: [VNHumanBodyPoseObservation],
        classifications: [VNClassificationObservation],
        cgImage: CGImage,
        occasion: OccasionCategory
    ) -> LookAnalysisResult {

        var goodPoints: [String] = []
        var badPoints: [String] = []
        var suggestions: [StyleSuggestion] = []
        var baseScore = 7.6

        // ─── Face Shape ───
        var faceShape = "Balanced Oval"
        if let face = faces.first {
            let ratio = face.boundingBox.height / max(0.01, face.boundingBox.width)
            if ratio > 1.35 { faceShape = "Elongated / Oblong" }
            else if ratio < 1.15 { faceShape = "Round / Square" }
        }

        // ─── Lighting & Capture Quality ───
        var lightingScore = 70
        var qualityScore = 0.7
        if let q = faceQualityRequest.results?.first?.faceCaptureQuality {
            qualityScore = Double(q)
            lightingScore = Int(qualityScore * 100)
        }

        if qualityScore >= 0.65 {
            goodPoints.append("Well-lit portrait with minimal shadow and crisp sharpness (\(lightingScore)% clarity).")
            baseScore += 0.4
        } else if qualityScore >= 0.45 {
            badPoints.append("Lighting slightly flat or soft – ideal clarity score is 65%+, yours is \(lightingScore)%.")
            suggestions.append(StyleSuggestion(
                category: "Lighting",
                icon: "sun.max.fill",
                iconColor: .yellow,
                title: "Face a Natural Light Source",
                recommendation: "Turn toward a window or soft lamp at 45° to your face. Avoid harsh overhead or direct sunlight.",
                effortTime: "10 sec"
            ))
        } else {
            badPoints.append("Poor lighting conditions — photo appears dark or backlit (\(lightingScore)% clarity score). Reshoot near natural light.")
            baseScore -= 0.5
            suggestions.append(StyleSuggestion(
                category: "Lighting",
                icon: "sun.max.fill",
                iconColor: .yellow,
                title: "Move to Better Light Immediately",
                recommendation: "Face a window or turn on a lamp behind your phone. Avoid backlighting (bright window behind you).",
                effortTime: "15 sec"
            ))
        }

        // ─── Face Detected ───
        if faces.isEmpty {
            badPoints.append("No clear face detected – ensure your face is visible and well-framed.")
        } else {
            goodPoints.append("Clear face detection with well-centered framing.")
        }

        // ─── Posture & Body Pose ───
        var postureGood = false
        if let body = bodyPoses.first {
            if let neck = try? body.recognizedPoint(.neck), let root = try? body.recognizedPoint(.root),
               neck.confidence > 0.3, root.confidence > 0.3 {
                let dx = abs(Double(neck.location.x - root.location.x))
                if dx < 0.04 {
                    goodPoints.append("Upright, aligned posture projects confidence and authority.")
                    baseScore += 0.5
                    postureGood = true
                } else {
                    badPoints.append("Slight postural lean detected – shoulders appear uneven.")
                }
            }
        }
        if !postureGood {
            suggestions.append(StyleSuggestion(
                category: "Posture & Angle",
                icon: "figure.walk",
                iconColor: .teal,
                title: "Turn Shoulders 15° for Depth",
                recommendation: "Rotate your body slightly while keeping your face forward. This creates a slimmer, more dynamic silhouette.",
                effortTime: "5 sec"
            ))
        }

        // Chin angle advice (always added as a quick micro-tweak)
        if faces.first != nil {
            suggestions.append(StyleSuggestion(
                category: "Chin & Jawline",
                icon: "arrow.up.and.down.and.sparkles",
                iconColor: .green,
                title: "Extend Chin Forward & Slightly Down",
                recommendation: "Push ears slightly forward (\"turtle move\") and lower chin ~5° to sharpen the jawline and remove any under-chin softness.",
                effortTime: "5 sec"
            ))
        }

        // ─── Hairstyle & Face Shape Advice ───
        let hairKeywords = ["beard", "mustache", "afro", "curls"]
        let hasBeard = classifications.contains { item in
            hairKeywords.contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.25
        }

        if faceShape == "Round / Square" {
            badPoints.append("Round/Square face shape benefits from vertical height – avoid flat, wide hair styles.")
            suggestions.append(StyleSuggestion(
                category: "Hairstyle",
                icon: "comb.fill",
                iconColor: .orange,
                title: "Add Height on Top or High Part",
                recommendation: "A higher side part or slight quiff/volume on top creates an elongating effect, visually slimming a round or square face.",
                effortTime: "1 min"
            ))
        } else if faceShape == "Elongated / Oblong" {
            suggestions.append(StyleSuggestion(
                category: "Hairstyle",
                icon: "comb.fill",
                iconColor: .orange,
                title: "Keep Sides Full, Avoid Extra Height",
                recommendation: "Side volume and textured layers balance an elongated face. Avoid pompadours or top-heavy styles that add more height.",
                effortTime: "1 min"
            ))
        } else {
            goodPoints.append("Balanced oval face shape – most hairstyles and frame shapes suit your proportions well.")
        }

        if hasBeard {
            suggestions.append(StyleSuggestion(
                category: "Beard Lineup",
                icon: "scissors",
                iconColor: .brown,
                title: "Define Neckline Two Fingers Above Adam's Apple",
                recommendation: "A sharp, clean neckline at this height instantly sculpts the jawline. Use a trimmer for a clean edge before stepping out.",
                effortTime: "2 mins"
            ))
        } else {
            suggestions.append(StyleSuggestion(
                category: "Grooming",
                icon: "comb.fill",
                iconColor: .orange,
                title: "Tame Flyaways with Matte Paste",
                recommendation: "Use a pea-sized amount of matte paste or water to smooth the hairline and keep your silhouette sharp under camera and bright light.",
                effortTime: "30 sec"
            ))
        }

        // ─── Eyewear ───
        let eyewearMatches = classifications.filter { item in
            ["sunglass", "glasses", "spectacles", "eyewear"].contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.3
        }
        if eyewearMatches.isEmpty == false {
            goodPoints.append("Eyewear detected – adds structure and visual interest to the eye region.")
            suggestions.append(StyleSuggestion(
                category: "Glasses Position",
                icon: "eyeglasses",
                iconColor: .blue,
                title: "Align Bridge: Pupils in Upper Lens Third",
                recommendation: "Slide frames slightly up so your pupils sit in the upper third of each lens. This centers the eye zone and maximizes eye contact.",
                effortTime: "10 sec"
            ))
        } else if faceShape == "Round / Square" {
            suggestions.append(StyleSuggestion(
                category: "Eyewear Tip",
                icon: "eyeglasses",
                iconColor: .blue,
                title: "Choose Angular or Geometric Frames",
                recommendation: "Rectangular or cat-eye frames contrast a round face and add sharp definition. Avoid round or oval frames that echo face shape.",
                effortTime: "Tip"
            ))
        }

        // ─── Outfit / Apparel ───
        let dominantColor = sampleDominantColorName(in: cgImage, normalizedRect: CGRect(x: 0.35, y: 0.45, width: 0.30, height: 0.25)) ?? "Neutral tone"
        let formalKeywords = ["suit", "blazer", "tie", "jacket", "shirt", "collar", "dress"]
        let casualKeywords = ["t-shirt", "hoodie", "sweater", "jersey", "denim"]

        let isFormal = classifications.contains { item in
            formalKeywords.contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.25
        }
        let isCasual = classifications.contains { item in
            casualKeywords.contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.25
        }

        let occasionFormalExpected: Bool
        switch occasion {
        case .businessMeeting, .formalEvent: occasionFormalExpected = true
        default: occasionFormalExpected = false
        }

        if isFormal {
            goodPoints.append("Structured, formal attire with a strong collar and lapel outline.")
            baseScore += 0.5
            if occasionFormalExpected {
                goodPoints.append("Outfit is well-matched for the occasion (\(occasion.rawValue)).")
                baseScore += 0.3
            }
            suggestions.append(StyleSuggestion(
                category: "Collar & Lapel",
                icon: "tshirt.fill",
                iconColor: .purple,
                title: "Straighten Collar Points & Center Lapels",
                recommendation: "Check that shirt collar points lie flat under jacket lapels without curling. Creates a clean vertical chest line for a sharp look.",
                effortTime: "30 sec"
            ))
        } else if isCasual {
            if occasionFormalExpected {
                badPoints.append("Casual outfit may be underdressed for \(occasion.rawValue). Consider adding a blazer or structured layer.")
                baseScore -= 0.4
                suggestions.append(StyleSuggestion(
                    category: "Outfit Upgrade",
                    icon: "tshirt.fill",
                    iconColor: .red,
                    title: "Add a Blazer or Structured Jacket",
                    recommendation: "A simple navy or grey blazer over almost any outfit instantly elevates the look by 2-3 points for business or formal occasions.",
                    effortTime: "2 mins"
                ))
            } else {
                goodPoints.append("Relaxed, appropriate casual outfit for \(occasion.rawValue).")
                suggestions.append(StyleSuggestion(
                    category: "Casual Layering",
                    icon: "tshirt.fill",
                    iconColor: .purple,
                    title: "Layer for Depth & Dimension",
                    recommendation: "An open overshirt or unzipped jacket over a base layer adds visual structure and broadens the shoulder frame in casual settings.",
                    effortTime: "2 mins"
                ))
            }
        } else {
            suggestions.append(StyleSuggestion(
                category: "Outfit Framing",
                icon: "tshirt.fill",
                iconColor: .purple,
                title: "Check Neckline & Shoulder Fit",
                recommendation: "Ensure your top's neckline sits clean and the shoulders align to your actual shoulder line. Ill-fitting shoulders instantly reduce visual sharpness.",
                effortTime: "1 min"
            ))
        }

        if dominantColor.contains("Dark") || dominantColor.contains("Black") || dominantColor.contains("Navy") {
            goodPoints.append("Dark \(dominantColor.lowercased()) top creates high contrast against skin tones, sharpening facial features.")
        } else if dominantColor.contains("Light") || dominantColor.contains("White") {
            goodPoints.append("Light/white top reflects flattering illumination toward the face, brightening the overall look.")
        }

        // ─── Final Score ───
        let finalScore = max(6.5, min(9.8, baseScore))
        let headline: String
        switch finalScore {
        case 9.3...: headline = "Executive & Flawless"
        case 8.5..<9.3: headline = "Sharp & Polished"
        case 7.8..<8.5: headline = "Well-Put-Together"
        case 7.0..<7.8: headline = "Clean & Casual"
        default: headline = "Needs a Few Tweaks"
        }

        // Ensure at least 1 good and 1 bad point
        if goodPoints.isEmpty { goodPoints.append("Natural relaxed presence and authentic expression.") }
        if badPoints.isEmpty { badPoints.append("No major issues detected – see the 5-min tweaks below for polishing details.") }

        return LookAnalysisResult(
            score: finalScore,
            headlineBadge: headline,
            goodPoints: goodPoints,
            badPoints: badPoints,
            suggestions: suggestions,
            detectedOutfitColor: dominantColor,
            detectedFaceShape: faceShape,
            lightingScore: lightingScore
        )
    }

    // MARK: - Color Sampling
    private static func sampleDominantColorName(in cgImage: CGImage, normalizedRect: CGRect) -> String? {
        let w = CGFloat(cgImage.width)
        let h = CGFloat(cgImage.height)
        let cropRect = CGRect(x: normalizedRect.origin.x * w, y: normalizedRect.origin.y * h,
                              width: normalizedRect.size.width * w, height: normalizedRect.size.height * h)
        guard let cropped = cgImage.cropping(to: cropRect) else { return nil }

        let ctx = CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                           space: CGColorSpaceCreateDeviceRGB(),
                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.draw(cropped, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        guard let data = ctx?.data else { return nil }

        let ptr = data.bindMemory(to: UInt8.self, capacity: 4)
        let r = Double(ptr[0]) / 255
        let g = Double(ptr[1]) / 255
        let b = Double(ptr[2]) / 255
        let brightness = (r + g + b) / 3

        if brightness < 0.20 { return "Dark / Black" }
        if brightness > 0.82 { return "Light / White" }
        let maxDiff = max(abs(r-g), abs(g-b), abs(r-b))
        if maxDiff < 0.08 { return "Neutral Grey" }
        if r > g && r > b { return g > 0.5 ? "Warm Amber" : "Warm Red / Burgundy" }
        if g > r && g > b { return "Olive / Green" }
        if b > r && b > g { return r > 0.4 ? "Purple / Violet" : "Navy / Blue" }
        return "Neutral"
    }
}

// =============================================================================
// MARK: - PROFILE ONBOARDING VIEW
// =============================================================================
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
                                        .frame(width: 96, height: 96).clipShape(RoundedRectangle(cornerRadius: 12))
                                    Button(action: { selectedImages.remove(at: index) }) {
                                        Image(systemName: "xmark.circle.fill")
                                            .foregroundColor(.white).background(Circle().fill(Color.black.opacity(0.6)))
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
                                    .overlay(RoundedRectangle(cornerRadius: 12)
                                        .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [4]))
                                        .foregroundColor(.blue.opacity(0.5)))
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
                            Text("Save Profile").font(.headline).frame(maxWidth: .infinity).padding()
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
                if let p = profile { name = p.name; selectedImages = p.photoDataList.compactMap { UIImage(data: $0) } }
            }
        }
    }

    private var canSave: Bool { !name.trimmingCharacters(in: .whitespaces).isEmpty && !selectedImages.isEmpty }

    private func enrollProfile() {
        isProcessing = true; errorMessage = nil
        DispatchQueue.global(qos: .userInitiated).async {
            var sigs: [FaceBiometricSignature] = []
            var photos: [Data] = []
            let req = VNDetectFaceLandmarksRequest()
            for img in selectedImages {
                guard let cg = img.cgImage else { continue }
                let h = VNImageRequestHandler(cgImage: cg, orientation: CGImagePropertyOrientation(img.imageOrientation), options: [:])
                try? h.perform([req])
                if let face = req.results?.first, let lm = face.landmarks, let sig = extractSig(from: lm, bbox: face.boundingBox) {
                    sigs.append(sig)
                    if let d = img.jpegData(compressionQuality: 0.7) { photos.append(d) }
                }
            }
            DispatchQueue.main.async {
                isProcessing = false
                if sigs.isEmpty {
                    errorMessage = "No clear face detected. Please use clearer front-facing photos."
                } else {
                    let p = UserProfile(name: name.trimmingCharacters(in: .whitespaces), photoDataList: photos, signatures: sigs, dateCreated: Date())
                    p.save(); profile = p; presentationMode.wrappedValue.dismiss()
                }
            }
        }
    }

    private func extractSig(from lm: VNFaceLandmarks2D, bbox: CGRect) -> FaceBiometricSignature? {
        guard let le = lm.leftEye, let re = lm.rightEye, let n = lm.nose ?? lm.noseCrest, let lips = lm.outerLips else { return nil }
        let lec = center(le.normalizedPoints); let rec = center(re.normalizedPoints)
        let nt = center(n.normalizedPoints); let mc = center(lips.normalizedPoints)
        let ed = dist(lec, rec); guard ed > 0.01 else { return nil }
        let mid = CGPoint(x: (lec.x+rec.x)/2, y: (lec.y+rec.y)/2)
        let noseToChin: Double = (lm.faceContour?.normalizedPoints.last).map { dist(nt, $0)/ed } ?? 1.2
        let mw = (lips.normalizedPoints.map(\.x).max() ?? 0) - (lips.normalizedPoints.map(\.x).min() ?? 0)
        let jw: Double = (lm.faceContour?.normalizedPoints).flatMap { pts in
            guard let f = pts.first, let l = pts.last else { return nil }
            return dist(f, l)/ed
        } ?? 2.0
        return FaceBiometricSignature(eyeToNoseRatio: dist(mid,nt)/ed, eyeToMouthRatio: dist(mid,mc)/ed,
                                     noseToChinRatio: noseToChin, mouthWidthRatio: Double(mw)/ed,
                                     jawWidthRatio: jw, faceAspectRatio: Double(bbox.height/max(0.01,bbox.width)))
    }
    private func center(_ pts: [CGPoint]) -> CGPoint {
        guard !pts.isEmpty else { return .zero }
        return CGPoint(x: pts.map(\.x).reduce(0,+)/CGFloat(pts.count), y: pts.map(\.y).reduce(0,+)/CGFloat(pts.count))
    }
    private func dist(_ a: CGPoint, _ b: CGPoint) -> Double { sqrt(pow(Double(a.x-b.x),2)+pow(Double(a.y-b.y),2)) }
}

// =============================================================================
// MARK: - CAMERA CONTROLLER
// =============================================================================
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
        guard let device = getCameraDevice(for: cameraPosition) else { session.commitConfiguration(); return }
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
            if session.canAddInput(inp) { session.addInput(inp); currentDeviceInput = inp; cameraPosition = newPos; currentZoomFactor = 1.0 }
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
            let clamp = max(device.minAvailableVideoZoomFactor, min(factor, min(device.activeFormat.videoMaxZoomFactor, 10.0)))
            device.videoZoomFactor = clamp; currentZoomFactor = clamp
            device.unlockForConfiguration()
        } catch { print("Zoom error: \(error)") }
    }

    func focus(at point: CGPoint, viewSize: CGSize) {
        guard let device = currentDeviceInput?.device else { return }
        let fp = CGPoint(x: point.y / viewSize.height, y: 1.0 - (point.x / viewSize.width))
        do {
            try device.lockForConfiguration()
            if device.isFocusPointOfInterestSupported && device.isFocusModeSupported(.autoFocus) {
                device.focusPointOfInterest = fp; device.focusMode = .autoFocus
            }
            if device.isExposurePointOfInterestSupported && device.isExposureModeSupported(.autoExpose) {
                device.exposurePointOfInterest = fp; device.exposureMode = .autoExpose
            }
            device.unlockForConfiguration()
            focusPoint = point; isFocusing = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { self.isFocusing = false }
        } catch { print("Focus error: \(error)") }
    }

    func triggerCapture() {
        selectedTimerDuration > 0 ? startCountdown(seconds: selectedTimerDuration) : performCapture()
    }

    private func startCountdown(seconds: Int) {
        countdownRemaining = seconds; isCountdownActive = true
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        countdownTimer?.invalidate()
        countdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] t in
            guard let self else { return }
            self.countdownRemaining -= 1
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            if self.countdownRemaining <= 0 { t.invalidate(); self.isCountdownActive = false; self.performCapture() }
        }
    }

    func cancelCountdown() {
        countdownTimer?.invalidate(); countdownTimer = nil; isCountdownActive = false; countdownRemaining = 0
    }

    private func performCapture() {
        guard session.isRunning else { return }
        let settings = AVCapturePhotoSettings()
        if let device = currentDeviceInput?.device, device.hasFlash { settings.flashMode = flashMode }
        photoOutput.capturePhoto(with: settings, delegate: self)
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let data = photo.fileDataRepresentation(), let image = UIImage(data: data) else { return }
        let final: UIImage
        if cameraPosition == .front, let cg = image.cgImage {
            final = UIImage(cgImage: cg, scale: image.scale, orientation: .leftMirrored)
        } else { final = image }
        DispatchQueue.main.async { self.onPhotoCaptured?(final) }
    }
}

// MARK: - Custom Camera View
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
            GeometryReader { geo in
                ZStack {
                    CameraPreviewView(camera: camera)
                        .onTapGesture { loc in camera.focus(at: loc, viewSize: geo.size) }
                        .gesture(MagnificationGesture()
                            .onChanged { camera.setZoom(baseZoomFactor * $0) }
                            .onEnded { _ in baseZoomFactor = camera.currentZoomFactor })
                    if camera.isGridVisible { CameraGridView() }
                    if camera.isFocusing, let pt = camera.focusPoint {
                        Rectangle().stroke(Color.yellow, lineWidth: 1.5)
                            .frame(width: 70, height: 70).position(pt)
                            .animation(.easeInOut(duration: 0.2), value: camera.isFocusing)
                    }
                }
            }.ignoresSafeArea()

            if camera.isCountdownActive {
                ZStack {
                    Color.black.opacity(0.4).ignoresSafeArea()
                    VStack(spacing: 24) {
                        Text("\(camera.countdownRemaining)")
                            .font(.system(size: 110, weight: .bold, design: .rounded))
                            .foregroundColor(.white).shadow(radius: 12)
                            .scaleEffect(camera.countdownRemaining > 0 ? 1.1 : 0.8)
                            .animation(.easeInOut(duration: 0.3), value: camera.countdownRemaining)
                        Button(action: { camera.cancelCountdown() }) {
                            Text("Cancel Timer").font(.subheadline.bold()).foregroundColor(.white)
                                .padding(.horizontal, 20).padding(.vertical, 10)
                                .background(Capsule().fill(Color.red.opacity(0.85)))
                        }
                    }
                }
            }

            VStack {
                HStack(spacing: 20) {
                    Button(action: { camera.toggleFlash() }) {
                        Image(systemName: flashIcon).font(.system(size: 20, weight: .semibold))
                            .foregroundColor(camera.flashMode == .off ? .white : .yellow)
                            .frame(width: 44, height: 44).background(.ultraThinMaterial).clipShape(Circle())
                    }
                    Menu {
                        Button("Timer Off") { camera.selectedTimerDuration = 0 }
                        Button("3 Seconds") { camera.selectedTimerDuration = 3 }
                        Button("5 Seconds") { camera.selectedTimerDuration = 5 }
                        Button("10 Seconds") { camera.selectedTimerDuration = 10 }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "timer")
                            if camera.selectedTimerDuration > 0 { Text("\(camera.selectedTimerDuration)s").font(.caption.bold()) }
                        }
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(camera.selectedTimerDuration > 0 ? .yellow : .white)
                        .padding(.horizontal, camera.selectedTimerDuration > 0 ? 12 : 10)
                        .frame(height: 44).background(.ultraThinMaterial).clipShape(Capsule())
                    }
                    Button(action: { camera.isGridVisible.toggle() }) {
                        Image(systemName: camera.isGridVisible ? "grid.circle.fill" : "grid")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(camera.isGridVisible ? .yellow : .white)
                            .frame(width: 44, height: 44).background(.ultraThinMaterial).clipShape(Circle())
                    }
                    Spacer()
                    Button(action: { camera.stopCamera(); isPresented = false }) {
                        Image(systemName: "xmark").font(.system(size: 18, weight: .bold)).foregroundColor(.white)
                            .frame(width: 44, height: 44).background(.ultraThinMaterial).clipShape(Circle())
                    }
                }.padding(.horizontal, 20).padding(.top, 50)

                Spacer()

                if camera.isUltraWideAvailable || camera.isTelephotoAvailable {
                    HStack(spacing: 12) {
                        if camera.isUltraWideAvailable {
                            Button(action: { let t = camera.minZoomFactor; camera.setZoom(t); baseZoomFactor = t }) {
                                Text(".5x").font(.caption.bold())
                                    .foregroundColor(camera.currentZoomFactor < 0.9 ? .yellow : .white)
                                    .frame(width: 38, height: 38).background(Color.black.opacity(0.65)).clipShape(Circle())
                            }
                        }
                        Button(action: { camera.setZoom(1.0); baseZoomFactor = 1.0 }) {
                            Text("1x").font(.caption.bold())
                                .foregroundColor(abs(camera.currentZoomFactor - 1.0) < 0.2 ? .yellow : .white)
                                .frame(width: 38, height: 38).background(Color.black.opacity(0.65)).clipShape(Circle())
                        }
                        if camera.isTelephotoAvailable {
                            Button(action: { camera.setZoom(2.0); baseZoomFactor = 2.0 }) {
                                Text("2x").font(.caption.bold())
                                    .foregroundColor(camera.currentZoomFactor >= 1.8 ? .yellow : .white)
                                    .frame(width: 38, height: 38).background(Color.black.opacity(0.65)).clipShape(Circle())
                            }
                        }
                    }.padding(.bottom, 16)
                }

                HStack(alignment: .center) {
                    Button(action: { showingLibraryPicker = true }) {
                        Image(systemName: "photo.on.rectangle").font(.system(size: 24)).foregroundColor(.white)
                            .frame(width: 54, height: 54).background(Color.white.opacity(0.2)).clipShape(Circle())
                    }
                    Spacer()
                    Button(action: { camera.triggerCapture() }) {
                        ZStack {
                            Circle().strokeBorder(Color.white, lineWidth: 4).frame(width: 78, height: 78)
                            Circle().fill(camera.isCountdownActive ? Color.yellow : Color.white).frame(width: 64, height: 64)
                                .scaleEffect(camera.isCountdownActive ? 0.85 : 1.0).animation(.spring(), value: camera.isCountdownActive)
                        }
                    }
                    Spacer()
                    Button(action: { camera.switchCamera() }) {
                        Image(systemName: "camera.rotate.fill").font(.system(size: 24)).foregroundColor(.white)
                            .frame(width: 54, height: 54).background(Color.white.opacity(0.2)).clipShape(Circle())
                    }
                }.padding(.horizontal, 30).padding(.bottom, 40)
            }
        }
        .onAppear {
            camera.onPhotoCaptured = { photo in
                camera.stopCamera(); isPresented = false; onPhotoCaptured(photo)
            }
            camera.setupCamera()
        }
        .onDisappear { camera.stopCamera() }
        .sheet(isPresented: $showingLibraryPicker) {
            ImagePicker(image: $pickerImage, sourceType: .photoLibrary)
                .onDisappear {
                    if let img = pickerImage { camera.stopCamera(); isPresented = false; onPhotoCaptured(img) }
                }
        }
    }

    private var flashIcon: String {
        switch camera.flashMode {
        case .auto: return "bolt.badge.a.fill"
        case .on: return "bolt.fill"
        case .off: return "bolt.slash.fill"
        @unknown default: return "bolt.badge.a.fill"
        }
    }
}

// MARK: - Camera Preview UIViewRepresentable
struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var camera: CameraController
    func makeUIView(context: Context) -> CameraPreviewUIView {
        let v = CameraPreviewUIView()
        v.previewLayer.session = camera.session
        v.previewLayer.videoGravity = .resizeAspectFill
        return v
    }
    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {}
}

class CameraPreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

// MARK: - Camera Grid
struct CameraGridView: View {
    var body: some View {
        GeometryReader { geo in
            Path { path in
                let (w, h) = (geo.size.width, geo.size.height)
                path.move(to: CGPoint(x: w/3, y: 0)); path.addLine(to: CGPoint(x: w/3, y: h))
                path.move(to: CGPoint(x: 2*w/3, y: 0)); path.addLine(to: CGPoint(x: 2*w/3, y: h))
                path.move(to: CGPoint(x: 0, y: h/3)); path.addLine(to: CGPoint(x: w, y: h/3))
                path.move(to: CGPoint(x: 0, y: 2*h/3)); path.addLine(to: CGPoint(x: w, y: 2*h/3))
            }.stroke(Color.white.opacity(0.3), lineWidth: 1)
        }.allowsHitTesting(false)
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
        picker.sourceType = (sourceType == .camera && UIImagePickerController.isSourceTypeAvailable(.camera)) ? .camera : .photoLibrary
        return picker
    }
    func updateUIViewController(_ vc: UIImagePickerController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker
        init(_ p: ImagePicker) { parent = p }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            parent.image = info[.originalImage] as? UIImage
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