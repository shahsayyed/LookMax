import Foundation
import UIKit
import Combine

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
