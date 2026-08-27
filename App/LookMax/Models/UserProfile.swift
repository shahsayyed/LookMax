import Foundation

struct UserProfile: Codable {
    var name: String
    var photoDataList: [Data]
    var signatures: [FaceBiometricSignature]
    var dateCreated: Date

    static let storageKey = "LookMax_UserProfile"

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
