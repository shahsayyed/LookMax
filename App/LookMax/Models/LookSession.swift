import Foundation

struct LookSession: Identifiable, Codable {
    let id: UUID
    var title: String
    var occasion: OccasionCategory
    let createdAt: Date
    var looks: [LookItem]

    init(
        id: UUID = UUID(),
        title: String,
        occasion: OccasionCategory,
        createdAt: Date = Date(),
        looks: [LookItem] = []
    ) {
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
