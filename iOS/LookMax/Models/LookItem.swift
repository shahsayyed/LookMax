import UIKit

struct LookItem: Identifiable, Codable {
    let id: UUID
    let imagePath: String
    let timestamp: Date
    let score: Double
    let headlineBadge: String
    let goodPoints: [String]
    let badPoints: [String]
    var suggestions: [StyleSuggestion]
    let detectedOutfitColor: String
    let detectedFaceShape: String
    let lightingScore: Int

    init(
        id: UUID = UUID(),
        imagePath: String,
        timestamp: Date = Date(),
        score: Double,
        headlineBadge: String,
        goodPoints: [String],
        badPoints: [String],
        suggestions: [StyleSuggestion],
        detectedOutfitColor: String,
        detectedFaceShape: String,
        lightingScore: Int
    ) {
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
