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
    
    // Sub-metrics for detailed Before/After analysis
    var postureScore: Double
    var fitScore: Double
    var groomingScore: Double
    var postureNote: String
    var fitNote: String
    var styleNote: String

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
        lightingScore: Int,
        postureScore: Double = 7.0,
        fitScore: Double = 7.2,
        groomingScore: Double = 7.5,
        postureNote: String = "Upright posture",
        fitNote: String = "Balanced fit",
        styleNote: String = "Clean aesthetic"
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
        self.postureScore = postureScore
        self.fitScore = fitScore
        self.groomingScore = groomingScore
        self.postureNote = postureNote
        self.fitNote = fitNote
        self.styleNote = styleNote
    }

    enum CodingKeys: String, CodingKey {
        case id, imagePath, timestamp, score, headlineBadge, goodPoints, badPoints
        case suggestions, detectedOutfitColor, detectedFaceShape, lightingScore
        case postureScore, fitScore, groomingScore, postureNote, fitNote, styleNote
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        imagePath = try container.decode(String.self, forKey: .imagePath)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        score = try container.decode(Double.self, forKey: .score)
        headlineBadge = try container.decode(String.self, forKey: .headlineBadge)
        goodPoints = try container.decode([String].self, forKey: .goodPoints)
        badPoints = try container.decode([String].self, forKey: .badPoints)
        suggestions = try container.decode([StyleSuggestion].self, forKey: .suggestions)
        detectedOutfitColor = try container.decode(String.self, forKey: .detectedOutfitColor)
        detectedFaceShape = try container.decode(String.self, forKey: .detectedFaceShape)
        lightingScore = try container.decode(Int.self, forKey: .lightingScore)
        postureScore = try container.decodeIfPresent(Double.self, forKey: .postureScore) ?? (score * 0.9)
        fitScore = try container.decodeIfPresent(Double.self, forKey: .fitScore) ?? (score * 0.95)
        groomingScore = try container.decodeIfPresent(Double.self, forKey: .groomingScore) ?? (score * 0.92)
        postureNote = try container.decodeIfPresent(String.self, forKey: .postureNote) ?? "Balanced posture"
        fitNote = try container.decodeIfPresent(String.self, forKey: .fitNote) ?? "Good proportion"
        styleNote = try container.decodeIfPresent(String.self, forKey: .styleNote) ?? "Cohesive look"
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
