import Foundation

struct LookSession: Identifiable, Codable {
    let id: UUID
    var title: String
    var occasion: OccasionCategory
    let createdAt: Date
    var looks: [LookItem]
    var tags: [String]

    init(
        id: UUID = UUID(),
        title: String,
        occasion: OccasionCategory,
        createdAt: Date = Date(),
        looks: [LookItem] = [],
        tags: [String]? = nil
    ) {
        self.id = id
        self.title = title
        self.occasion = occasion
        self.createdAt = createdAt
        self.looks = looks
        self.tags = tags ?? Self.defaultTags(for: occasion)
    }

    enum CodingKeys: String, CodingKey {
        case id, title, occasion, createdAt, looks, tags
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        title = try container.decode(String.self, forKey: .title)
        occasion = try container.decode(OccasionCategory.self, forKey: .occasion)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        looks = try container.decode([LookItem].self, forKey: .looks)
        tags = try container.decodeIfPresent([String].self, forKey: .tags) ?? Self.defaultTags(for: occasion)
    }

    var bestLook: LookItem? { looks.max(by: { $0.score < $1.score }) }
    var latestLook: LookItem? { looks.last }
    var firstLook: LookItem? { looks.first }

    var averageScore: Double {
        guard !looks.isEmpty else { return 0 }
        return looks.reduce(0) { $0 + $1.score } / Double(looks.count)
    }

    var formattedDate: String {
        let f = DateFormatter()
        f.dateFormat = "MMM d"
        return f.string(from: createdAt)
    }

    var tagsFormatted: String {
        tags.joined(separator: " | ")
    }

    static func defaultTags(for occasion: OccasionCategory) -> [String] {
        switch occasion {
        case .businessMeeting:
            return ["Formal", "Confident", "Sharp"]
        case .dateNight:
            return ["Romantic", "Stylish", "Elegant"]
        case .casualEveryday:
            return ["Relaxed", "Clean", "Versatile"]
        case .formalEvent:
            return ["Black Tie", "Structured", "Polished"]
        case .custom:
            return ["Curated", "Modern", "Distinct"]
        }
    }
}
