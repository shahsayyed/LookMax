import SwiftUI

struct StyleSuggestion: Identifiable, Codable {
    let id: UUID
    let category: String
    let icon: String
    let iconColorHex: String
    let title: String
    let recommendation: String
    let effortTime: String
    var pointImpact: Double
    var isDone: Bool

    init(
        id: UUID = UUID(),
        category: String,
        icon: String,
        iconColor: Color,
        title: String,
        recommendation: String,
        effortTime: String,
        pointImpact: Double = 0.5,
        isDone: Bool = false
    ) {
        self.id = id
        self.category = category
        self.icon = icon
        self.iconColorHex = iconColor.toHex()
        self.title = title
        self.recommendation = recommendation
        self.effortTime = effortTime
        self.pointImpact = pointImpact
        self.isDone = isDone
    }

    enum CodingKeys: String, CodingKey {
        case id, category, icon, iconColorHex, title, recommendation, effortTime, pointImpact, isDone
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        category = try container.decode(String.self, forKey: .category)
        icon = try container.decode(String.self, forKey: .icon)
        iconColorHex = try container.decode(String.self, forKey: .iconColorHex)
        title = try container.decode(String.self, forKey: .title)
        recommendation = try container.decode(String.self, forKey: .recommendation)
        effortTime = try container.decode(String.self, forKey: .effortTime)
        pointImpact = try container.decodeIfPresent(Double.self, forKey: .pointImpact) ?? 0.5
        isDone = try container.decodeIfPresent(Bool.self, forKey: .isDone) ?? false
    }

    var iconColor: Color { Color(hex: iconColorHex) ?? .blue }
}
