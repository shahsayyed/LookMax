import SwiftUI

struct StyleSuggestion: Identifiable, Codable {
    let id: UUID
    let category: String
    let icon: String
    let iconColorHex: String
    let title: String
    let recommendation: String
    let effortTime: String
    var isDone: Bool

    init(
        id: UUID = UUID(),
        category: String,
        icon: String,
        iconColor: Color,
        title: String,
        recommendation: String,
        effortTime: String,
        isDone: Bool = false
    ) {
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
