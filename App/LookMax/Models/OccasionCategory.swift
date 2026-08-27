import SwiftUI

enum OccasionCategory: String, Codable, CaseIterable {
    case businessMeeting = "Business Meeting"
    case dateNight = "Date Night"
    case casualEveryday = "Casual Everyday"
    case formalEvent = "Formal Event"
    case custom = "Custom"

    var icon: String {
        switch self {
        case .businessMeeting: return "briefcase.fill"
        case .dateNight:       return "heart.fill"
        case .casualEveryday:  return "sun.max.fill"
        case .formalEvent:     return "star.fill"
        case .custom:          return "tag.fill"
        }
    }

    var color: Color {
        switch self {
        case .businessMeeting: return .blue
        case .dateNight:       return .pink
        case .casualEveryday:  return .orange
        case .formalEvent:     return .purple
        case .custom:          return .gray
        }
    }
}
