import Foundation

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
