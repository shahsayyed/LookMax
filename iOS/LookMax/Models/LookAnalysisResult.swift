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
    let postureScore: Double
    let fitScore: Double
    let groomingScore: Double
    let postureNote: String
    let fitNote: String
    let styleNote: String
}
