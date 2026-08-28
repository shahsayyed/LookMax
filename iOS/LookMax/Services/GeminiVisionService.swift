import Foundation
import UIKit

// MARK: - Structured JSON Response from Gemini

/// The structured evaluation Gemini returns for each look photo.
struct GeminiLookEvaluation: Codable {
    let formalityScore: Double
    let sharpnessScore: Double
    let occasionMatchScore: Double
    let overallScore: Double
    let headlineBadge: String
    let goodPoints: [String]
    let improvementPoints: [String]
    let tweaks: [GeminiTweak]
    let postureNote: String
    let fitNote: String
    let styleNote: String

    enum CodingKeys: String, CodingKey {
        case formalityScore = "formality_score"
        case sharpnessScore = "sharpness_score"
        case occasionMatchScore = "occasion_match_score"
        case overallScore = "overall_score"
        case headlineBadge = "headline_badge"
        case goodPoints = "good_points"
        case improvementPoints = "improvement_points"
        case tweaks
        case postureNote = "posture_note"
        case fitNote = "fit_note"
        case styleNote = "style_note"
    }
}

struct GeminiTweak: Codable {
    let category: String
    let title: String
    let recommendation: String
    let effortTime: String

    enum CodingKeys: String, CodingKey {
        case category, title, recommendation
        case effortTime = "effort_time"
    }
}

// MARK: - Service Errors

enum GeminiServiceError: Error, LocalizedError {
    case missingAPIKey
    case imageTooLarge
    case networkError(String)
    case parsingError(String)
    case rateLimited

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:    return "Gemini API key not configured. Add GEMINI_API_KEY to Info.plist."
        case .imageTooLarge:    return "Image is too large. Please try a smaller photo."
        case .networkError(let m): return "Network error: \(m)"
        case .parsingError(let m): return "Could not parse AI response: \(m)"
        case .rateLimited:      return "AI analysis limit reached. Please wait a moment."
        }
    }
}

// MARK: - Gemini Vision Service

/// Sends compressed outfit photos to Gemini 3.7 Flash for structured look evaluation.
final class GeminiVisionService {

    static let shared = GeminiVisionService()
    private init() {}

    private var apiKey: String? {
        // Try Info.plist first
        if let key = Bundle.main.object(forInfoDictionaryKey: "GEMINI_API_KEY") as? String,
           !key.isEmpty, !key.hasPrefix("$(") { return key }
        // Fall back to environment variable (for CI / testing)
        let env = ProcessInfo.processInfo.environment["GEMINI_API_KEY"] ?? ""
        return env.isEmpty ? nil : env
    }

    private let endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent"

    // MARK: - Public API

    func evaluate(image: UIImage, occasion: OccasionCategory) async throws -> GeminiLookEvaluation {
        guard let key = apiKey else { throw GeminiServiceError.missingAPIKey }

        guard let imageData = compressImage(image, targetKB: 900) else {
            throw GeminiServiceError.imageTooLarge
        }
        let base64Image = imageData.base64EncodedString()

        let prompt = buildPrompt(for: occasion)
        let body = buildRequestBody(base64Image: base64Image, prompt: prompt)

        let url = URL(string: "\(endpoint)?key=\(key)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 30

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw GeminiServiceError.networkError("Invalid response type")
        }

        switch http.statusCode {
        case 200: return try parseResponse(data: data)
        case 429: throw GeminiServiceError.rateLimited
        default:
            let msg = String(data: data, encoding: .utf8) ?? "Unknown"
            throw GeminiServiceError.networkError("HTTP \(http.statusCode): \(msg)")
        }
    }

    // MARK: - Private Helpers

    private func compressImage(_ image: UIImage, targetKB: Int) -> Data? {
        let maxDim: CGFloat = 1024
        let sz = image.size
        let scale = min(maxDim / sz.width, maxDim / sz.height, 1.0)
        let newSize = CGSize(width: sz.width * scale, height: sz.height * scale)

        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: newSize))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        guard let r = resized else { return nil }
        var q: CGFloat = 0.85
        var data = r.jpegData(compressionQuality: q)
        while let d = data, d.count > targetKB * 1024, q > 0.2 {
            q -= 0.15
            data = r.jpegData(compressionQuality: q)
        }
        return data
    }

    private func buildPrompt(for occasion: OccasionCategory) -> String {
        let ctx: String
        switch occasion {
        case .formalEvent:
            ctx = "a formal black-tie event (e.g., wedding, gala). Suits, tuxedos, or formal dresses expected. Casual wear is heavily penalised."
        case .businessMeeting:
            ctx = "a professional business meeting. Blazer, dress shirt, tailored trousers expected. Smart-casual may be acceptable."
        case .dateNight:
            ctx = "a date night. Smart-casual to semi-formal. Dark jeans + blazer, dresses, or stylish separates are ideal."
        case .casualEveryday:
            ctx = "casual everyday wear. Clean, well-fitting comfortable clothes. Formality is not required."
        case .custom:
            ctx = "a custom occasion. Evaluate as smart-casual with moderate formality expectations."
        }

        return """
You are LookMax, an elite AI style consultant. Analyze this outfit photo and evaluate it for \(ctx)

Evaluate on:
1. formality_score (1-10): How dressy/formal is the clothing?
2. sharpness_score (1-10): Are clothes crisp, clean, well-tailored? Penalise wrinkles, baggy fit, sloppy collar, dropped shoulders, poor trouser break.
3. occasion_match_score (1-10): Does the outfit formality appropriately match the occasion?
4. overall_score (1-10): Weighted composite — sharpness 40%, occasion match 40%, formality 20%.

RULES:
- Be HONEST. A plain t-shirt at a formal event scores 2-4 on occasion_match. A wrinkled shirt scores 4-6 on sharpness.
- Most real-world looks score 6.0–8.0 overall. Reserve 9+ for truly outstanding outfits.
- Provide SPECIFIC observations. Never say "great look" without explaining WHY.

Respond with ONLY valid JSON (no markdown fences):
{
  "formality_score": <float>,
  "sharpness_score": <float>,
  "occasion_match_score": <float>,
  "overall_score": <float>,
  "headline_badge": "<3-4 word label e.g. Sharp & Polished>",
  "good_points": ["<specific observation>", "<another>"],
  "improvement_points": ["<specific critique>", "<another>"],
  "tweaks": [
    {"category": "<label>", "title": "<action>", "recommendation": "<instruction>", "effort_time": "<duration>"},
    {"category": "<label>", "title": "<action>", "recommendation": "<instruction>", "effort_time": "<duration>"},
    {"category": "<label>", "title": "<action>", "recommendation": "<instruction>", "effort_time": "<duration>"}
  ],
  "posture_note": "<posture/silhouette obs>",
  "fit_note": "<fit obs>",
  "style_note": "<style/color obs>"
}
"""
    }

    private func buildRequestBody(base64Image: String, prompt: String) -> [String: Any] {
        [
            "contents": [
                [
                    "parts": [
                        ["text": prompt],
                        ["inline_data": ["mime_type": "image/jpeg", "data": base64Image]]
                    ]
                ]
            ],
            "generationConfig": [
                "temperature": 0.25,
                "topP": 0.85,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json"
            ],
            "safetySettings": [
                ["category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"],
                ["category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"]
            ]
        ]
    }

    private func parseResponse(data: Data) throws -> GeminiLookEvaluation {
        // Gemini wraps output in candidates[0].content.parts[0].text
        guard let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let candidates = raw["candidates"] as? [[String: Any]],
              let first = candidates.first,
              let content = first["content"] as? [String: Any],
              let parts = content["parts"] as? [[String: Any]],
              let text = parts.first?["text"] as? String else {
            throw GeminiServiceError.parsingError("Unexpected Gemini response structure")
        }

        // Strip markdown code fences if present despite responseMimeType setting
        let cleaned = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard let jsonData = cleaned.data(using: .utf8) else {
            throw GeminiServiceError.parsingError("Cannot encode response as UTF-8")
        }

        do {
            return try JSONDecoder().decode(GeminiLookEvaluation.self, from: jsonData)
        } catch {
            throw GeminiServiceError.parsingError("JSON decode: \(error.localizedDescription)\nRaw: \(cleaned.prefix(200))")
        }
    }
}
