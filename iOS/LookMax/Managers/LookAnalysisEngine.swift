import Foundation
import Vision
import CoreImage
import SwiftUI

enum LookAnalysisEngine {

    static func analyze(
        faces: [VNFaceObservation],
        faceQualityRequest: VNDetectFaceCaptureQualityRequest,
        bodyPoses: [VNHumanBodyPoseObservation],
        classifications: [VNClassificationObservation],
        cgImage: CGImage,
        occasion: OccasionCategory
    ) -> LookAnalysisResult {

        var goodPoints: [String] = []
        var badPoints: [String] = []
        var suggestions: [StyleSuggestion] = []

        // ─── Occasion-Based Scoring Baseline ───
        // Each occasion has a different baseline and strictness. More formal events
        // have a lower initial baseline (harder to impress) and penalise casual looks harder.
        let occasionProfile = LookAnalysisEngine.occasionProfile(for: occasion)
        var baseScore = occasionProfile.baseline
        var postureScore = occasionProfile.postureBaseline
        var fitScore = occasionProfile.fitBaseline
        var groomingScore = 7.5

        var postureNote = "Balanced posture"
        var fitNote = "Proportional fit"
        var styleNote = "Cohesive color palette"

        // Tell the user which event standards their look is being evaluated against
        goodPoints.append("Evaluated against \(occasion.rawValue) standards (baseline: \(String(format: "%.1f", occasionProfile.baseline))/10).")

        // ─── Face Shape ───
        var faceShape = "Balanced Oval"
        if let face = faces.first {
            let ratio = face.boundingBox.height / max(0.01, face.boundingBox.width)
            if ratio > 1.35      { faceShape = "Elongated / Oblong" }
            else if ratio < 1.15 { faceShape = "Round / Square" }
        }

        // ─── Lighting & Capture Quality ───
        var lightingScore = 70
        var qualityScore = 0.7
        if let q = faceQualityRequest.results?.first?.faceCaptureQuality {
            qualityScore = Double(q)
            lightingScore = Int(qualityScore * 100)
        }

        if qualityScore >= 0.65 {
            goodPoints.append("Well-lit portrait with minimal shadow and crisp sharpness (\(lightingScore)% clarity).")
            baseScore += 0.4
            groomingScore += 0.6
        } else if qualityScore >= 0.45 {
            badPoints.append("Lighting slightly flat or soft – ideal clarity score is 65%+, yours is \(lightingScore)%.")
            suggestions.append(StyleSuggestion(
                category: "Lighting", icon: "sun.max.fill", iconColor: .yellow,
                title: "Face a Natural Light Source",
                recommendation: "Turn toward a window or soft lamp at 45° to your face. Avoid harsh overhead or direct sunlight.",
                effortTime: "10 sec"
            ))
        } else {
            badPoints.append("Poor lighting conditions — photo appears dark or backlit (\(lightingScore)% clarity). Reshoot near natural light.")
            baseScore -= 0.5
            groomingScore -= 0.8
            suggestions.append(StyleSuggestion(
                category: "Lighting", icon: "sun.max.fill", iconColor: .yellow,
                title: "Move to Better Light Immediately",
                recommendation: "Face a window or turn on a lamp behind your phone. Avoid backlighting.",
                effortTime: "15 sec"
            ))
        }

        // ─── Face Detection ───
        if faces.isEmpty {
            badPoints.append("No clear face detected – ensure your face is visible and well-framed.")
        } else {
            goodPoints.append("Clear face detection with well-centered framing.")
        }

        // ─── Posture & Body Pose ───
        var postureGood = false
        if let body = bodyPoses.first {
            if let neck = try? body.recognizedPoint(.neck),
               let root = try? body.recognizedPoint(.root),
               neck.confidence > 0.3, root.confidence > 0.3 {
                let dx = abs(Double(neck.location.x - root.location.x))
                if dx < 0.04 {
                    goodPoints.append("Upright, aligned posture projects confidence and authority.")
                    baseScore += 0.5
                    postureScore = 8.2
                    postureGood = true
                    postureNote = "Aligned spine & upright shoulders"
                } else {
                    badPoints.append("Slight postural lean detected – shoulders appear uneven.")
                    postureScore = 6.2
                    postureNote = "Slight lateral lean / slouch"
                }
            }
        }
        if !postureGood {
            suggestions.append(StyleSuggestion(
                category: "Posture & Angle", icon: "figure.walk", iconColor: .teal,
                title: "Straighten Posture & Roll Shoulders",
                recommendation: "Pull shoulder blades gently down and back, lifting chest slightly for a confident silhouette.",
                effortTime: "5 sec"
            ))
            suggestions.append(StyleSuggestion(
                category: "Angle", icon: "arrow.triangle.2.circlepath", iconColor: .teal,
                title: "Turn Shoulders 15° for Depth",
                recommendation: "Rotate body slightly while keeping face forward for a slimmer, more dynamic silhouette.",
                effortTime: "5 sec"
            ))
        }

        if faces.first != nil {
            suggestions.append(StyleSuggestion(
                category: "Chin & Jawline", icon: "arrow.up.and.down.and.sparkles", iconColor: .green,
                title: "Extend Chin Forward & Slightly Down",
                recommendation: "Push ears slightly forward and lower chin ~5° to sharpen the jawline and remove under-chin softness.",
                effortTime: "5 sec"
            ))
        }

        // ─── Hairstyle / Beard ───
        let hairKeywords = ["beard", "mustache", "afro", "curls"]
        let hasBeard = classifications.contains { item in
            hairKeywords.contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.25
        }

        if faceShape == "Round / Square" {
            badPoints.append("Round/Square face shape benefits from vertical height – avoid flat, wide hair styles.")
            suggestions.append(StyleSuggestion(
                category: "Hairstyle", icon: "comb.fill", iconColor: .orange,
                title: "Add Height on Top or High Part",
                recommendation: "A higher side part or slight volume on top creates an elongating effect.",
                effortTime: "1 min"
            ))
        } else if faceShape == "Elongated / Oblong" {
            suggestions.append(StyleSuggestion(
                category: "Hairstyle", icon: "comb.fill", iconColor: .orange,
                title: "Keep Sides Full, Avoid Extra Height",
                recommendation: "Side volume and textured layers balance an elongated face.",
                effortTime: "1 min"
            ))
        } else {
            goodPoints.append("Balanced oval face shape – most hairstyles and frame shapes suit your proportions well.")
        }

        if hasBeard {
            groomingScore += 0.3
            suggestions.append(StyleSuggestion(
                category: "Beard Lineup", icon: "scissors", iconColor: .brown,
                title: "Define Neckline Two Fingers Above Adam's Apple",
                recommendation: "A sharp, clean neckline at this height instantly sculpts the jawline.",
                effortTime: "2 mins"
            ))
        } else {
            suggestions.append(StyleSuggestion(
                category: "Grooming", icon: "comb.fill", iconColor: .orange,
                title: "Tame Flyaways with Matte Paste",
                recommendation: "A pea-sized amount of matte paste keeps your silhouette sharp under bright light.",
                effortTime: "30 sec"
            ))
        }

        // ─── Eyewear ───
        let eyewearMatches = classifications.filter { item in
            ["sunglass", "glasses", "spectacles", "eyewear"].contains { kw in
                item.identifier.lowercased().contains(kw)
            } && item.confidence > 0.3
        }
        if !eyewearMatches.isEmpty {
            goodPoints.append("Eyewear detected – adds structure and visual interest to the eye region.")
            suggestions.append(StyleSuggestion(
                category: "Glasses Position", icon: "eyeglasses", iconColor: .blue,
                title: "Align Bridge: Pupils in Upper Lens Third",
                recommendation: "Slide frames slightly up so your pupils sit in the upper third of each lens.",
                effortTime: "10 sec"
            ))
        }

        // ─── Outfit / Apparel ───
        let dominantColor = sampleDominantColorName(in: cgImage, normalizedRect: CGRect(x: 0.35, y: 0.45, width: 0.30, height: 0.25)) ?? "Neutral tone"
        let formalKeywords = ["suit", "blazer", "tie", "jacket", "shirt", "collar", "dress"]
        let casualKeywords = ["t-shirt", "hoodie", "sweater", "jersey", "denim"]

        let isFormal = classifications.contains { item in
            formalKeywords.contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.25
        }
        let isCasual = classifications.contains { item in
            casualKeywords.contains { kw in item.identifier.lowercased().contains(kw) } && item.confidence > 0.25
        }

        let occasionFormalExpected: Bool
        switch occasion {
        case .businessMeeting, .formalEvent: occasionFormalExpected = true
        default: occasionFormalExpected = false
        }

        if isFormal {
            goodPoints.append("Structured, formal attire with a strong collar and lapel outline.")
            baseScore += 0.5
            fitScore = 8.4
            fitNote = "Collar aligned & crisp lapels"
            if occasionFormalExpected {
                goodPoints.append("Outfit is well-matched for the occasion (\(occasion.rawValue)).")
                baseScore += 0.3
                styleNote = "Executive & Formal"
            }
            suggestions.append(StyleSuggestion(
                category: "Collar & Lapel", icon: "tshirt.fill", iconColor: .purple,
                title: "Straighten Collar Points & Center Lapels",
                recommendation: "Check that shirt collar points lie flat under jacket lapels without curling.",
                effortTime: "30 sec"
            ))
            suggestions.append(StyleSuggestion(
                category: "Shirt Tuck", icon: "arrow.down.to.line", iconColor: .purple,
                title: "Tuck Shirt Neatly (Military Tuck)",
                recommendation: "Pinch excess fabric at the side seams and fold backward before buckling your belt.",
                effortTime: "30 sec"
            ))
            suggestions.append(StyleSuggestion(
                category: "Accessories", icon: "watch.analog", iconColor: .blue,
                title: "Align Watch & Cuff Length",
                recommendation: "Show 1/4 to 1/2 inch of shirt cuff below your jacket sleeve.",
                effortTime: "10 sec"
            ))
        } else if isCasual {
            if occasionFormalExpected {
                let formalityPenalty = occasionProfile.formalityWeight * 0.4
                badPoints.append("Casual outfit may be underdressed for \(occasion.rawValue). Consider adding a blazer.")
                baseScore -= formalityPenalty
                fitScore = max(5.8, 6.8 - formalityPenalty * 0.5)
                fitNote = "Relaxed / under-structured"
                styleNote = "Too informal for occasion"
                suggestions.append(StyleSuggestion(
                    category: "Outfit Upgrade", icon: "tshirt.fill", iconColor: .red,
                    title: "Add a Blazer or Structured Jacket",
                    recommendation: "A navy or grey blazer over almost any outfit instantly elevates the look by 2–3 points.",
                    effortTime: "2 mins"
                ))
            } else {
                goodPoints.append("Relaxed, appropriate casual outfit for \(occasion.rawValue).")
                fitScore = 7.6
                fitNote = "Clean casual fit"
                styleNote = "Versatile Everyday"
                suggestions.append(StyleSuggestion(
                    category: "Casual Layering", icon: "tshirt.fill", iconColor: .purple,
                    title: "Layer for Depth & Dimension",
                    recommendation: "An open overshirt or unzipped jacket adds visual structure and broadens the shoulder frame.",
                    effortTime: "2 mins"
                ))
            }
        } else {
            fitScore = 7.2
            fitNote = "Standard proportion"
            suggestions.append(StyleSuggestion(
                category: "Outfit Framing", icon: "tshirt.fill", iconColor: .purple,
                title: "Check Neckline & Shoulder Fit",
                recommendation: "Ensure your top's neckline sits clean and the shoulders align to your actual shoulder line.",
                effortTime: "1 min"
            ))
        }

        if dominantColor.contains("Dark") || dominantColor.contains("Black") || dominantColor.contains("Navy") {
            goodPoints.append("Dark \(dominantColor.lowercased()) top creates high contrast against skin tones, sharpening facial features.")
        } else if dominantColor.contains("Light") || dominantColor.contains("White") {
            goodPoints.append("Light/white top reflects flattering illumination toward the face, brightening the overall look.")
        }

        // ─── Final Score ───
        let finalScore = max(6.5, min(9.8, baseScore))
        let headline: String
        switch finalScore {
        case 9.3...:        headline = "Executive & Flawless"
        case 8.5..<9.3:     headline = "Sharp & Polished"
        case 7.8..<8.5:     headline = "Well-Put-Together"
        case 7.0..<7.8:     headline = "Clean & Casual"
        default:            headline = "Needs a Few Tweaks"
        }

        if goodPoints.isEmpty { goodPoints.append("Natural relaxed presence and authentic expression.") }
        if badPoints.isEmpty  { badPoints.append("No major issues detected – see the 5-min tweaks below for polishing details.") }

        return LookAnalysisResult(
            score: finalScore,
            headlineBadge: headline,
            goodPoints: goodPoints,
            badPoints: badPoints,
            suggestions: suggestions,
            detectedOutfitColor: dominantColor,
            detectedFaceShape: faceShape,
            lightingScore: lightingScore,
            postureScore: min(9.8, max(5.5, postureScore)),
            fitScore: min(9.8, max(5.5, fitScore)),
            groomingScore: min(9.8, max(5.5, groomingScore)),
            postureNote: postureNote,
            fitNote: fitNote,
            styleNote: styleNote
        )
    }

    // MARK: - Occasion Profile
    private struct OccasionProfile {
        let baseline: Double       // Starting score before modifiers
        let postureBaseline: Double
        let fitBaseline: Double
        let formalityWeight: Double  // How much outfit formality mismatch penalises
    }

    private static func occasionProfile(for occasion: OccasionCategory) -> OccasionProfile {
        switch occasion {
        case .formalEvent:
            // Black-tie / wedding: strictest. High bar. Casual outfits heavily penalised.
            return OccasionProfile(baseline: 6.8, postureBaseline: 7.2, fitBaseline: 7.0, formalityWeight: 1.6)
        case .businessMeeting:
            // Professional: expects structured attire; slight leniency vs. formal.
            return OccasionProfile(baseline: 7.0, postureBaseline: 7.2, fitBaseline: 7.2, formalityWeight: 1.3)
        case .dateNight:
            // Stylish but flexible; well-groomed smart-casual is fine.
            return OccasionProfile(baseline: 7.4, postureBaseline: 7.0, fitBaseline: 7.2, formalityWeight: 0.9)
        case .casualEveryday:
            // Most lenient. Clean, comfortable fits score well.
            return OccasionProfile(baseline: 7.6, postureBaseline: 7.0, fitBaseline: 7.0, formalityWeight: 0.5)
        case .custom:
            // Flexible / custom event: generous baseline, moderate standards.
            return OccasionProfile(baseline: 7.5, postureBaseline: 7.0, fitBaseline: 7.0, formalityWeight: 0.6)
        }
    }

    // MARK: - Color Sampling
    private static func sampleDominantColorName(in cgImage: CGImage, normalizedRect: CGRect) -> String? {
        let w = CGFloat(cgImage.width)
        let h = CGFloat(cgImage.height)
        let cropRect = CGRect(
            x: normalizedRect.origin.x * w,
            y: normalizedRect.origin.y * h,
            width: normalizedRect.size.width * w,
            height: normalizedRect.size.height * h
        )
        guard let cropped = cgImage.cropping(to: cropRect) else { return nil }

        let ctx = CGContext(
            data: nil, width: 1, height: 1,
            bitsPerComponent: 8, bytesPerRow: 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        ctx?.draw(cropped, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        guard let data = ctx?.data else { return nil }

        let ptr = data.bindMemory(to: UInt8.self, capacity: 4)
        let r = Double(ptr[0]) / 255
        let g = Double(ptr[1]) / 255
        let b = Double(ptr[2]) / 255
        let brightness = (r + g + b) / 3

        if brightness < 0.20 { return "Dark / Black" }
        if brightness > 0.82 { return "Light / White" }
        let maxDiff = max(abs(r-g), abs(g-b), abs(r-b))
        if maxDiff < 0.08  { return "Neutral Grey" }
        if r > g && r > b  { return g > 0.5 ? "Warm Amber" : "Warm Red / Burgundy" }
        if g > r && g > b  { return "Olive / Green" }
        if b > r && b > g  { return r > 0.4 ? "Purple / Violet" : "Navy / Blue" }
        return "Neutral"
    }
}
