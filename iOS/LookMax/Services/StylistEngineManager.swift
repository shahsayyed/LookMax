import Foundation
import SwiftUI
import CoreML

class StylistEngineManager {
    static let shared = StylistEngineManager()

    private init() {}

    // MARK: - Canonical Occasion Mapping
    static func sanitizeOccasion(_ rawOccasion: String) -> String {
        let clean = rawOccasion.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if clean.contains("tech") || clean.contains("coding") || clean.contains("software") {
            return "Tech Job Interview"
        }
        if clean.contains("creative") || clean.contains("design") || clean.contains("portfolio") {
            return "Creative Field Interview"
        }
        if clean.contains("interview") {
            return "Tech Job Interview"
        }
        if clean.contains("business") || clean.contains("meeting") || clean.contains("corporate") || clean.contains("office") {
            return "Formal Business Meeting"
        }
        if clean.contains("errand") || clean.contains("daily") || clean.contains("casual everyday") || clean.contains("groceries") {
            return "Everyday Errands"
        }
        if clean.contains("date") || clean.contains("romantic") {
            return "First Date"
        }
        if clean.contains("night") || clean.contains("party") || clean.contains("club") || clean.contains("drinks") || clean.contains("cocktail") {
            return "Night Out"
        }
        if clean.contains("gym") || clean.contains("workout") || clean.contains("fitness") || clean.contains("athletic") {
            return "Gym / Athleisure"
        }
        if clean.contains("wedding") || clean.contains("formal event") || clean.contains("reception") {
            return "Summer Wedding"
        }
        if clean.contains("family") || clean.contains("reunion") || clean.contains("bbq") {
            return "Family Gathering"
        }
        return "Casual Dinner"
    }

    // MARK: - Domain Isolation Filter
    static func enforceDomainIsolation(category: String, adviceLines: [String], score: Double) -> [String] {
        let isGrooming = category.contains("Grooming")
        let isMen = category.contains("Men")

        let outfitKeywords = Set([
            "shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "heel", "heels",
            "loafer", "loafers", "oxford", "oxfords", "pump", "pumps", "footwear",
            "belt", "belts", "buckle", "blazer", "blazers", "jacket", "jackets",
            "suit", "suits", "coat", "coats", "shirt", "shirts", "tee", "tees", "polo",
            "top", "tops", "shoulder", "shoulders", "trousers", "pants", "jeans", "denim",
            "chinos", "shorts", "skirt", "skirts", "dress", "dresses", "socks", "tie", "ties",
            "cufflinks", "cuff", "cuffs", "collar", "collars", "hem", "hems", "sleeves",
            "tuck", "tucked", "tucking", "steaming", "steam", "ironing", "iron", "outfit",
            "clothing", "clothes", "garment", "garments", "blouse", "sweater", "hoodie", "cardigan"
        ])

        let groomingKeywords = Set([
            "shave", "shaving", "razor", "beard", "mustache", "stubble", "facial hair",
            "mascara", "lipstick", "eyebrow", "eyebrows", "brows", "skincare", "haircut",
            "comb hair", "brush hair", "pomade", "styling clay", "cleanser", "moisturizer"
        ])

        let forbidden = isGrooming ? outfitKeywords : groomingKeywords
        var cleaned: [String] = []

        for line in adviceLines {
            let words = line.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty }

            let hasForbidden = words.contains(where: { forbidden.contains($0) })
            if !hasForbidden {
                cleaned.append(line)
            }
        }

        // Supply domain-pure fallbacks if filtered below 2 lines
        if cleaned.count < 2 {
            if isGrooming {
                if isMen {
                    if score >= 7.5 {
                        cleaned.append("Apply a lightweight matte styling cream or clay to tame flyaways.")
                        cleaned.append("Keep facial hair lines precisely defined and moisturize skin.")
                    } else {
                        cleaned.append("Tidy up the neckline and sideburns with a precision trimmer.")
                        cleaned.append("Work a dab of pomade or styling paste into hair for intentional hold.")
                    }
                } else {
                    if score >= 7.5 {
                        cleaned.append("Smooth hairline flyaways with a light finishing serum or edge brush.")
                        cleaned.append("Set brows with clear gel and apply a nourishing lip balm for a clean glow.")
                    } else {
                        cleaned.append("Brush and shape eyebrows, setting them lightly with a brow gel.")
                        cleaned.append("Style hair to reduce flyaways and add clean texture or volume.")
                    }
                }
            } else {
                if score >= 7.5 {
                    cleaned.append("Ensure garments remain crisp and unwrinkled throughout the occasion.")
                    cleaned.append("Keep footwear clean and maintain clean posture for a polished silhouette.")
                } else {
                    cleaned.append("Steam or iron garments to eliminate wrinkles and sharpen the silhouette.")
                    cleaned.append("Adjust fit and proportions to ensure clean drape and structured lines.")
                }
            }
        }

        return Array(cleaned.prefix(4))
    }

    // MARK: - Generate Actionable Checklist
    func generateSuggestions(
        category: String,
        occasion: OccasionCategory,
        score: Double,
        visionResult: LookMaxVisionResult
    ) -> [StyleSuggestion] {
        let isGrooming = category.contains("Grooming")
        let canonOccasion = Self.sanitizeOccasion(occasion.rawValue)

        // Raw candidate lines based on vision detections
        var candidateLines: [String] = []

        if isGrooming {
            if let hairStyled = visionResult.attributes["hair_styled"], hairStyled == "0" {
                candidateLines.append("Style hair with a matte pomade or styling cream for intentional shape.")
            }
            if (visionResult.numericAttributes["hair_untidy"] ?? 0) > 0 {
                candidateLines.append("Tame flyaways and stray hairline strands with a touch of finishing balm.")
            }
            if let beardGroomed = visionResult.attributes["facial_hair_groomed"], beardGroomed == "0" {
                candidateLines.append("Clean up and define beard cheeklines and neckline with a trimmer.")
            }
            if let eyebrowsGroomed = visionResult.attributes["eyebrows_groomed"], eyebrowsGroomed == "0" {
                candidateLines.append("Groom and set eyebrows upward with a spoolie brush.")
            }
            if score >= 8.0 {
                candidateLines.append("Hydrate face with a lightweight daily moisturizer for a refreshed look.")
            }
        } else {
            if (visionResult.numericAttributes["fabric_wrinkled"] ?? 0) > 0 {
                candidateLines.append("Steam or press garments to remove visible creases before leaving.")
            }
            if (visionResult.numericAttributes["footwear_worn"] ?? 0) > 0 {
                candidateLines.append("Clean and wipe footwear along sole edges for a crisp finish.")
            }
            if let upper = visionResult.attributes["upper_type"], upper == "plain_crewneck_tee" && canonOccasion.contains("Dinner") {
                candidateLines.append("Layer with a structured overshirt or casual jacket to elevate the dinner look.")
            }
            if let footwear = visionResult.attributes["footwear_type"], footwear.contains("canvas") {
                candidateLines.append("Swap canvas sneakers for clean leather sneakers or loafers.")
            }
            if score >= 8.0 {
                candidateLines.append("Ensure trousers have a clean break above shoes with no excess pooling.")
            }
        }

        // Domain isolation guarantee
        let cleanLines = Self.enforceDomainIsolation(category: category, adviceLines: candidateLines, score: score)

        // Map to StyleSuggestion models
        return cleanLines.enumerated().map { index, line in
            let (catName, icon, color) = suggestionMetadata(for: line, isGrooming: isGrooming)
            return StyleSuggestion(
                category: catName,
                icon: icon,
                iconColor: color,
                title: suggestionTitle(for: line),
                recommendation: line,
                effortTime: "\(2 + (index * 2)) min",
                pointImpact: 0.4
            )
        }
    }

    private func suggestionTitle(for line: String) -> String {
        let lower = line.lowercased()
        if lower.contains("hair") || lower.contains("flyaways") { return "Hairline & Shape" }
        if lower.contains("beard") || lower.contains("trimmer") || lower.contains("shaven") { return "Facial Hair Lineup" }
        if lower.contains("steam") || lower.contains("wrinkle") { return "De-Wrinkle Garments" }
        if lower.contains("shoe") || lower.contains("sneaker") || lower.contains("footwear") { return "Footwear Polish" }
        if lower.contains("eyebrow") || lower.contains("brows") { return "Brow Definition" }
        if lower.contains("layer") || lower.contains("jacket") || lower.contains("blazer") { return "Layering Accent" }
        return "Styling Alignment"
    }

    private func suggestionMetadata(for line: String, isGrooming: Bool) -> (String, String, Color) {
        let lower = line.lowercased()
        if lower.contains("hair") {
            return ("Hair Styling", "comb.fill", Theme.neonCyan)
        }
        if lower.contains("beard") || lower.contains("trimmer") {
            return ("Grooming Lineup", "mustache.fill", Theme.warmAmber)
        }
        if lower.contains("steam") || lower.contains("wrinkle") {
            return ("Fabric Care", "sparkles", Theme.electricBlue)
        }
        if lower.contains("shoe") || lower.contains("sneaker") || lower.contains("footwear") {
            return ("Footwear", "shoeprints.fill", Theme.purpleAccent)
        }
        if lower.contains("brow") {
            return ("Brow Grooming", "eye.fill", Theme.emerald)
        }
        return (isGrooming ? "Grooming Polish" : "Outfit Silhouette", "sparkle", Theme.neonCyan)
    }
}
