import Foundation
import UIKit
import CoreML
import Vision

// MARK: - On-Device Vision Inference Result

struct LookMaxVisionResult {
    let score: Double
    let rawScore: Double
    let tier: String
    let tierLabel: String
    let category: String
    let attributes: [String: String]
    let numericAttributes: [String: Int]
    let activeFlaws: [String]
    let goodPoints: [String]
    let badPoints: [String]
}

// MARK: - CoreML Vision Manager

class CoreMLVisionManager {
    static let shared = CoreMLVisionManager()

    private let config: MLModelConfiguration = {
        let c = MLModelConfiguration()
        c.computeUnits = .all
        return c
    }()

    // Cached model instances
    private var menGroomingModel: LookMax_Men_Grooming?
    private var menOutfitModel: LookMax_Men_Outfit?
    private var womenGroomingModel: LookMax_Women_Grooming?
    private var womenOutfitModel: LookMax_Women_Outfit?

    private init() {
        // Pre-warm models asynchronously on background queue
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            self.menGroomingModel = try? LookMax_Men_Grooming(configuration: self.config)
            self.menOutfitModel = try? LookMax_Men_Outfit(configuration: self.config)
            self.womenGroomingModel = try? LookMax_Women_Grooming(configuration: self.config)
            self.womenOutfitModel = try? LookMax_Women_Outfit(configuration: self.config)
        }
    }

    // MARK: - Public Inference API

    func analyze(image: UIImage, category: String) async throws -> LookMaxVisionResult {
        let isGrooming = category.contains("Grooming")
        let isMen = category.contains("Men")

        // 1. Prepare Target Resolution & Non-Destructive Crop
        let targetSize = isGrooming ? CGSize(width: 384, height: 384) : CGSize(width: 384, height: 512)
        guard let preparedCGImage = prepareImageForInference(image: image, isGrooming: isGrooming, targetSize: targetSize) else {
            throw NSError(domain: "CoreMLVisionManager", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to preprocess image for CoreML inference."])
        }

        // 2. Dispatch Model Prediction
        var rawScore: Double = 5.0
        var attributes: [String: String] = [:]
        var numericAttributes: [String: Int] = [:]

        if isGrooming {
            if isMen {
                let model = try (self.menGroomingModel ?? LookMax_Men_Grooming(configuration: self.config))
                let input = try LookMax_Men_GroomingInput(imageWith: preparedCGImage)
                let out = try await model.prediction(input: input)
                rawScore = Double(out.score[0].floatValue)

                let untidyIdx = argmax(out.hair_untidy)
                let styledIdx = argmax(out.hair_styled)
                let skinNegIdx = argmax(out.skin_neglected)
                let skinHealthIdx = argmax(out.skin_healthy)
                let browsUnkemptIdx = argmax(out.eyebrows_unkempt)
                let browsGroomedIdx = argmax(out.eyebrows_groomed)
                let beardUntidyIdx = argmax(out.facial_hair_untidy)
                let beardGroomedIdx = argmax(out.facial_hair_groomed)
                let beardStyleIdx = argmax(out.facial_hair_style)

                let beardStyles = ["clean_shaven", "five_o_clock_shadow", "short_stubble", "heavy_stubble", "full_short_beard", "medium_full_beard", "long_full_beard", "trimmed_goatee", "stubble_goatee", "mustache_only", "mustache_with_stubble", "van_dyke", "circle_beard", "chin_strap", "anchor_beard", "mutton_chops"]

                attributes["hair_styled"] = "\(styledIdx)"
                attributes["hair_untidy"] = "\(untidyIdx)"
                attributes["skin_healthy"] = "\(skinHealthIdx)"
                attributes["skin_neglected"] = "\(skinNegIdx)"
                attributes["eyebrows_groomed"] = "\(browsGroomedIdx)"
                attributes["eyebrows_unkempt"] = "\(browsUnkemptIdx)"
                attributes["facial_hair_groomed"] = "\(beardGroomedIdx)"
                attributes["facial_hair_style"] = beardStyleIdx < beardStyles.count ? beardStyles[beardStyleIdx] : "clean_shaven"

                numericAttributes["hair_untidy"] = untidyIdx
                numericAttributes["skin_neglected"] = skinNegIdx
                numericAttributes["eyebrows_unkempt"] = browsUnkemptIdx
                numericAttributes["facial_hair_untidy"] = beardUntidyIdx
            } else {
                let model = try (self.womenGroomingModel ?? LookMax_Women_Grooming(configuration: self.config))
                let input = try LookMax_Women_GroomingInput(imageWith: preparedCGImage)
                let out = try await model.prediction(input: input)
                rawScore = Double(out.score[0].floatValue)

                let untidyIdx = argmax(out.hair_untidy)
                let styledIdx = argmax(out.hair_styled)
                let skinNegIdx = argmax(out.skin_neglected)
                let skinHealthIdx = argmax(out.skin_healthy)
                let browsUnkemptIdx = argmax(out.eyebrows_unkempt)
                let browsGroomedIdx = argmax(out.eyebrows_groomed)
                let makeupStyleIdx = argmax(out.makeup_style)
                let makeupUnevenIdx = argmax(out.makeup_uneven)

                let makeupStyles = ["none", "minimal", "dewy", "everyday_neutral", "soft_glam", "bold_lip", "smokey_eye", "dramatic"]

                attributes["hair_styled"] = "\(styledIdx)"
                attributes["hair_untidy"] = "\(untidyIdx)"
                attributes["skin_healthy"] = "\(skinHealthIdx)"
                attributes["skin_neglected"] = "\(skinNegIdx)"
                attributes["eyebrows_groomed"] = "\(browsGroomedIdx)"
                attributes["eyebrows_unkempt"] = "\(browsUnkemptIdx)"
                attributes["makeup_style"] = makeupStyleIdx < makeupStyles.count ? makeupStyles[makeupStyleIdx] : "minimal"
                attributes["makeup_uneven"] = "\(makeupUnevenIdx)"

                numericAttributes["hair_untidy"] = untidyIdx
                numericAttributes["skin_neglected"] = skinNegIdx
                numericAttributes["eyebrows_unkempt"] = browsUnkemptIdx
                numericAttributes["makeup_uneven"] = makeupUnevenIdx
            }
        } else {
            // Outfit Predictions
            if isMen {
                let model = try (self.menOutfitModel ?? LookMax_Men_Outfit(configuration: self.config))
                let input = try LookMax_Men_OutfitInput(imageWith: preparedCGImage)
                let out = try await model.prediction(input: input)
                rawScore = Double(out.score[0].floatValue)

                let uTypes = ["tank_top", "graphic_tee", "plain_crewneck_tee", "henley", "casual_polo", "dress_shirt", "oxford_button_down", "flannel_shirt", "chambray_shirt", "oversized_tee", "linen_shirt", "knit_polo", "grandad_collar_shirt", "camp_collar_shirt"]
                let lTypes = ["denim_jeans", "distressed_jeans", "chino_pants", "tailored_trousers", "jogger_pants", "cargo_pants", "drawstring_linen_pants", "pleated_dress_trousers", "corduroy_trousers", "slim_fit_chinos", "wide_leg_trousers", "raw_denim_jeans"]
                let fTypes = ["athletic_running_shoes", "canvas_sneakers", "leather_low_top_sneakers", "leather_loafers", "dress_oxfords", "leather_derby_shoes", "chelsea_boots", "combat_boots", "dress_monk_straps", "minimalist_white_sneakers", "suede_loafers", "chukka_boots"]
                let forms = ["ultra_casual", "casual", "smart_casual", "business_casual", "formal"]

                let uIdx = argmax(out.upper_type)
                let lIdx = argmax(out.lower_type)
                let fIdx = argmax(out.footwear_type)
                let formIdx = argmax(out.formality)

                attributes["upper_type"] = uIdx < uTypes.count ? uTypes[uIdx] : "plain_crewneck_tee"
                attributes["lower_type"] = lIdx < lTypes.count ? lTypes[lIdx] : "denim_jeans"
                attributes["footwear_type"] = fIdx < fTypes.count ? fTypes[fIdx] : "canvas_sneakers"
                attributes["formality"] = formIdx < forms.count ? forms[formIdx] : "casual"

                numericAttributes["fabric_wrinkled"] = argmax(out.fabric_wrinkled)
                numericAttributes["fit_baggy"] = argmax(out.fit_baggy)
                numericAttributes["fit_tight"] = argmax(out.fit_tight)
                numericAttributes["footwear_worn"] = argmax(out.footwear_worn)
                numericAttributes["styling_sloppy"] = argmax(out.styling_sloppy)
            } else {
                let model = try (self.womenOutfitModel ?? LookMax_Women_Outfit(configuration: self.config))
                let input = try LookMax_Women_OutfitInput(imageWith: preparedCGImage)
                let out = try await model.prediction(input: input)
                rawScore = Double(out.score[0].floatValue)

                let uTypes = ["tank_top", "graphic_tee", "casual_camisole", "henley", "ribbed_tank", "casual_tee", "tailored_blouse", "silk_button_up", "wrap_top", "off_the_shoulder_top", "linen_blouse", "bodysuit", "fitted_turtleneck", "peplum_top", "crop_blouse", "sleeveless_blouse"]
                let lTypes = ["denim_jeans", "distressed_jeans", "tailored_trousers", "wide_leg_trousers", "midi_skirt", "mini_skirt", "cargo_pants", "linen_wide_leg_pants", "pleated_trousers", "high_waisted_culottes", "satin_slip_skirt", "leather_pants"]
                let fTypes = ["athletic_sneakers", "canvas_sneakers", "leather_white_sneakers", "heeled_pumps", "strappy_heels", "ankle_boots", "ballet_flats", "mules", "combat_boots", "pointed_toe_flats", "chunky_loafers", "block_heel_sandals"]
                let forms = ["ultra_casual", "casual", "smart_casual", "business_casual", "formal"]

                let uIdx = argmax(out.upper_type)
                let lIdx = argmax(out.lower_type)
                let fIdx = argmax(out.footwear_type)
                let formIdx = argmax(out.formality)

                attributes["upper_type"] = uIdx < uTypes.count ? uTypes[uIdx] : "tailored_blouse"
                attributes["lower_type"] = lIdx < lTypes.count ? lTypes[lIdx] : "tailored_trousers"
                attributes["footwear_type"] = fIdx < fTypes.count ? fTypes[fIdx] : "heeled_pumps"
                attributes["formality"] = formIdx < forms.count ? forms[formIdx] : "smart_casual"

                numericAttributes["fabric_wrinkled"] = argmax(out.fabric_wrinkled)
                numericAttributes["fit_baggy"] = argmax(out.fit_baggy)
                numericAttributes["fit_tight"] = argmax(out.fit_tight)
                numericAttributes["footwear_worn"] = argmax(out.footwear_worn)
                numericAttributes["styling_sloppy"] = argmax(out.styling_sloppy)
            }
        }

        // 3. Flaw-Consistency Score Calibration
        var hasActiveFlaw = false
        var activeFlaws: [String] = []
        var goodPoints: [String] = []
        var badPoints: [String] = []

        if isGrooming {
            if attributes["hair_styled"] == "0" {
                hasActiveFlaw = true
                activeFlaws.append("Hair unstyled or lacking product")
                badPoints.append("Hair lacks intentional styling or product.")
            } else {
                goodPoints.append("Hair styled with clean hold and intentional shape.")
            }

            if (numericAttributes["hair_untidy"] ?? 0) > 0 {
                hasActiveFlaw = true
                activeFlaws.append("Visible hair flyaways or untidy strands")
                badPoints.append("Visible stray strands or flyaways detected.")
            }

            if attributes["facial_hair_groomed"] == "0" && attributes["facial_hair_style"] != "clean_shaven" {
                hasActiveFlaw = true
                activeFlaws.append("Beard neckline/edges need precision trimming")
                badPoints.append("Beard edges or neckline appear un-groomed.")
            } else if attributes["facial_hair_groomed"] == "1" {
                goodPoints.append("Sharp facial hair lining and well-maintained profile.")
            }

            if attributes["skin_healthy"] == "1" {
                goodPoints.append("Hydrated, healthy skin appearance.")
            }
        } else {
            if (numericAttributes["fabric_wrinkled"] ?? 0) > 0 {
                hasActiveFlaw = true
                activeFlaws.append("Visible fabric wrinkles")
                badPoints.append("Visible wrinkles or creases in garments.")
            } else {
                goodPoints.append("Crisp, wrinkle-free garment presentation.")
            }

            if (numericAttributes["styling_sloppy"] ?? 0) > 0 {
                hasActiveFlaw = true
                activeFlaws.append("Sloppy garment alignment / unkempt tuck")
                badPoints.append("Garment styling or silhouette lacks structure.")
            }

            if (numericAttributes["footwear_worn"] ?? 0) > 0 {
                hasActiveFlaw = true
                activeFlaws.append("Scuffed or worn footwear")
                badPoints.append("Footwear appears worn or scuffed.")
            } else {
                goodPoints.append("Clean, well-maintained footwear.")
            }
        }

        // Apply 7.2 clamp rule if active flaws exist
        var calibratedScore = max(1.0, min(10.0, rawScore))
        if hasActiveFlaw && calibratedScore >= 7.5 {
            calibratedScore = 7.2
        }

        // Determine tier
        let tier: String
        let tierLabel: String
        if calibratedScore >= 7.5 {
            tier = "3_Polished"
            tierLabel = "Polished & Sharp"
        } else if calibratedScore >= 5.0 {
            tier = "2_Average"
            tierLabel = "Average / Baseline Everyday"
        } else {
            tier = "1_NeedsImprovement"
            tierLabel = "Needs Improvement / Poor Execution"
        }

        return LookMaxVisionResult(
            score: round(calibratedScore * 10) / 10.0,
            rawScore: round(rawScore * 10) / 10.0,
            tier: tier,
            tierLabel: tierLabel,
            category: category,
            attributes: attributes,
            numericAttributes: numericAttributes,
            activeFlaws: activeFlaws,
            goodPoints: goodPoints,
            badPoints: badPoints
        )
    }

    // MARK: - Preprocessing Helpers

    private func prepareImageForInference(image: UIImage, isGrooming: Bool, targetSize: CGSize) -> CGImage? {
        guard let cg = image.cgImage else { return nil }

        if isGrooming {
            // Face-crop with +35% margin for maximum facial feature density
            let faceReq = VNDetectFaceRectanglesRequest()
            let handler = VNImageRequestHandler(cgImage: cg, options: [:])
            try? handler.perform([faceReq])

            if let face = faceReq.results?.first {
                let imgW = CGFloat(cg.width)
                let imgH = CGFloat(cg.height)
                let bbox = face.boundingBox

                // Convert Vision to Image coordinates
                let fx = bbox.origin.x * imgW
                let fy = (1.0 - bbox.origin.y - bbox.height) * imgH
                let fw = bbox.width * imgW
                let fh = bbox.height * imgH

                // Expand by 35% margin
                let side = max(fw, fh) * 1.35
                let cx = fx + fw * 0.5
                let cy = fy + fh * 0.5

                let cropRect = CGRect(
                    x: max(0, cx - side * 0.5),
                    y: max(0, cy - side * 0.5),
                    width: min(imgW, side),
                    height: min(imgH, side)
                )

                if let cropped = cg.cropping(to: cropRect) {
                    return resizeCGImage(cropped, to: targetSize)
                }
            }
        }

        // Outfit: preserve 3:4 aspect ratio
        return resizeCGImage(cg, to: targetSize)
    }

    private func resizeCGImage(_ image: CGImage, to targetSize: CGSize) -> CGImage? {
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }

        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()
    }

    private func argmax(_ multiArray: MLMultiArray) -> Int {
        var maxVal: Float = -Float.greatestFiniteMagnitude
        var maxIdx = 0
        let count = multiArray.count
        for i in 0..<count {
            let val = multiArray[i].floatValue
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }
        return maxIdx
    }
}
