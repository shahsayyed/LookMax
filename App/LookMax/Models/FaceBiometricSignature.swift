import Foundation

struct FaceBiometricSignature: Codable, Equatable {
    let eyeToNoseRatio: Double
    let eyeToMouthRatio: Double
    let noseToChinRatio: Double
    let mouthWidthRatio: Double
    let jawWidthRatio: Double
    let faceAspectRatio: Double

    func similarity(to other: FaceBiometricSignature) -> Double {
        let diffs = [
            abs(eyeToNoseRatio  - other.eyeToNoseRatio)  * 1.4,
            abs(eyeToMouthRatio - other.eyeToMouthRatio) * 1.4,
            abs(noseToChinRatio - other.noseToChinRatio) * 1.1,
            abs(mouthWidthRatio - other.mouthWidthRatio) * 1.0,
            abs(jawWidthRatio   - other.jawWidthRatio)   * 1.0,
            abs(faceAspectRatio - other.faceAspectRatio) * 1.1
        ]
        let avgDiff = diffs.reduce(0, +) / Double(diffs.count)
        return max(0.0, min(1.0, 1.0 - (avgDiff * 2.5)))
    }
}
