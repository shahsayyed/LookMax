import SwiftUI

struct AlignmentGuideOverlayView: View {
    @ObservedObject var camera: CameraController

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let isMatched = camera.isFramingMatched && camera.isLightingAdequate
            let accentColor = isMatched ? Theme.emerald : (camera.isLightingAdequate ? Theme.neonCyan : Theme.warmAmber)

            ZStack {
                // Semi-darkened vignette outside guide to focus user attention
                Color.black.opacity(0.18)
                    .mask(
                        cutoutMask(for: size)
                            .fill(style: FillStyle(eoFill: true))
                    )

                if camera.scanMode == .grooming {
                    groomingReticle(in: size, accentColor: accentColor, isMatched: isMatched)
                } else {
                    outfitReticle(in: size, accentColor: accentColor, isMatched: isMatched)
                }

                // Auto-Capture Lock-on Arc
                if camera.isAutoCaptureEnabled && camera.autoCaptureProgress > 0 {
                    autoCaptureRing(in: size, accentColor: accentColor)
                }
            }
        }
        .allowsHitTesting(false)
    }

    // MARK: - Grooming Oval Reticle
    @ViewBuilder
    private func groomingReticle(in size: CGSize, accentColor: Color, isMatched: Bool) -> some View {
        let ovalWidth = size.width * 0.65
        let ovalHeight = ovalWidth * 1.35
        let center = CGPoint(x: size.width * 0.5, y: size.height * 0.42)

        ZStack {
            // Main Head Oval
            Ellipse()
                .stroke(
                    accentColor,
                    style: StrokeStyle(lineWidth: isMatched ? 3.5 : 2.0, dash: isMatched ? [] : [8, 6])
                )
                .frame(width: ovalWidth, height: ovalHeight)
                .position(center)
                .neonGlow(color: accentColor, radius: isMatched ? 12 : 4)

            // Eye Level Line
            Path { path in
                let y = center.y - (ovalHeight * 0.08)
                path.move(to: CGPoint(x: center.x - (ovalWidth * 0.35), y: y))
                path.addLine(to: CGPoint(x: center.x + (ovalWidth * 0.35), y: y))
            }
            .stroke(accentColor.opacity(0.6), style: StrokeStyle(lineWidth: 1.5, dash: [4, 4]))

            // Chin Base Line
            Path { path in
                let y = center.y + (ovalHeight * 0.38)
                path.move(to: CGPoint(x: center.x - (ovalWidth * 0.20), y: y))
                path.addLine(to: CGPoint(x: center.x + (ovalWidth * 0.20), y: y))
            }
            .stroke(accentColor.opacity(0.6), style: StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
        }
    }

    // MARK: - Outfit Full-Body Silhouette Reticle
    @ViewBuilder
    private func outfitReticle(in size: CGSize, accentColor: Color, isMatched: Bool) -> some View {
        let bodyWidth = size.width * 0.72
        let bodyHeight = size.height * 0.76
        let center = CGPoint(x: size.width * 0.5, y: size.height * 0.48)

        ZStack {
            // Outer 3:4 Portrait Framing Box
            RoundedRectangle(cornerRadius: 24)
                .stroke(
                    accentColor,
                    style: StrokeStyle(lineWidth: isMatched ? 3.0 : 1.8, dash: isMatched ? [] : [10, 6])
                )
                .frame(width: bodyWidth, height: bodyHeight)
                .position(center)
                .neonGlow(color: accentColor, radius: isMatched ? 10 : 3)

            // Shoulder Level Marker
            Path { path in
                let y = center.y - (bodyHeight * 0.33)
                path.move(to: CGPoint(x: center.x - (bodyWidth * 0.40), y: y))
                path.addLine(to: CGPoint(x: center.x + (bodyWidth * 0.40), y: y))
            }
            .stroke(accentColor.opacity(0.5), style: StrokeStyle(lineWidth: 1.5, dash: [6, 6]))

            // Waist Level Marker
            Path { path in
                let y = center.y - (bodyHeight * 0.05)
                path.move(to: CGPoint(x: center.x - (bodyWidth * 0.30), y: y))
                path.addLine(to: CGPoint(x: center.x + (bodyWidth * 0.30), y: y))
            }
            .stroke(accentColor.opacity(0.5), style: StrokeStyle(lineWidth: 1.5, dash: [6, 6]))

            // Footwear Floor Baseline Marker
            Path { path in
                let y = center.y + (bodyHeight * 0.44)
                path.move(to: CGPoint(x: center.x - (bodyWidth * 0.35), y: y))
                path.addLine(to: CGPoint(x: center.x + (bodyWidth * 0.35), y: y))
            }
            .stroke(accentColor.opacity(0.8), style: StrokeStyle(lineWidth: 2.0))

            // Text Label at bottom of guide
            Text("ALIGN SHOES TO BASELINE")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(accentColor.opacity(0.9))
                .position(x: center.x, y: center.y + (bodyHeight * 0.47))
        }
    }

    // MARK: - Auto Capture Lock-on Ring
    @ViewBuilder
    private func autoCaptureRing(in size: CGSize, accentColor: Color) -> some View {
        let radius: CGFloat = 45
        let center = CGPoint(x: size.width * 0.5, y: size.height * 0.82)

        ZStack {
            Circle()
                .stroke(Color.white.opacity(0.2), lineWidth: 4)
                .frame(width: radius * 2, height: radius * 2)

            Circle()
                .trim(from: 0, to: camera.autoCaptureProgress)
                .stroke(
                    Theme.emerald,
                    style: StrokeStyle(lineWidth: 5, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .frame(width: radius * 2, height: radius * 2)
                .neonGlow(color: Theme.emerald, radius: 10)

            Image(systemName: "camera.shutter.button.fill")
                .font(.system(size: 24))
                .foregroundColor(Theme.emerald)
        }
        .position(center)
    }

    // MARK: - Cutout Mask
    private func cutoutMask(for size: CGSize) -> Path {
        var path = Path()
        path.addRect(CGRect(origin: .zero, size: size))

        if camera.scanMode == .grooming {
            let ovalWidth = size.width * 0.65
            let ovalHeight = ovalWidth * 1.35
            let center = CGPoint(x: size.width * 0.5, y: size.height * 0.42)
            let rect = CGRect(
                x: center.x - (ovalWidth * 0.5),
                y: center.y - (ovalHeight * 0.5),
                width: ovalWidth,
                height: ovalHeight
            )
            path.addEllipse(in: rect)
        } else {
            let bodyWidth = size.width * 0.72
            let bodyHeight = size.height * 0.76
            let center = CGPoint(x: size.width * 0.5, y: size.height * 0.48)
            let rect = CGRect(
                x: center.x - (bodyWidth * 0.5),
                y: center.y - (bodyHeight * 0.5),
                width: bodyWidth,
                height: bodyHeight
            )
            path.addRoundedRect(in: rect, cornerSize: CGSize(width: 24, height: 24))
        }
        return path
    }
}
