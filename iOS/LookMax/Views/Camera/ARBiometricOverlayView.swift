import SwiftUI

struct ARBiometricOverlayView: View {
    @ObservedObject var camera: CameraController
    let occasion: OccasionCategory

    var body: some View {
        GeometryReader { geo in
            let size = geo.size

            ZStack {
                // Posture Spine Line
                if let start = camera.spineStartPoint, let end = camera.spineEndPoint {
                    let p1 = CGPoint(x: start.x * size.width, y: start.y * size.height)
                    let p2 = CGPoint(x: end.x * size.width, y: end.y * size.height)
                    let lineColor = camera.isPostureAligned ? Theme.neonCyan : Theme.warmAmber

                    Path { path in
                        path.move(to: p1)
                        path.addLine(to: p2)
                    }
                    .stroke(
                        LinearGradient(
                            colors: [lineColor.opacity(0.9), lineColor.opacity(0.4)],
                            startPoint: .top,
                            endPoint: .bottom
                        ),
                        style: StrokeStyle(lineWidth: 3.5, lineCap: .round, dash: [8, 4])
                    )
                    .shadow(color: lineColor.opacity(0.8), radius: 8)

                    // Spine Nodes
                    Circle()
                        .fill(lineColor)
                        .frame(width: 12, height: 12)
                        .shadow(color: lineColor, radius: 6)
                        .position(p1)

                    Circle()
                        .fill(lineColor.opacity(0.8))
                        .frame(width: 10, height: 10)
                        .shadow(color: lineColor, radius: 4)
                        .position(p2)
                }

                // Face Landmarks Contour
                if !camera.faceLandmarkPoints.isEmpty {
                    Path { path in
                        for (i, pt) in camera.faceLandmarkPoints.enumerated() {
                            let mapped = CGPoint(x: pt.x * size.width, y: pt.y * size.height)
                            if i == 0 { path.move(to: mapped) }
                            else { path.addLine(to: mapped) }
                        }
                    }
                    .stroke(Theme.neonCyan.opacity(0.7), style: StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .shadow(color: Theme.neonCyan.opacity(0.6), radius: 4)
                }
            }
        }
        .allowsHitTesting(false)
    }
}
