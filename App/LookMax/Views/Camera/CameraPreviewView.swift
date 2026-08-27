import SwiftUI
import AVFoundation

struct CameraGridView: View {
    var body: some View {
        GeometryReader { geo in
            Path { path in
                let (w, h) = (geo.size.width, geo.size.height)
                path.move(to: CGPoint(x: w/3, y: 0));   path.addLine(to: CGPoint(x: w/3, y: h))
                path.move(to: CGPoint(x: 2*w/3, y: 0)); path.addLine(to: CGPoint(x: 2*w/3, y: h))
                path.move(to: CGPoint(x: 0, y: h/3));   path.addLine(to: CGPoint(x: w, y: h/3))
                path.move(to: CGPoint(x: 0, y: 2*h/3)); path.addLine(to: CGPoint(x: w, y: 2*h/3))
            }
            .stroke(Color.white.opacity(0.3), lineWidth: 1)
        }
        .allowsHitTesting(false)
    }
}

struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var camera: CameraController

    func makeUIView(context: Context) -> CameraPreviewUIView {
        let v = CameraPreviewUIView()
        v.previewLayer.session = camera.session
        v.previewLayer.videoGravity = .resizeAspectFill
        return v
    }

    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {}
}

class CameraPreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}
