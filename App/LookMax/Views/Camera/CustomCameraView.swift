import SwiftUI
import AVFoundation

struct CustomCameraView: View {
    @Binding var isPresented: Bool
    var onPhotoCaptured: (UIImage) -> Void

    @StateObject private var camera = CameraController()
    @State private var showingLibraryPicker = false
    @State private var pickerImage: UIImage?
    @State private var baseZoomFactor: CGFloat = 1.0

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            GeometryReader { geo in
                ZStack {
                    CameraPreviewView(camera: camera)
                        .onTapGesture { loc in camera.focus(at: loc, viewSize: geo.size) }
                        .gesture(MagnificationGesture()
                            .onChanged { camera.setZoom(baseZoomFactor * $0) }
                            .onEnded { _ in baseZoomFactor = camera.currentZoomFactor }
                        )
                    if camera.isGridVisible { CameraGridView() }
                    if camera.isFocusing, let pt = camera.focusPoint {
                        Rectangle().stroke(Color.yellow, lineWidth: 1.5)
                            .frame(width: 70, height: 70).position(pt)
                            .animation(.easeInOut(duration: 0.2), value: camera.isFocusing)
                    }
                }
            }.ignoresSafeArea()

            // Countdown Overlay
            if camera.isCountdownActive {
                ZStack {
                    Color.black.opacity(0.4).ignoresSafeArea()
                    VStack(spacing: 24) {
                        Text("\(camera.countdownRemaining)")
                            .font(.system(size: 110, weight: .bold, design: .rounded))
                            .foregroundColor(.white).shadow(radius: 12)
                            .scaleEffect(camera.countdownRemaining > 0 ? 1.1 : 0.8)
                            .animation(.easeInOut(duration: 0.3), value: camera.countdownRemaining)
                        Button(action: { camera.cancelCountdown() }) {
                            Text("Cancel Timer").font(.subheadline.bold()).foregroundColor(.white)
                                .padding(.horizontal, 20).padding(.vertical, 10)
                                .background(Capsule().fill(Color.red.opacity(0.85)))
                        }
                    }
                }
            }

            VStack {
                // Top Controls
                HStack(spacing: 20) {
                    Button(action: { camera.toggleFlash() }) {
                        Image(systemName: flashIcon)
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(camera.flashMode == .off ? .white : .yellow)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial).clipShape(Circle())
                    }
                    Menu {
                        Button("Timer Off")    { camera.selectedTimerDuration = 0 }
                        Button("3 Seconds")   { camera.selectedTimerDuration = 3 }
                        Button("5 Seconds")   { camera.selectedTimerDuration = 5 }
                        Button("10 Seconds")  { camera.selectedTimerDuration = 10 }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "timer")
                            if camera.selectedTimerDuration > 0 {
                                Text("\(camera.selectedTimerDuration)s").font(.caption.bold())
                            }
                        }
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(camera.selectedTimerDuration > 0 ? .yellow : .white)
                        .padding(.horizontal, camera.selectedTimerDuration > 0 ? 12 : 10)
                        .frame(height: 44)
                        .background(.ultraThinMaterial).clipShape(Capsule())
                    }
                    Button(action: { camera.isGridVisible.toggle() }) {
                        Image(systemName: camera.isGridVisible ? "grid.circle.fill" : "grid")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(camera.isGridVisible ? .yellow : .white)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial).clipShape(Circle())
                    }
                    Spacer()
                    Button(action: { camera.stopCamera(); isPresented = false }) {
                        Image(systemName: "xmark")
                            .font(.system(size: 18, weight: .bold)).foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial).clipShape(Circle())
                    }
                }
                .padding(.horizontal, 20).padding(.top, 50)

                Spacer()

                // Zoom Buttons
                if camera.isUltraWideAvailable || camera.isTelephotoAvailable {
                    HStack(spacing: 12) {
                        if camera.isUltraWideAvailable {
                            Button(action: { let t = camera.minZoomFactor; camera.setZoom(t); baseZoomFactor = t }) {
                                Text(".5x").font(.caption.bold())
                                    .foregroundColor(camera.currentZoomFactor < 0.9 ? .yellow : .white)
                                    .frame(width: 38, height: 38)
                                    .background(Color.black.opacity(0.65)).clipShape(Circle())
                            }
                        }
                        Button(action: { camera.setZoom(1.0); baseZoomFactor = 1.0 }) {
                            Text("1x").font(.caption.bold())
                                .foregroundColor(abs(camera.currentZoomFactor - 1.0) < 0.2 ? .yellow : .white)
                                .frame(width: 38, height: 38)
                                .background(Color.black.opacity(0.65)).clipShape(Circle())
                        }
                        if camera.isTelephotoAvailable {
                            Button(action: { camera.setZoom(2.0); baseZoomFactor = 2.0 }) {
                                Text("2x").font(.caption.bold())
                                    .foregroundColor(camera.currentZoomFactor >= 1.8 ? .yellow : .white)
                                    .frame(width: 38, height: 38)
                                    .background(Color.black.opacity(0.65)).clipShape(Circle())
                            }
                        }
                    }.padding(.bottom, 16)
                }

                // Shutter Row
                HStack(alignment: .center) {
                    Button(action: { showingLibraryPicker = true }) {
                        Image(systemName: "photo.on.rectangle")
                            .font(.system(size: 24)).foregroundColor(.white)
                            .frame(width: 54, height: 54)
                            .background(Color.white.opacity(0.2)).clipShape(Circle())
                    }
                    Spacer()
                    Button(action: { camera.triggerCapture() }) {
                        ZStack {
                            Circle().strokeBorder(Color.white, lineWidth: 4).frame(width: 78, height: 78)
                            Circle()
                                .fill(camera.isCountdownActive ? Color.yellow : Color.white)
                                .frame(width: 64, height: 64)
                                .scaleEffect(camera.isCountdownActive ? 0.85 : 1.0)
                                .animation(.spring(), value: camera.isCountdownActive)
                        }
                    }
                    Spacer()
                    Button(action: { camera.switchCamera() }) {
                        Image(systemName: "camera.rotate.fill")
                            .font(.system(size: 24)).foregroundColor(.white)
                            .frame(width: 54, height: 54)
                            .background(Color.white.opacity(0.2)).clipShape(Circle())
                    }
                }
                .padding(.horizontal, 30).padding(.bottom, 40)
            }
        }
        .onAppear {
            camera.onPhotoCaptured = { photo in
                camera.stopCamera()
                isPresented = false
                onPhotoCaptured(photo)
            }
            camera.setupCamera()
        }
        .onDisappear { camera.stopCamera() }
        .sheet(isPresented: $showingLibraryPicker) {
            ImagePicker(image: $pickerImage, sourceType: .photoLibrary)
                .onDisappear {
                    if let img = pickerImage {
                        camera.stopCamera()
                        isPresented = false
                        onPhotoCaptured(img)
                    }
                }
        }
    }

    private var flashIcon: String {
        switch camera.flashMode {
        case .auto:     return "bolt.badge.a.fill"
        case .on:       return "bolt.fill"
        case .off:      return "bolt.slash.fill"
        @unknown default: return "bolt.badge.a.fill"
        }
    }
}
