import SwiftUI
import AVFoundation

struct CustomCameraView: View {
    @Binding var isPresented: Bool
    var occasion: OccasionCategory = .casualEveryday
    var onPhotoCaptured: (UIImage) -> Void

    @StateObject private var camera = CameraController()
    @State private var showingLibraryPicker = false
    @State private var pickerImage: UIImage?
    @State private var baseZoomFactor: CGFloat = 1.0
    @State private var lastCapturedThumb: UIImage?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            GeometryReader { geo in
                ZStack {
                    // Camera preview full screen
                    CameraPreviewView(camera: camera)
                        .onTapGesture { loc in camera.focus(at: loc, viewSize: geo.size) }
                        .gesture(MagnificationGesture()
                            .onChanged { camera.setZoom(baseZoomFactor * $0) }
                            .onEnded { _ in baseZoomFactor = camera.currentZoomFactor }
                        )

                    // Real-time AR overlay
                    ARBiometricOverlayView(camera: camera, occasion: occasion)

                    // Grid overlay
                    if camera.isGridVisible { CameraGridView() }

                    // Tap-to-focus ring
                    if camera.isFocusing, let pt = camera.focusPoint {
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Theme.neonCyan, lineWidth: 2)
                            .frame(width: 68, height: 68)
                            .position(pt)
                            .neonGlow(color: Theme.neonCyan, radius: 8)
                            .animation(.easeInOut(duration: 0.2), value: camera.isFocusing)
                    }
                }
            }
            .ignoresSafeArea()

            // ─── UI Overlay Layer ───
            VStack {
                // ── Top Bar ──
                topBar
                    .padding(.top, 52)

                // ── Contextual HUD Prompt ──
                hudPrompt
                    .padding(.top, 12)

                Spacer()

                // ── Countdown overlay ──
                if camera.isCountdownActive {
                    countdownView
                }

                // ── Native Zoom Rings ──
                zoomRings
                    .padding(.bottom, 20)

                // ── Bottom Shutter Bar ──
                shutterBar
                    .padding(.horizontal, 24)
                    .padding(.bottom, 48)
            }
        }
        .onAppear {
            camera.onPhotoCaptured = { photo in
                lastCapturedThumb = photo
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

    // MARK: - Top Bar: flash | timer | grid | Spacer | ⚙️ gear | ✕
    private var topBar: some View {
        HStack(spacing: 12) {
            Button(action: { camera.toggleFlash() }) {
                Image(systemName: flashIcon)
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(camera.flashMode == .off ? .white : Theme.warmAmber)
                    .frame(width: 42, height: 42)
                    .background(.ultraThinMaterial)
                    .clipShape(Circle())
            }

            Menu {
                Button("Timer Off")  { camera.selectedTimerDuration = 0 }
                Button("3 Seconds")  { camera.selectedTimerDuration = 3 }
                Button("5 Seconds")  { camera.selectedTimerDuration = 5 }
                Button("10 Seconds") { camera.selectedTimerDuration = 10 }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "timer")
                    if camera.selectedTimerDuration > 0 {
                        Text("\(camera.selectedTimerDuration)s").font(.caption.bold())
                    }
                }
                .font(.system(size: 16, weight: .semibold))
                .foregroundColor(camera.selectedTimerDuration > 0 ? Theme.neonCyan : .white)
                .padding(.horizontal, camera.selectedTimerDuration > 0 ? 12 : 10)
                .frame(height: 42)
                .background(.ultraThinMaterial)
                .clipShape(Capsule())
            }

            Button(action: { camera.isGridVisible.toggle() }) {
                Image(systemName: camera.isGridVisible ? "grid.circle.fill" : "grid")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(camera.isGridVisible ? Theme.neonCyan : .white)
                    .frame(width: 42, height: 42)
                    .background(.ultraThinMaterial)
                    .clipShape(Circle())
            }

            Spacer()

            // Gear icon (top-right, per design)
            Button(action: {}) {
                Image(systemName: "gearshape")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(width: 42, height: 42)
                    .background(.ultraThinMaterial)
                    .clipShape(Circle())
            }

            Button(action: { camera.stopCamera(); isPresented = false }) {
                Image(systemName: "xmark")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.white)
                    .frame(width: 42, height: 42)
                    .background(.ultraThinMaterial)
                    .clipShape(Circle())
            }
        }
        .padding(.horizontal, 20)
    }

    // MARK: - Contextual HUD Prompt — matches design: cyan text, frosted pill
    private var hudPrompt: some View {
        HStack(spacing: 8) {
            Image(systemName: "figure.stand")
                .foregroundColor(Theme.neonCyan)
                .font(.system(size: 16, weight: .semibold))

            Text("Align posture for \(occasion.rawValue)")
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundColor(Theme.neonCyan)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 10)
        .background(
            Capsule()
                .fill(Color.black.opacity(0.55))
                .background(Capsule().fill(.ultraThinMaterial))
        )
        .overlay(
            Capsule()
                .stroke(Theme.neonCyan.opacity(0.55), lineWidth: 1.5)
        )
        .shadow(color: Theme.neonCyan.opacity(0.3), radius: 8)
    }

    // MARK: - Countdown overlay
    private var countdownView: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()
            VStack(spacing: 24) {
                Text("\(camera.countdownRemaining)")
                    .font(.system(size: 110, weight: .heavy, design: .rounded))
                    .foregroundColor(Theme.neonCyan)
                    .neonGlow(color: Theme.neonCyan, radius: 20)
                    .scaleEffect(camera.countdownRemaining > 0 ? 1.1 : 0.8)
                    .animation(.spring(response: 0.3, dampingFraction: 0.6), value: camera.countdownRemaining)

                Button(action: { camera.cancelCountdown() }) {
                    Text("Cancel Timer")
                        .font(.subheadline.bold())
                        .foregroundColor(.white)
                        .padding(.horizontal, 22)
                        .padding(.vertical, 10)
                        .background(Capsule().fill(Theme.crimson.opacity(0.9)))
                }
            }
        }
    }

    // MARK: - Zoom rings (.5x, 1x, 2x) — native iOS camera style
    private var zoomRings: some View {
        HStack(spacing: 24) {
            zoomButton(title: ".5x", factor: 0.5, isAvailable: camera.isUltraWideAvailable)
            zoomButton(title: "1x", factor: 1.0, isAvailable: true)
            zoomButton(title: "2x", factor: 2.0, isAvailable: camera.isTelephotoAvailable)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .background(Capsule().fill(Color.black.opacity(0.5)).background(.ultraThinMaterial))
    }

    @ViewBuilder
    private func zoomButton(title: String, factor: CGFloat, isAvailable: Bool) -> some View {
        let isSelected = abs(camera.currentZoomFactor - factor) < 0.25
        Button(action: {
            camera.setZoom(factor)
            baseZoomFactor = factor
            HapticManager.light()
        }) {
            Text(title)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundColor(isSelected ? Theme.neonCyan : .white.opacity(0.9))
                .frame(width: 40, height: 40)
                .background(
                    Circle()
                        .stroke(isSelected ? Theme.neonCyan : Color.clear, lineWidth: 2)
                        .background(
                            Circle().fill(isSelected ? Theme.neonCyan.opacity(0.15) : Color.clear)
                        )
                )
        }
        .disabled(!isAvailable)
        .opacity(isAvailable ? 1.0 : 0.35)
    }

    // MARK: - Shutter Bar: thumbnail (bottom-left) | large shutter | camera rotate (bottom-right)
    private var shutterBar: some View {
        HStack(alignment: .center) {
            // Last captured thumbnail (bottom-left corner per design)
            Button(action: { showingLibraryPicker = true }) {
                if let thumb = lastCapturedThumb {
                    Image(uiImage: thumb)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 52, height: 52)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .overlay(RoundedRectangle(cornerRadius: 10).stroke(.white, lineWidth: 1.5))
                } else {
                    Image(systemName: "photo.on.rectangle")
                        .font(.system(size: 22))
                        .foregroundColor(.white)
                        .frame(width: 52, height: 52)
                        .background(Circle().fill(Color.white.opacity(0.15)))
                }
            }

            Spacer()

            // Large white shutter + thick Neon Cyan ring (exact design match)
            Button(action: { camera.triggerCapture() }) {
                ZStack {
                    // Outer thick neon ring
                    Circle()
                        .stroke(Theme.neonCyan, lineWidth: 5)
                        .frame(width: 84, height: 84)
                        .shadow(color: Theme.neonCyan.opacity(0.7), radius: 10)

                    // Solid white inner circle
                    Circle()
                        .fill(camera.isCountdownActive ? Theme.warmAmber : Color.white)
                        .frame(width: 68, height: 68)
                        .scaleEffect(camera.isCountdownActive ? 0.85 : 1.0)
                        .animation(.spring(response: 0.25, dampingFraction: 0.6), value: camera.isCountdownActive)
                }
            }

            Spacer()

            // Camera rotation icon (bottom-right per design)
            Button(action: { camera.switchCamera() }) {
                Image(systemName: "camera.on.rectangle")
                    .font(.system(size: 22))
                    .foregroundColor(.white)
                    .frame(width: 52, height: 52)
                    .background(Circle().fill(Color.white.opacity(0.15)))
            }
        }
    }

    private var flashIcon: String {
        switch camera.flashMode {
        case .auto:  return "bolt.badge.a.fill"
        case .on:    return "bolt.fill"
        case .off:   return "bolt.slash.fill"
        @unknown default: return "bolt.badge.a.fill"
        }
    }
}
