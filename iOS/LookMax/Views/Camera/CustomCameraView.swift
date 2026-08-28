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

    var body: some View {
        ZStack {
            Theme.oledBlack.ignoresSafeArea()

            GeometryReader { geo in
                ZStack {
                    CameraPreviewView(camera: camera)
                        .onTapGesture { loc in camera.focus(at: loc, viewSize: geo.size) }
                        .gesture(MagnificationGesture()
                            .onChanged { camera.setZoom(baseZoomFactor * $0) }
                            .onEnded { _ in baseZoomFactor = camera.currentZoomFactor }
                        )

                    // AR Biometric Real-time Overlay
                    ARBiometricOverlayView(camera: camera, occasion: occasion)

                    // Optional Grid
                    if camera.isGridVisible { CameraGridView() }

                    // Focus Indicator Box
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

            // Contextual Top HUD Pill
            VStack {
                HStack(spacing: 8) {
                    Image(systemName: "figure.stand")
                        .foregroundColor(camera.isPostureAligned ? Theme.neonCyan : Theme.warmAmber)
                        .font(.subheadline)

                    Text("Align posture for \(occasion.rawValue)")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(
                    Capsule()
                        .fill(Theme.cardDark.opacity(0.85))
                        .background(Capsule().fill(.ultraThinMaterial))
                )
                .overlay(
                    Capsule()
                        .stroke(camera.isPostureAligned ? Theme.neonCyan.opacity(0.4) : Theme.warmAmber.opacity(0.5), lineWidth: 1)
                )
                .shadow(color: (camera.isPostureAligned ? Theme.neonCyan : Theme.warmAmber).opacity(0.3), radius: 8)
                .padding(.top, 56)

                Spacer()
            }

            // Countdown Overlay
            if camera.isCountdownActive {
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

            // Bottom Controls & Top Nav
            VStack {
                // Top Bar Controls
                HStack(spacing: 16) {
                    Button(action: { camera.toggleFlash() }) {
                        Image(systemName: flashIcon)
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(camera.flashMode == .off ? .white : Theme.warmAmber)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                    }

                    Menu {
                        Button("Timer Off")   { camera.selectedTimerDuration = 0 }
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
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(camera.selectedTimerDuration > 0 ? Theme.neonCyan : .white)
                        .padding(.horizontal, camera.selectedTimerDuration > 0 ? 12 : 10)
                        .frame(height: 44)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())
                    }

                    Button(action: { camera.isGridVisible.toggle() }) {
                        Image(systemName: camera.isGridVisible ? "grid.circle.fill" : "grid")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(camera.isGridVisible ? Theme.neonCyan : .white)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                    }

                    Spacer()

                    Button(action: { camera.stopCamera(); isPresented = false }) {
                        Image(systemName: "xmark")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 54)

                Spacer()

                // Floating Native iOS Zoom Rings (.5x, 1x, 2x)
                HStack(spacing: 14) {
                    zoomButton(title: ".5x", factor: 0.5, isAvailable: camera.isUltraWideAvailable)
                    zoomButton(title: "1x", factor: 1.0, isAvailable: true)
                    zoomButton(title: "2x", factor: 2.0, isAvailable: camera.isTelephotoAvailable)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
                .background(Capsule().fill(Color.black.opacity(0.6)).background(.ultraThinMaterial))
                .padding(.bottom, 20)

                // Shutter Bar
                HStack(alignment: .center) {
                    Button(action: { showingLibraryPicker = true }) {
                        Image(systemName: "photo.on.rectangle")
                            .font(.system(size: 22))
                            .foregroundColor(.white)
                            .frame(width: 52, height: 52)
                            .background(Circle().fill(Color.white.opacity(0.15)))
                    }

                    Spacer()

                    // Neon Haptic Shutter Button
                    Button(action: { camera.triggerCapture() }) {
                        ZStack {
                            Circle()
                                .stroke(LinearGradient(colors: [Theme.neonCyan, Theme.electricBlue], startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 4)
                                .frame(width: 80, height: 80)
                                .shadow(color: Theme.neonCyan.opacity(0.6), radius: 10)

                            Circle()
                                .fill(camera.isCountdownActive ? Theme.warmAmber : Color.white)
                                .frame(width: 64, height: 64)
                                .scaleEffect(camera.isCountdownActive ? 0.85 : 1.0)
                                .animation(.spring(response: 0.25, dampingFraction: 0.6), value: camera.isCountdownActive)
                        }
                    }

                    Spacer()

                    Button(action: { camera.switchCamera() }) {
                        Image(systemName: "camera.rotate.fill")
                            .font(.system(size: 22))
                            .foregroundColor(.white)
                            .frame(width: 52, height: 52)
                            .background(Circle().fill(Color.white.opacity(0.15)))
                    }
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 44)
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

    @ViewBuilder
    private func zoomButton(title: String, factor: CGFloat, isAvailable: Bool) -> some View {
        let isSelected = abs(camera.currentZoomFactor - factor) < 0.2
        Button(action: {
            camera.setZoom(factor)
            baseZoomFactor = factor
            HapticManager.light()
        }) {
            Text(title)
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundColor(isSelected ? Theme.neonCyan : .white.opacity(0.8))
                .frame(width: 36, height: 36)
                .background(Circle().fill(isSelected ? Theme.neonCyan.opacity(0.2) : Color.clear))
        }
        .disabled(!isAvailable)
        .opacity(isAvailable ? 1.0 : 0.4)
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
