import SwiftUI

enum Theme {
    // Backgrounds
    static let oledBlack = Color(red: 0.03, green: 0.03, blue: 0.04)
    static let cardDark = Color(red: 0.08, green: 0.08, blue: 0.10)
    static let surfaceDark = Color(red: 0.13, green: 0.13, blue: 0.15)
    static let cardBorder = Color.white.opacity(0.12)
    static let cardBorderSubtle = Color.white.opacity(0.06)

    // Accents & Signals
    static let neonCyan = Color(red: 0.0, green: 0.94, blue: 1.0)
    static let electricBlue = Color(red: 0.04, green: 0.52, blue: 1.0)
    static let emerald = Color(red: 0.20, green: 0.78, blue: 0.35)
    static let emeraldGlow = Color(red: 0.20, green: 0.78, blue: 0.35).opacity(0.4)
    static let warmAmber = Color(red: 1.0, green: 0.58, blue: 0.0)
    static let crimson = Color(red: 1.0, green: 0.23, blue: 0.19)
    static let purpleAccent = Color(red: 0.69, green: 0.32, blue: 0.87)

    // Gradients
    static let neonGradient = LinearGradient(
        colors: [neonCyan, electricBlue],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let cardGradient = LinearGradient(
        colors: [Color(red: 0.10, green: 0.10, blue: 0.12), Color(red: 0.06, green: 0.06, blue: 0.08)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let emeraldGradient = LinearGradient(
        colors: [Color(red: 0.20, green: 0.85, blue: 0.40), Color(red: 0.15, green: 0.65, blue: 0.30)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static func scoreColor(_ score: Double) -> Color {
        if score >= 8.5 { return emerald }
        if score >= 7.0 { return warmAmber }
        return crimson
    }
}

struct GlassCardModifier: ViewModifier {
    var cornerRadius: CGFloat = 16
    var borderColor: Color = Theme.cardBorder

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(Theme.cardDark.opacity(0.85))
                    .background(
                        RoundedRectangle(cornerRadius: cornerRadius)
                            .fill(.ultraThinMaterial)
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(borderColor, lineWidth: 1)
            )
    }
}

struct NeonGlowModifier: ViewModifier {
    var color: Color = Theme.neonCyan
    var radius: CGFloat = 8

    func body(content: Content) -> some View {
        content
            .shadow(color: color.opacity(0.5), radius: radius, x: 0, y: 0)
    }
}

extension View {
    func glassCard(cornerRadius: CGFloat = 16, borderColor: Color = Theme.cardBorder) -> some View {
        modifier(GlassCardModifier(cornerRadius: cornerRadius, borderColor: borderColor))
    }

    func neonGlow(color: Color = Theme.neonCyan, radius: CGFloat = 8) -> some View {
        modifier(NeonGlowModifier(color: color, radius: radius))
    }
}

enum HapticManager {
    static func light() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    static func medium() {
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
    }

    static func heavy() {
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
    }

    static func success() {
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }
}
