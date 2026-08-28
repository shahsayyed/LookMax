import SwiftUI

struct BeforeAfterComparisonView: View {
    let look1: LookItem
    let look2: LookItem
    var occasion: OccasionCategory = .casualEveryday
    @Environment(\.presentationMode) var presentationMode

    @State private var splitFraction: CGFloat = 0.5
    @State private var isDragging: Bool = false
    @State private var lastHapticStep: Int = 5

    private var scoreDiff: Double {
        look2.score - look1.score
    }

    private var postureDiff: Double {
        look2.postureScore - look1.postureScore
    }

    var body: some View {
        NavigationView {
            ZStack {
                Theme.oledBlack.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 20) {
                        // Gamified Score Jump Badge
                        scoreJumpHeader

                        // Interactive Split Curtain View
                        interactiveCurtain
                            .frame(height: 380)
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                            .overlay(
                                RoundedRectangle(cornerRadius: 20)
                                    .stroke(Theme.cardBorder, lineWidth: 1)
                            )
                            .padding(.horizontal, 16)

                        // Side-by-Side Micro Metrics
                        microMetricsSection
                            .padding(.horizontal, 16)

                        // Performance Insights Bars
                        performanceInsightsCard
                            .padding(.horizontal, 16)
                            .padding(.bottom, 32)
                    }
                    .padding(.top, 8)
                }
            }
            .navigationTitle("Outfit Comparison")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { presentationMode.wrappedValue.dismiss() }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                            .font(.title3)
                    }
                }
            }
        }
    }

    // MARK: - Gamified Score Jump Badge
    private var scoreJumpHeader: some View {
        VStack(spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: scoreDiff >= 0 ? "arrow.up.right.circle.fill" : "arrow.down.right.circle.fill")
                    .foregroundColor(scoreDiff >= 0 ? Theme.emerald : Theme.crimson)
                    .font(.title3)

                Text(String(format: "%@%.1f Score Jump!", scoreDiff >= 0 ? "+" : "", scoreDiff))
                    .font(.system(size: 18, weight: .bold, design: .rounded))
                    .foregroundColor(scoreDiff >= 0 ? Theme.emerald : Theme.crimson)
            }
            .neonGlow(color: scoreDiff >= 0 ? Theme.emerald : Theme.crimson, radius: 10)

            Text(scoreDiff >= 0 ? "Better posture detected! Your outfit looks sharper." : "Try straightening posture and adjusting collar.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(
            Capsule()
                .fill(Theme.cardDark)
                .background(Capsule().fill(.ultraThinMaterial))
        )
        .overlay(
            Capsule()
                .stroke((scoreDiff >= 0 ? Theme.emerald : Theme.warmAmber).opacity(0.4), lineWidth: 1)
        )
    }

    // MARK: - Interactive Split Curtain
    private var interactiveCurtain: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let dividerX = max(20, min(w - 20, splitFraction * w))

            ZStack(alignment: .leading) {
                // Right Image (Look 2 / After)
                if let img2 = look2.image {
                    Image(uiImage: img2)
                        .resizable()
                        .scaledToFill()
                        .frame(width: w, height: h)
                        .clipped()
                } else {
                    Rectangle().fill(Theme.surfaceDark)
                }

                // Left Image (Look 1 / Before) clipped
                if let img1 = look1.image {
                    Image(uiImage: img1)
                        .resizable()
                        .scaledToFill()
                        .frame(width: w, height: h)
                        .clipped()
                        .mask(
                            HStack(spacing: 0) {
                                Rectangle().frame(width: dividerX)
                                Spacer()
                            }
                        )
                }

                // Header Tags (LOOK 1 vs LOOK 2)
                VStack {
                    HStack {
                        Text("LOOK 1")
                            .font(.system(size: 11, weight: .black, design: .rounded))
                            .foregroundColor(.white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.7))
                            .clipShape(Capsule())
                            .padding(12)

                        Spacer()

                        Text("LOOK 2")
                            .font(.system(size: 11, weight: .black, design: .rounded))
                            .foregroundColor(Theme.neonCyan)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.7))
                            .clipShape(Capsule())
                            .padding(12)
                    }
                    Spacer()
                }

                // Divider Line
                Rectangle()
                    .fill(
                        LinearGradient(
                            colors: [Theme.neonCyan, Theme.electricBlue],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(width: 3)
                    .shadow(color: Theme.neonCyan, radius: 6)
                    .offset(x: dividerX - 1.5)

                // Drag Handle Knob
                ZStack {
                    Circle()
                        .fill(Theme.cardDark)
                        .frame(width: 42, height: 42)
                        .overlay(Circle().stroke(Theme.neonCyan, lineWidth: 2))
                        .shadow(color: Theme.neonCyan.opacity(0.8), radius: 8)

                    Image(systemName: "chevron.left.and.chevron.right")
                        .font(.system(size: 13, weight: .heavy))
                        .foregroundColor(Theme.neonCyan)
                }
                .position(x: dividerX, y: h / 2)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture()
                    .onChanged { val in
                        isDragging = true
                        let fraction = max(0.05, min(0.95, val.location.x / w))
                        splitFraction = fraction
                        
                        let step = Int(fraction * 10)
                        if step != lastHapticStep {
                            lastHapticStep = step
                            HapticManager.light()
                        }
                    }
                    .onEnded { _ in
                        isDragging = false
                    }
            )
        }
    }

    // MARK: - Side-by-Side Micro Metrics
    private var microMetricsSection: some View {
        HStack(alignment: .top, spacing: 12) {
            // Look 1 Column
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("LOOK 1")
                        .font(.caption.bold())
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.1f", look1.score))
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                        .foregroundColor(Theme.scoreColor(look1.score))
                }

                Text(String(format: "Posture: %.1f • Style: %.1f", look1.postureScore, look1.fitScore))
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Divider().background(Theme.cardBorderSubtle)

                VStack(alignment: .leading, spacing: 4) {
                    metricBullet(text: look1.postureNote, isPositive: look1.postureScore >= 7.5)
                    metricBullet(text: look1.fitNote, isPositive: look1.fitScore >= 7.5)
                    metricBullet(text: look1.styleNote, isPositive: look1.score >= 7.5)
                }
            }
            .padding(12)
            .glassCard(cornerRadius: 14)

            // Look 2 Column
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("LOOK 2 (LATEST)")
                        .font(.caption.bold())
                        .foregroundColor(Theme.neonCyan)
                    Spacer()
                    Text(String(format: "%.1f", look2.score))
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                        .foregroundColor(Theme.scoreColor(look2.score))
                }

                Text(String(format: "Posture: %.1f • Style: %.1f", look2.postureScore, look2.fitScore))
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Divider().background(Theme.cardBorderSubtle)

                VStack(alignment: .leading, spacing: 4) {
                    metricBullet(text: look2.postureNote, isPositive: look2.postureScore >= 7.5)
                    metricBullet(text: look2.fitNote, isPositive: look2.fitScore >= 7.5)
                    metricBullet(text: look2.styleNote, isPositive: look2.score >= 7.5)
                }
            }
            .padding(12)
            .glassCard(cornerRadius: 14, borderColor: Theme.neonCyan.opacity(0.3))
        }
    }

    private func metricBullet(text: String, isPositive: Bool) -> some View {
        HStack(alignment: .top, spacing: 4) {
            Image(systemName: isPositive ? "checkmark.circle.fill" : "circle.fill")
                .font(.system(size: 8))
                .foregroundColor(isPositive ? Theme.emerald : Theme.warmAmber)
                .padding(.top, 3)

            Text(text)
                .font(.caption2)
                .foregroundColor(.primary)
                .lineLimit(2)
        }
    }

    // MARK: - Performance Insights Card
    private var performanceInsightsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("PERFORMANCE INSIGHTS")
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundColor(.secondary)
                .tracking(0.8)

            insightBar(title: "Posture Alignment", val1: look1.postureScore, val2: look2.postureScore)
            insightBar(title: "Fit & Proportions", val1: look1.fitScore, val2: look2.fitScore)
            insightBar(title: "Overall Style Score", val1: look1.score, val2: look2.score)
        }
        .padding(16)
        .glassCard(cornerRadius: 16)
    }

    private func insightBar(title: String, val1: Double, val2: Double) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.subheadline)
                Spacer()
                HStack(spacing: 6) {
                    Text(String(format: "%.1f", val1))
                        .font(.caption.bold())
                        .foregroundColor(.secondary)
                    Image(systemName: "arrow.right")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.1f", val2))
                        .font(.caption.bold())
                        .foregroundColor(val2 >= val1 ? Theme.emerald : Theme.warmAmber)
                }
            }

            GeometryReader { geo in
                let w = geo.size.width
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.08)).frame(height: 8)

                    // Look 1 bar
                    Capsule()
                        .fill(Color.gray.opacity(0.4))
                        .frame(width: max(10, w * CGFloat(val1 / 10.0)), height: 8)

                    // Look 2 bar
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [Theme.neonCyan, Theme.emerald],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: max(10, w * CGFloat(val2 / 10.0)), height: 8)
                }
            }
            .frame(height: 8)
        }
    }
}
