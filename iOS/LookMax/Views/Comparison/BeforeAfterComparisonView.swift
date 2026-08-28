import SwiftUI

struct BeforeAfterComparisonView: View {
    let look1: LookItem
    let look2: LookItem
    var occasion: OccasionCategory = .casualEveryday
    @Environment(\.presentationMode) var presentationMode

    @State private var splitFraction: CGFloat = 0.5
    @State private var isDragging: Bool = false
    @State private var lastHapticStep: Int = 5

    private var scoreDiff: Double { look2.score - look1.score }

    var body: some View {
        NavigationView {
            ZStack {
                Theme.oledBlack.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 20) {
                        interactiveCurtain
                            .frame(height: 340)
                            .clipShape(RoundedRectangle(cornerRadius: 18))
                            .overlay(RoundedRectangle(cornerRadius: 18).stroke(Theme.cardBorder, lineWidth: 1))
                            .padding(.horizontal, 16)

                        microMetricsSection
                            .padding(.horizontal, 16)

                        performanceInsightsCard
                            .padding(.horizontal, 16)
                            .padding(.bottom, 36)
                    }
                    .padding(.top, 8)
                }
            }
            .navigationTitle("Outfit Comparison")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: { presentationMode.wrappedValue.dismiss() }) {
                        Image(systemName: "chevron.left")
                            .foregroundColor(.white)
                            .font(.system(size: 16, weight: .semibold))
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {}) {
                        Image(systemName: "square.and.arrow.up")
                            .foregroundColor(.white)
                    }
                }
            }
        }
    }

    // MARK: - Interactive Split Curtain
    private var interactiveCurtain: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let dividerX = max(20, min(w - 20, splitFraction * w))

            ZStack(alignment: .leading) {
                // Right image (Look 2)
                if let img2 = look2.image {
                    Image(uiImage: img2)
                        .resizable()
                        .scaledToFill()
                        .frame(width: w, height: h)
                        .clipped()
                } else {
                    Rectangle().fill(Theme.surfaceDark)
                }

                // Left image (Look 1) clipped
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

                // Score jump badge overlay (centered at top)
                VStack {
                    scoreBadge
                        .padding(.top, 12)
                    Spacer()
                }
                .frame(width: w)

                // LOOK 1 / LOOK 2 labels
                VStack {
                    HStack {
                        Text("LOOK 1")
                            .font(.system(size: 11, weight: .black, design: .rounded))
                            .foregroundColor(.white)
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(Color.black.opacity(0.65))
                            .clipShape(Capsule())
                            .padding(10)

                        Spacer()

                        Text("LOOK 2")
                            .font(.system(size: 11, weight: .black, design: .rounded))
                            .foregroundColor(Theme.neonCyan)
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(Color.black.opacity(0.65))
                            .clipShape(Capsule())
                            .padding(10)
                    }
                    Spacer()
                }

                // Divider line
                Rectangle()
                    .fill(Color.white.opacity(0.7))
                    .frame(width: 2)
                    .shadow(color: .white.opacity(0.4), radius: 4)
                    .offset(x: dividerX - 1)

                // Drag handle
                ZStack {
                    Capsule()
                        .fill(Color(white: 0.25))
                        .frame(width: 32, height: 56)
                        .overlay(Capsule().stroke(Color.white.opacity(0.3), lineWidth: 1))

                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .heavy))
                        .foregroundColor(.white)
                        .offset(x: 3)
                }
                .position(x: dividerX, y: h / 2)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture()
                    .onChanged { val in
                        isDragging = true
                        let frac = max(0.05, min(0.95, val.location.x / w))
                        splitFraction = frac
                        let step = Int(frac * 10)
                        if step != lastHapticStep {
                            lastHapticStep = step
                            HapticManager.light()
                        }
                    }
                    .onEnded { _ in isDragging = false }
            )
        }
    }

    // Score jump badge shown inside the curtain
    private var scoreBadge: some View {
        VStack(spacing: 3) {
            HStack(spacing: 4) {
                Text(String(format: "%@%.1f Score Jump!", scoreDiff >= 0 ? "+" : "", scoreDiff))
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundColor(scoreDiff >= 0 ? Theme.emerald : Theme.crimson)

                Image(systemName: scoreDiff >= 0 ? "arrow.up" : "arrow.down")
                    .font(.system(size: 12, weight: .heavy))
                    .foregroundColor(scoreDiff >= 0 ? Theme.emerald : Theme.crimson)
            }

            Text(scoreDiff >= 0 ? "Better posture detected!\nYour outfit looks better." : "Try adjusting posture and collar.")
                .font(.system(size: 11))
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.black.opacity(0.78))
                .background(.ultraThinMaterial)
        )
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.2), lineWidth: 1))
    }

    // MARK: - Micro Metrics (side-by-side, matching design)
    private var microMetricsSection: some View {
        HStack(alignment: .top, spacing: 12) {
            lookMetricColumn(label: "LOOK 1", look: look1, accentColor: .white)
            lookMetricColumn(label: "LOOK 2", look: look2, accentColor: Theme.neonCyan)
        }
    }

    private func lookMetricColumn(label: String, look: LookItem, accentColor: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label)
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundColor(accentColor)

            // Sub-scores
            Text(String(format: "Posture: %.1f  |  Style: %.1f", look.postureScore, look.fitScore))
                .font(.system(size: 12))
                .foregroundColor(.secondary)

            // Star rating
            starRow(score: look.score)

            // Overall score label
            Text(String(format: "%.1f/10", look.score))
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundColor(.white)

            Divider().background(Theme.cardBorderSubtle)

            // Bullet points
            VStack(alignment: .leading, spacing: 4) {
                metricBullet(look.postureNote)
                metricBullet(look.fitNote)
                metricBullet(look.styleNote)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassCard(cornerRadius: 14, borderColor: accentColor == Theme.neonCyan ? Theme.neonCyan.opacity(0.3) : Theme.cardBorder)
    }

    private func starRow(score: Double) -> some View {
        HStack(spacing: 2) {
            ForEach(1...5, id: \.self) { i in
                Image(systemName: starIcon(for: i, score: score))
                    .font(.system(size: 12))
                    .foregroundColor(Theme.warmAmber)
            }
        }
    }

    private func starIcon(for position: Int, score: Double) -> String {
        let stars = (score / 10.0) * 5.0
        if Double(position) <= stars { return "star.fill" }
        if Double(position) - 0.5 <= stars { return "star.leadinghalf.filled" }
        return "star"
    }

    private func metricBullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 5) {
            Text("•")
                .foregroundColor(.secondary)
                .font(.caption)
            Text(text)
                .font(.caption)
                .foregroundColor(.primary)
        }
    }

    // MARK: - Performance Insights (vertical grouped bar chart — 3 metrics)
    private var performanceInsightsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Performance Insights")
                .font(.system(size: 17, weight: .bold))
                .foregroundColor(.white)

            GeometryReader { geo in
                let totalW = geo.size.width
                let groupW = (totalW - 40) / 3   // 3 groups

                HStack(alignment: .bottom, spacing: 20) {
                    insightGroup(title: "Posture",
                                 val1: look1.postureScore, val2: look2.postureScore,
                                 groupWidth: groupW)

                    insightGroup(title: "Fit",
                                 val1: look1.fitScore, val2: look2.fitScore,
                                 groupWidth: groupW)

                    insightGroup(title: "Overall Score",
                                 val1: look1.score, val2: look2.score,
                                 groupWidth: groupW)
                }
            }
            .frame(height: 130)
        }
        .padding(16)
        .glassCard(cornerRadius: 16)
    }

    private func insightGroup(title: String, val1: Double, val2: Double, groupWidth: CGFloat) -> some View {
        let maxH: CGFloat = 90
        let h1 = CGFloat(max(0.1, (val1 - 4.0) / 6.0)) * maxH
        let h2 = CGFloat(max(0.1, (val2 - 4.0) / 6.0)) * maxH

        return VStack(spacing: 4) {
            HStack(alignment: .bottom, spacing: 6) {
                // Look 1 bar (grey)
                VStack {
                    Spacer()
                    RoundedRectangle(cornerRadius: 5)
                        .fill(Color(white: 0.35))
                        .frame(width: (groupWidth - 12) / 2, height: h1)
                }
                .frame(height: maxH)

                // Look 2 bar (white/bright)
                VStack {
                    Spacer()
                    RoundedRectangle(cornerRadius: 5)
                        .fill(Color(white: 0.85))
                        .frame(width: (groupWidth - 12) / 2, height: h2)
                }
                .frame(height: maxH)
            }
            .frame(width: groupWidth)

            Rectangle()
                .fill(Color.white.opacity(0.08))
                .frame(height: 1)
                .frame(width: groupWidth)

            Text(title)
                .font(.system(size: 10, weight: .medium))
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(width: groupWidth)
        }
    }
}
