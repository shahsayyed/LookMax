import SwiftUI

struct SuggestionRow: View {
    let suggestion: StyleSuggestion
    @State private var isDone: Bool

    init(suggestion: StyleSuggestion) {
        self.suggestion = suggestion
        _isDone = State(initialValue: suggestion.isDone)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Button(action: {
                isDone.toggle()
                HapticManager.medium()
            }) {
                Image(systemName: isDone ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isDone ? Theme.emerald : .secondary)
                    .font(.title3)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Label(suggestion.category, systemImage: suggestion.icon)
                        .font(.caption2.bold())
                        .foregroundColor(suggestion.iconColor)

                    Spacer()

                    Text(suggestion.effortTime)
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Theme.neonCyan.opacity(0.12))
                        .foregroundColor(Theme.neonCyan)
                        .clipShape(Capsule())
                }

                Text(suggestion.title)
                    .font(.subheadline.bold())
                    .strikethrough(isDone, color: .secondary)
                    .foregroundColor(isDone ? .secondary : .white)

                Text(suggestion.recommendation)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(isDone ? Theme.emerald.opacity(0.08) : Theme.surfaceDark.opacity(0.7))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isDone ? Theme.emerald.opacity(0.3) : Theme.cardBorderSubtle, lineWidth: 1)
        )
    }
}

struct LookDetailCard: View {
    let look: LookItem
    let isBestLook: Bool
    var onCompareTapped: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            // Header / Overall Score Section
            HStack(alignment: .center, spacing: 16) {
                // Score Box
                VStack(spacing: 2) {
                    Text(String(format: "%.1f", look.score))
                        .font(.system(size: 46, weight: .heavy, design: .rounded))
                        .foregroundColor(Theme.scoreColor(look.score))
                        .neonGlow(color: Theme.scoreColor(look.score), radius: 12)

                    Text("OVERALL SCORE")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundColor(.secondary)
                        .tracking(0.5)

                    Text(look.headlineBadge)
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(Theme.scoreColor(look.score))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Theme.scoreColor(look.score).opacity(0.15))
                        .clipShape(Capsule())
                        .padding(.top, 4)
                }
                .frame(width: 130)

                VStack(alignment: .leading, spacing: 6) {
                    if isBestLook {
                        Label("Top Pick in Session", systemImage: "trophy.fill")
                            .font(.caption2.bold())
                            .foregroundColor(Theme.warmAmber)
                    }

                    Text("\(look.detectedFaceShape) Face")
                        .font(.subheadline.bold())
                        .foregroundColor(.white)

                    Text(look.detectedOutfitColor)
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 6) {
                        Text("Clarity: \(look.lightingScore)%")
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.white.opacity(0.08))
                            .clipShape(Capsule())
                    }
                    .padding(.top, 4)
                }

                Spacer()
            }

            Divider().background(Theme.cardBorder)

            // What Looked Good
            VStack(alignment: .leading, spacing: 8) {
                Label("WHAT LOOKED GOOD", systemImage: "hand.thumbsup.fill")
                    .font(.caption.bold())
                    .foregroundColor(Theme.emerald)

                ForEach(look.goodPoints, id: \.self) { point in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(Theme.emerald)
                            .font(.caption)
                            .padding(.top, 2)
                        Text(point)
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.9))
                    }
                }
            }

            Divider().background(Theme.cardBorder)

            // What Needs Improvement
            VStack(alignment: .leading, spacing: 8) {
                Label("WHAT NEEDS IMPROVEMENT", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption.bold())
                    .foregroundColor(Theme.warmAmber)

                ForEach(look.badPoints, id: \.self) { point in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "arrow.up.circle.fill")
                            .foregroundColor(Theme.warmAmber)
                            .font(.caption)
                            .padding(.top, 2)
                        Text(point)
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.9))
                    }
                }
            }

            // 5-Min Tweaks Checklist
            if !look.suggestions.isEmpty {
                Divider().background(Theme.cardBorder)

                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Label("5-MIN TWEAKS INTERACTIVE CHECKLIST", systemImage: "clock.arrow.circlepath")
                            .font(.caption.bold())
                            .foregroundColor(Theme.neonCyan)

                        Spacer()
                    }

                    ForEach(look.suggestions) { sug in
                        SuggestionRow(suggestion: sug)
                    }
                }
            }
        }
        .padding(18)
        .glassCard(cornerRadius: 20)
    }
}
