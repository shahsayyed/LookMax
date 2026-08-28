import SwiftUI

struct SuggestionRow: View {
    let suggestion: StyleSuggestion
    @State private var isDone: Bool

    init(suggestion: StyleSuggestion) {
        self.suggestion = suggestion
        _isDone = State(initialValue: suggestion.isDone)
    }

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            // Square-style checkmark matching the design image
            Button(action: {
                isDone.toggle()
                HapticManager.medium()
            }) {
                Image(systemName: isDone ? "checkmark.square.fill" : "square")
                    .foregroundColor(isDone ? Theme.emerald : Color.white.opacity(0.5))
                    .font(.system(size: 20))
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(suggestion.title)
                    .font(.system(size: 14, weight: .semibold))
                    .strikethrough(isDone, color: .secondary)
                    .foregroundColor(isDone ? .secondary : .white)

                Text(suggestion.effortTime)
                    .font(.system(size: 12, weight: .regular))
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            isDone.toggle()
            HapticManager.medium()
        }
    }
}

struct LookDetailCard: View {
    let look: LookItem
    let isBestLook: Bool
    var onCompareTapped: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            // ─── Left Column: Large Score ───
            VStack(spacing: 6) {
                Text("AI LOOK DETAIL")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .foregroundColor(.secondary)
                    .tracking(0.6)

                Text(String(format: "%.1f", look.score))
                    .font(.system(size: 64, weight: .heavy, design: .rounded))
                    .foregroundColor(Theme.scoreColor(look.score))
                    .neonGlow(color: Theme.scoreColor(look.score), radius: 16)

                Text("OVERALL SCORE")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .foregroundColor(.secondary)
                    .tracking(0.4)

                Text(look.headlineBadge)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Theme.scoreColor(look.score))
                    .clipShape(Capsule())
                    .padding(.top, 2)

                if isBestLook {
                    Label("Top Pick", systemImage: "trophy.fill")
                        .font(.caption2.bold())
                        .foregroundColor(Theme.warmAmber)
                        .padding(.top, 4)
                }
            }
            .frame(minWidth: 130)

            // ─── Right Column: Orange "5-Min Tweaks" Checklist ───
            VStack(alignment: .leading, spacing: 0) {
                // Orange header block — exact match to the design
                VStack(alignment: .leading, spacing: 2) {
                    Text("5-Min Tweaks")
                        .font(.system(size: 14, weight: .heavy, design: .rounded))
                        .foregroundColor(.white)
                    Text("Interactive Checklist")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.white.opacity(0.8))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(Theme.warmAmber)
                .clipShape(RoundedRectangle(cornerRadius: 10))

                // Suggestion rows
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(look.suggestions.prefix(5)) { sug in
                        SuggestionRow(suggestion: sug)
                        if sug.id != look.suggestions.prefix(5).last?.id {
                            Divider()
                                .background(Color.white.opacity(0.07))
                                .padding(.leading, 30)
                        }
                    }
                }
                .padding(.horizontal, 4)
                .padding(.top, 6)
            }
        }
        .padding(16)
        .glassCard(cornerRadius: 18)
    }
}
