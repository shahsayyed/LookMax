import SwiftUI

struct ScoreComparisonBar: View {
    let looks: [LookItem]
    let selectedId: UUID?
    var onCompareTapped: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("OUTFIT COMPARISON")
                    .font(.system(size: 12, weight: .bold, design: .rounded))
                    .foregroundColor(.secondary)
                    .tracking(0.8)

                Spacer()

                if looks.count >= 2, let onCompare = onCompareTapped {
                    Button(action: {
                        HapticManager.light()
                        onCompare()
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "slider.horizontal.below.rectangle")
                            Text("Before / After")
                        }
                        .font(.caption.bold())
                        .foregroundColor(Theme.neonCyan)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Theme.neonCyan.opacity(0.12))
                        .clipShape(Capsule())
                    }
                }
            }

            HStack(alignment: .bottom, spacing: 12) {
                ForEach(Array(looks.enumerated()), id: \.element.id) { idx, look in
                    let isSelected = (selectedId ?? looks.first?.id) == look.id
                    let col = Theme.scoreColor(look.score)

                    VStack(spacing: 6) {
                        Text(String(format: "%.1f", look.score))
                            .font(.system(size: 12, weight: .bold, design: .rounded))
                            .foregroundColor(isSelected ? col : .secondary)

                        GeometryReader { geo in
                            VStack {
                                Spacer()
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(
                                        LinearGradient(
                                            colors: isSelected ? [col, col.opacity(0.6)] : [col.opacity(0.4), col.opacity(0.2)],
                                            startPoint: .top,
                                            endPoint: .bottom
                                        )
                                    )
                                    .frame(height: barHeight(look.score, in: geo.size.height))
                                    .shadow(color: isSelected ? col.opacity(0.5) : Color.clear, radius: 6)
                            }
                        }
                        .frame(height: 70)

                        Text("Outfit \(idx + 1)")
                            .font(.system(size: 11, weight: isSelected ? .bold : .medium))
                            .foregroundColor(isSelected ? .white : .secondary)
                    }
                }
            }
        }
        .padding(16)
        .glassCard(cornerRadius: 16)
    }

    private func barHeight(_ score: Double, in total: CGFloat) -> CGFloat {
        let norm = max(0.2, (score - 4.0) / 6.0)
        return CGFloat(norm) * total
    }
}
