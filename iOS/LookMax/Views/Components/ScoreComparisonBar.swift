import SwiftUI

struct ScoreComparisonBar: View {
    let looks: [LookItem]
    let selectedId: UUID?
    var onCompareTapped: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Outfit Comparison")
                    .font(.system(size: 17, weight: .bold))
                    .foregroundColor(.white)

                Spacer()

                if looks.count >= 2, let onCompare = onCompareTapped {
                    Button(action: {
                        HapticManager.light()
                        onCompare()
                    }) {
                        Image(systemName: "arrow.left.arrow.right")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(Theme.neonCyan)
                    }
                }
            }

            // Bar chart: score above, solid bar, outfit label, thumbnail below
            HStack(alignment: .bottom, spacing: 16) {
                ForEach(Array(looks.enumerated()), id: \.element.id) { idx, look in
                    let isSelected = (selectedId ?? looks.first?.id) == look.id
                    let col = Theme.scoreColor(look.score)

                    VStack(spacing: 0) {
                        // Score value floating above bar
                        Text(String(format: "%.1f", look.score))
                            .font(.system(size: 13, weight: .bold, design: .rounded))
                            .foregroundColor(col)
                            .padding(.bottom, 4)

                        // Solid bar (no gradient — matches design)
                        GeometryReader { geo in
                            VStack(spacing: 0) {
                                Spacer()
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(col)
                                    .frame(height: barHeight(look.score, in: geo.size.height))
                                    .shadow(color: isSelected ? col.opacity(0.6) : Color.clear, radius: 8)
                            }
                        }
                        .frame(height: 100)

                        // X-axis baseline
                        Rectangle()
                            .fill(Color.white.opacity(0.10))
                            .frame(height: 1)
                            .padding(.vertical, 6)

                        // "Outfit N" label
                        Text("Outfit \(idx + 1)")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(.secondary)
                            .padding(.bottom, 6)

                        // Thumbnail + score badge below x-axis
                        ZStack(alignment: .bottomTrailing) {
                            if let img = look.image {
                                Image(uiImage: img)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 44, height: 44)
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(isSelected ? col : Color.white.opacity(0.15), lineWidth: 1.5)
                                    )
                            } else {
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Theme.surfaceDark)
                                    .frame(width: 44, height: 44)
                            }

                            Text(String(format: "%.1f", look.score))
                                .font(.system(size: 9, weight: .heavy, design: .rounded))
                                .foregroundColor(.white)
                                .padding(.horizontal, 3)
                                .padding(.vertical, 1)
                                .background(col)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                                .offset(x: 2, y: 2)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .opacity(isSelected ? 1.0 : 0.65)
                    .scaleEffect(isSelected ? 1.02 : 1.0)
                    .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isSelected)
                }
            }
        }
        .padding(16)
        .glassCard(cornerRadius: 16)
    }

    private func barHeight(_ score: Double, in total: CGFloat) -> CGFloat {
        let norm = max(0.15, (score - 4.0) / 6.0)
        return CGFloat(norm) * total
    }
}
