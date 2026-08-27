import SwiftUI

struct ScoreComparisonBar: View {
    let looks: [LookItem]
    let selectedId: UUID?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("LOOK COMPARISON")
                .font(.caption2.bold())
                .foregroundColor(.secondary)
                .tracking(0.5)

            HStack(alignment: .bottom, spacing: 6) {
                ForEach(Array(looks.enumerated()), id: \.element.id) { idx, look in
                    let isSelected = (selectedId ?? looks.first?.id) == look.id
                    VStack(spacing: 4) {
                        Text(String(format: "%.1f", look.score))
                            .font(.caption2.bold())
                            .foregroundColor(isSelected ? .white : .primary)
                        GeometryReader { geo in
                            VStack {
                                Spacer()
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(isSelected ? Color.purple : Color.blue.opacity(0.4))
                                    .frame(height: barHeight(look.score, in: geo.size.height))
                            }
                        }
                        .frame(height: 50)
                        Text("Look \(idx + 1)").font(.caption2).foregroundColor(.secondary)
                    }
                }
            }
            .frame(height: 80)
        }
        .padding(12)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
    }

    private func barHeight(_ score: Double, in total: CGFloat) -> CGFloat {
        let norm = (score - 6.0) / 4.0
        return max(8, CGFloat(norm) * total)
    }
}
