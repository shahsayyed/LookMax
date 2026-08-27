import SwiftUI

struct SuggestionRow: View {
    let suggestion: StyleSuggestion
    @State private var isDone: Bool

    init(suggestion: StyleSuggestion) {
        self.suggestion = suggestion
        _isDone = State(initialValue: suggestion.isDone)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Button(action: {
                isDone.toggle()
                UIImpactFeedbackGenerator(style: .light).impactOccurred()
            }) {
                Image(systemName: isDone ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isDone ? .green : .secondary)
                    .font(.title3)
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Label(suggestion.category, systemImage: suggestion.icon)
                        .font(.caption2.bold())
                        .foregroundColor(suggestion.iconColor)
                    Spacer()
                    Text(suggestion.effortTime)
                        .font(.caption2)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.blue.opacity(0.10))
                        .foregroundColor(.blue)
                        .clipShape(Capsule())
                }
                Text(suggestion.title)
                    .font(.subheadline.bold())
                    .strikethrough(isDone, color: .secondary)
                    .foregroundColor(isDone ? .secondary : .primary)
                Text(suggestion.recommendation)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .background(isDone ? Color.green.opacity(0.06) : Color(UIColor.tertiarySystemBackground))
        .cornerRadius(10)
    }
}

struct LookDetailCard: View {
    let look: LookItem
    let isBestLook: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {

            // Score Header
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 4) {
                    if isBestLook {
                        Label("Top Pick in Session", systemImage: "trophy.fill")
                            .font(.caption2.bold())
                            .foregroundColor(.yellow)
                    }
                    Text(look.headlineBadge).font(.title3.bold())
                    Text("\(look.detectedFaceShape) Face • \(look.detectedOutfitColor)")
                        .font(.caption).foregroundColor(.secondary)
                }
                Spacer()
                HStack(alignment: .firstTextBaseline, spacing: 2) {
                    Text(String(format: "%.1f", look.score))
                        .font(.system(size: 40, weight: .heavy, design: .rounded))
                        .foregroundColor(.purple)
                    Text("/10").font(.subheadline.bold()).foregroundColor(.secondary)
                }
            }

            Divider()

            // What Looked Good
            VStack(alignment: .leading, spacing: 6) {
                Label("WHAT LOOKED GOOD", systemImage: "hand.thumbsup.fill")
                    .font(.caption.bold()).foregroundColor(.green)
                ForEach(look.goodPoints, id: \.self) { point in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green).font(.caption).padding(.top, 1)
                        Text(point).font(.subheadline)
                    }
                }
            }

            Divider()

            // What Needs Improvement
            VStack(alignment: .leading, spacing: 6) {
                Label("WHAT NEEDS IMPROVEMENT", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption.bold()).foregroundColor(.orange)
                ForEach(look.badPoints, id: \.self) { point in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "arrow.up.circle.fill")
                            .foregroundColor(.orange).font(.caption).padding(.top, 1)
                        Text(point).font(.subheadline)
                    }
                }
            }

            if !look.suggestions.isEmpty {
                Divider()

                VStack(alignment: .leading, spacing: 10) {
                    Label("QUICK 5-MIN TWEAKS", systemImage: "clock.arrow.circlepath")
                        .font(.caption.bold()).foregroundColor(.blue)
                    ForEach(look.suggestions) { sug in
                        SuggestionRow(suggestion: sug)
                    }
                }
            }
        }
        .padding(16)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
    }
}
