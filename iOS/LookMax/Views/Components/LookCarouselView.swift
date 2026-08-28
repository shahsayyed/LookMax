import SwiftUI

struct LookCarouselView: View {
    let looks: [LookItem]
    @Binding var selectedId: UUID?
    var onDelete: ((LookItem) -> Void)? = nil

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 14) {
                ForEach(Array(looks.enumerated()), id: \.element.id) { index, look in
                    let isSelected = (selectedId ?? looks.first?.id) == look.id
                    let scoreCol = Theme.scoreColor(look.score)

                    VStack(spacing: 6) {
                        ZStack(alignment: .topTrailing) {
                            if let img = look.image {
                                Image(uiImage: img)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 104, height: 136)
                                    .clipShape(RoundedRectangle(cornerRadius: 14))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 14)
                                            .stroke(
                                                isSelected ? scoreCol : Color.white.opacity(0.12),
                                                lineWidth: isSelected ? 3 : 1
                                            )
                                    )
                                    .shadow(color: isSelected ? scoreCol.opacity(0.6) : Color.clear, radius: 8)
                            } else {
                                RoundedRectangle(cornerRadius: 14)
                                    .fill(Theme.cardDark)
                                    .frame(width: 104, height: 136)
                            }

                            // Score Badge — rounded rectangle at top-right, per design
                            Text(String(format: "%.1f", look.score))
                                .font(.system(size: 11, weight: .black, design: .rounded))
                                .foregroundColor(.white)
                                .padding(.horizontal, 7)
                                .padding(.vertical, 4)
                                .background(scoreCol)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .padding(.top, 6)
                                .padding(.trailing, 6)
                        }
                        .overlay(
                            // Trash delete button (bottom-trailing) — appears on selected look
                            Group {
                                if isSelected, let onDelete = onDelete {
                                    Button(action: {
                                        HapticManager.medium()
                                        onDelete(look)
                                    }) {
                                        Image(systemName: "trash.fill")
                                            .font(.system(size: 12, weight: .bold))
                                            .foregroundColor(.white)
                                            .frame(width: 28, height: 28)
                                            .background(Color.red.opacity(0.85))
                                            .clipShape(Circle())
                                            .shadow(color: .black.opacity(0.4), radius: 4)
                                    }
                                    .padding(6)
                                }
                            },
                            alignment: .bottomTrailing
                        )

                        HStack(spacing: 4) {
                            Text("Look \(index + 1)")
                                .font(.caption2.bold())
                                .foregroundColor(isSelected ? .white : .secondary)

                            Text("• \(look.formattedTime)")
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                        }
                    }
                    .scaleEffect(isSelected ? 1.04 : 0.98)
                    .animation(.spring(response: 0.25, dampingFraction: 0.7), value: isSelected)
                    .onTapGesture {
                        selectedId = look.id
                        HapticManager.light()
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
    }
}
