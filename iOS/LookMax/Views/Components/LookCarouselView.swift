import SwiftUI

struct LookCarouselView: View {
    let looks: [LookItem]
    @Binding var selectedId: UUID?

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(looks) { look in
                    let isSelected = (selectedId ?? looks.first?.id) == look.id
                    VStack(spacing: 4) {
                        ZStack(alignment: .topTrailing) {
                            if let img = look.image {
                                Image(uiImage: img)
                                    .resizable().scaledToFill()
                                    .frame(width: 100, height: 130)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(isSelected ? Color.purple : Color.clear, lineWidth: 2.5)
                                    )
                            }
                            Text(String(format: "%.1f", look.score))
                                .font(.caption2.bold())
                                .foregroundColor(.white)
                                .padding(.horizontal, 5).padding(.vertical, 2)
                                .background(scoreColor(look.score))
                                .clipShape(Capsule())
                                .padding(5)
                        }
                        Text(look.formattedTime)
                            .font(.caption2)
                            .foregroundColor(isSelected ? .purple : .secondary)
                    }
                    .onTapGesture { selectedId = look.id }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
        }
    }

    private func scoreColor(_ score: Double) -> Color {
        if score >= 8.5 { return .green }
        if score >= 7.5 { return .blue }
        return .orange
    }
}
