import SwiftUI

struct CreateSessionSheet: View {
    let onCreated: (LookSession) -> Void
    @Environment(\.presentationMode) var presentationMode

    @State private var title = ""
    @State private var occasion: OccasionCategory = .businessMeeting

    var body: some View {
        NavigationView {
            ZStack {
                Theme.oledBlack.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 24) {
                        // Title Input Card
                        VStack(alignment: .leading, spacing: 8) {
                            Text("SESSION NAME")
                                .font(.system(size: 11, weight: .bold, design: .rounded))
                                .foregroundColor(.secondary)
                                .tracking(0.8)

                            TextField("e.g. Business Meeting, Date Night", text: $title)
                                .font(.body)
                                .foregroundColor(.white)
                                .padding(14)
                                .background(Theme.surfaceDark)
                                .cornerRadius(12)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Theme.cardBorder, lineWidth: 1)
                                )
                        }
                        .padding(16)
                        .glassCard(cornerRadius: 18)

                        // Occasion Selector
                        VStack(alignment: .leading, spacing: 12) {
                            Text("SELECT OCCASION")
                                .font(.system(size: 11, weight: .bold, design: .rounded))
                                .foregroundColor(.secondary)
                                .tracking(0.8)

                            VStack(spacing: 8) {
                                ForEach(OccasionCategory.allCases, id: \.self) { cat in
                                    let isSelected = occasion == cat
                                    HStack {
                                        Label(cat.rawValue, systemImage: cat.icon)
                                            .font(.subheadline.bold())
                                            .foregroundColor(isSelected ? .white : .secondary)

                                        Spacer()

                                        if isSelected {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundColor(Theme.neonCyan)
                                        }
                                    }
                                    .padding(14)
                                    .background(isSelected ? Theme.neonCyan.opacity(0.12) : Theme.surfaceDark)
                                    .cornerRadius(12)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(isSelected ? Theme.neonCyan.opacity(0.6) : Theme.cardBorderSubtle, lineWidth: 1)
                                    )
                                    .contentShape(Rectangle())
                                    .onTapGesture {
                                        occasion = cat
                                        HapticManager.light()
                                    }
                                }
                            }
                        }
                        .padding(16)
                        .glassCard(cornerRadius: 18)

                        // Create Button
                        Button(action: createSession) {
                            Text("Create Session")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Theme.neonGradient)
                                .foregroundColor(.black)
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                                .shadow(color: Theme.neonCyan.opacity(0.4), radius: 8)
                        }
                        .padding(.top, 8)
                    }
                    .padding(16)
                }
            }
            .navigationTitle("New Style Session")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                    .foregroundColor(.secondary)
                }
            }
        }
    }

    private func createSession() {
        let trimmed = title.trimmingCharacters(in: .whitespaces)
        let name = trimmed.isEmpty ? "\(occasion.rawValue)" : trimmed
        let session = LookSession(title: name, occasion: occasion)
        onCreated(session)
        HapticManager.success()
        presentationMode.wrappedValue.dismiss()
    }
}
