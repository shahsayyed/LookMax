import SwiftUI

struct CreateSessionSheet: View {
    let onCreated: (LookSession) -> Void
    @Environment(\.presentationMode) var presentationMode

    @State private var title = ""
    @State private var occasion: OccasionCategory = .casualEveryday

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Session Name")) {
                    TextField("e.g. Board Meeting, Date Night, Job Interview", text: $title)
                }

                Section(header: Text("Occasion")) {
                    ForEach(OccasionCategory.allCases, id: \.self) { cat in
                        HStack {
                            Label(cat.rawValue, systemImage: cat.icon)
                                .foregroundColor(cat.color)
                            Spacer()
                            if occasion == cat {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.blue)
                            }
                        }
                        .contentShape(Rectangle())
                        .onTapGesture { occasion = cat }
                    }
                }
            }
            .navigationTitle("New Session")
            .navigationBarItems(
                leading: Button("Cancel") { presentationMode.wrappedValue.dismiss() },
                trailing: Button("Create") {
                    let trimmed = title.trimmingCharacters(in: .whitespaces)
                    let name = trimmed.isEmpty ? "\(occasion.rawValue) Session" : trimmed
                    let session = LookSession(title: name, occasion: occasion)
                    onCreated(session)
                    presentationMode.wrappedValue.dismiss()
                }.bold()
            )
        }
    }
}
