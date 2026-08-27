import SwiftUI

struct SessionCardView: View {
    let session: LookSession

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(session.occasion.rawValue, systemImage: session.occasion.icon)
                    .font(.caption.bold())
                    .foregroundColor(session.occasion.color)
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(session.occasion.color.opacity(0.12))
                    .clipShape(Capsule())

                Spacer()

                Text(session.formattedDate)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Text(session.title).font(.headline)

            if !session.looks.isEmpty {
                HStack(spacing: 6) {
                    ForEach(session.looks.prefix(4)) { look in
                        if let img = look.image {
                            Image(uiImage: img)
                                .resizable().scaledToFill()
                                .frame(width: 56, height: 56)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    if session.looks.count > 4 {
                        ZStack {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.secondary.opacity(0.15))
                                .frame(width: 56, height: 56)
                            Text("+\(session.looks.count - 4)")
                                .font(.subheadline.bold())
                                .foregroundColor(.secondary)
                        }
                    }
                    Spacer()
                    if let best = session.bestLook {
                        VStack(alignment: .trailing, spacing: 2) {
                            HStack(alignment: .firstTextBaseline, spacing: 2) {
                                Text(String(format: "%.1f", best.score))
                                    .font(.title3.bold()).foregroundColor(.purple)
                                Text("/10").font(.caption).foregroundColor(.secondary)
                            }
                            Text("Best Look").font(.caption2).foregroundColor(.secondary)
                        }
                    }
                }
            } else {
                Text("No looks added yet — tap to open and start scanning!")
                    .font(.caption).foregroundColor(.secondary).italic()
            }
        }
        .padding(14)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
        .contentShape(Rectangle())
    }
}

struct ProfileHeaderBanner: View {
    @Binding var profile: UserProfile?
    @Binding var showingOnboarding: Bool

    var body: some View {
        HStack(spacing: 12) {
            if let p = profile {
                if let data = p.photoDataList.first, let img = UIImage(data: data) {
                    Image(uiImage: img)
                        .resizable().scaledToFill()
                        .frame(width: 40, height: 40)
                        .clipShape(Circle())
                        .overlay(Circle().stroke(Color.blue, lineWidth: 2))
                } else {
                    Image(systemName: "person.circle.fill")
                        .font(.system(size: 40)).foregroundColor(.blue)
                }
                VStack(alignment: .leading, spacing: 1) {
                    Text(p.name).font(.subheadline.bold())
                    Text("Personal Style Client").font(.caption2).foregroundColor(.secondary)
                }
            } else {
                Image(systemName: "person.crop.circle.badge.plus")
                    .font(.system(size: 36)).foregroundColor(.secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Setup Profile").font(.subheadline.bold())
                    Text("For identity tracking & style history").font(.caption2).foregroundColor(.secondary)
                }
            }

            Spacer()

            Button(profile == nil ? "Enroll" : "Edit") { showingOnboarding = true }
                .font(.caption.bold())
                .padding(.horizontal, 12).padding(.vertical, 5)
                .background(Color.blue.opacity(0.12))
                .foregroundColor(.blue)
                .clipShape(Capsule())
        }
        .padding(12)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(14)
    }
}

struct EmptySessionsView: View {
    let onNew: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "sparkles.rectangle.stack.fill")
                .font(.system(size: 64)).foregroundColor(.purple.opacity(0.8))
            Text("No Style Sessions Yet").font(.title2.bold())
            Text("Create your first session to start building your personal style history. Try different outfits and let the AI tell you what works.")
                .font(.subheadline).foregroundColor(.secondary)
                .multilineTextAlignment(.center).padding(.horizontal, 36)
            Button(action: onNew) {
                Label("Start Your First Session", systemImage: "plus.circle.fill")
                    .font(.headline).padding().frame(maxWidth: .infinity)
                    .background(Color.blue).foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .padding(.horizontal, 30)
            Spacer()
        }
    }
}
