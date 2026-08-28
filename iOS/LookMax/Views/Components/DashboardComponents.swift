import SwiftUI

struct ProfileHeaderBanner: View {
    @Binding var profile: UserProfile?
    @Binding var showingOnboarding: Bool

    var body: some View {
        HStack(spacing: 14) {
            // User Avatar
            if let p = profile, let data = p.photoDataList.first, let img = UIImage(data: data) {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 48, height: 48)
                    .clipShape(Circle())
                    .overlay(Circle().stroke(Theme.neonCyan, lineWidth: 2))
                    .neonGlow(color: Theme.neonCyan, radius: 4)
            } else {
                ZStack {
                    Circle()
                        .fill(Theme.surfaceDark)
                        .frame(width: 48, height: 48)
                        .overlay(Circle().stroke(Theme.cardBorder, lineWidth: 1))

                    Image(systemName: "person.fill")
                        .font(.title3)
                        .foregroundColor(Theme.neonCyan)
                }
            }

            // Name & Subtitle
            VStack(alignment: .leading, spacing: 3) {
                Text(profile?.name ?? "Sarah Johnson")
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundColor(.white)

                Text("San Francisco, CA")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                // Glowing Biometrics Active Pill
                HStack(spacing: 4) {
                    Circle()
                        .fill(Theme.emerald)
                        .frame(width: 6, height: 6)
                        .shadow(color: Theme.emerald, radius: 3)

                    Text("Biometrics Active")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundColor(Theme.emerald)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .background(Theme.emerald.opacity(0.12))
                .clipShape(Capsule())
                .overlay(Capsule().stroke(Theme.emerald.opacity(0.3), lineWidth: 0.8))
            }

            Spacer()

            Button(action: {
                HapticManager.light()
                showingOnboarding = true
            }) {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.white.opacity(0.8))
                    .frame(width: 38, height: 38)
                    .glassCard(cornerRadius: 12)
            }
        }
        .padding(14)
        .glassCard(cornerRadius: 18)
    }
}

struct StartSessionCTA: View {
    var onAction: () -> Void

    var body: some View {
        Button(action: {
            HapticManager.medium()
            onAction()
        }) {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(Color.black.opacity(0.25))
                        .frame(width: 36, height: 36)

                    Image(systemName: "plus")
                        .font(.system(size: 16, weight: .heavy))
                        .foregroundColor(.black)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("START NEW STYLE SESSION")
                        .font(.system(size: 14, weight: .heavy, design: .rounded))
                        .foregroundColor(.black)
                        .tracking(0.5)

                    Text("Generate New Look")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.black.opacity(0.75))
                }

                Spacer()

                Image(systemName: "sparkles")
                    .font(.title3)
                    .foregroundColor(.black.opacity(0.8))
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .background(Theme.neonGradient)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .shadow(color: Theme.neonCyan.opacity(0.5), radius: 12)
        }
    }
}

struct SessionCardView: View {
    let session: LookSession

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header Row: Occasion Badge + Top Pick Badge
            HStack(alignment: .center) {
                Text(session.title.uppercased())
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundColor(.white)

                Spacer()

                if let best = session.bestLook {
                    HStack(spacing: 4) {
                        Text("TOP PICK:")
                            .font(.system(size: 10, weight: .black, design: .rounded))
                            .foregroundColor(.black)

                        Text(String(format: "%.1f/10", best.score))
                            .font(.system(size: 10, weight: .black, design: .rounded))
                            .foregroundColor(.black)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Theme.emeraldGradient)
                    .clipShape(Capsule())
                    .shadow(color: Theme.emerald.opacity(0.5), radius: 6)
                }
            }

            // Horizontal Scrollable Thumbnail Deck
            if !session.looks.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(session.looks) { look in
                            if let img = look.image {
                                ZStack(alignment: .bottomTrailing) {
                                    Image(uiImage: img)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(width: 72, height: 90)
                                        .clipShape(RoundedRectangle(cornerRadius: 10))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 10)
                                                .stroke(Theme.scoreColor(look.score).opacity(0.6), lineWidth: 1.5)
                                        )

                                    Text(String(format: "%.1f", look.score))
                                        .font(.system(size: 9, weight: .heavy, design: .rounded))
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 4)
                                        .padding(.vertical, 2)
                                        .background(Theme.scoreColor(look.score))
                                        .clipShape(Capsule())
                                        .padding(3)
                                }
                            }
                        }
                    }
                }
            } else {
                Text("No looks captured yet. Tap to start evaluation.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .italic()
                    .padding(.vertical, 12)
            }

            // Footer Metadata Row
            HStack {
                // Occasion Tag Pill
                HStack(spacing: 4) {
                    Image(systemName: session.occasion.icon)
                        .font(.system(size: 10))
                    Text(session.occasion.rawValue.uppercased())
                        .font(.system(size: 9, weight: .bold, design: .rounded))
                }
                .foregroundColor(session.occasion.color)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(session.occasion.color.opacity(0.12))
                .clipShape(Capsule())

                Text("•")
                    .foregroundColor(.secondary)
                    .font(.caption2)

                Text(session.formattedDate)
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Spacer()

                Text("Total Looks: \(session.looks.count)")
                    .font(.caption2.bold())
                    .foregroundColor(.secondary)
            }

            // Tags Footer
            Text(session.tagsFormatted)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.white.opacity(0.7))
        }
        .padding(16)
        .glassCard(cornerRadius: 18)
        .contentShape(Rectangle())
    }
}

struct EmptySessionsView: View {
    let onNew: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Spacer()

            ZStack {
                Circle()
                    .fill(Theme.neonCyan.opacity(0.08))
                    .frame(width: 120, height: 120)

                Image(systemName: "sparkles.rectangle.stack.fill")
                    .font(.system(size: 56))
                    .foregroundColor(Theme.neonCyan)
                    .neonGlow(color: Theme.neonCyan, radius: 16)
            }

            Text("No Style Sessions Yet")
                .font(.title2.bold())
                .foregroundColor(.white)

            Text("Create an occasion session to evaluate outfits, receive AI biometric posture scores, and track your styling progress.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 36)

            Button(action: {
                HapticManager.medium()
                onNew()
            }) {
                HStack(spacing: 8) {
                    Image(systemName: "plus.circle.fill")
                    Text("Start Your First Session")
                }
                .font(.headline)
                .padding()
                .frame(maxWidth: .infinity)
                .background(Theme.neonGradient)
                .foregroundColor(.black)
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .shadow(color: Theme.neonCyan.opacity(0.4), radius: 10)
            }
            .padding(.horizontal, 30)
            .padding(.top, 8)

            Spacer()
        }
    }
}
