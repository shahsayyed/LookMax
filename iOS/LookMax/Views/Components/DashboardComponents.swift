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
                    .frame(width: 52, height: 52)
                    .clipShape(Circle())
                    .overlay(Circle().stroke(Theme.neonCyan, lineWidth: 2))
                    .neonGlow(color: Theme.neonCyan, radius: 4)
            } else {
                ZStack {
                    Circle()
                        .fill(Theme.surfaceDark)
                        .frame(width: 52, height: 52)
                        .overlay(Circle().stroke(Theme.cardBorder, lineWidth: 1))
                    Image(systemName: "person.fill")
                        .font(.title3)
                        .foregroundColor(Theme.neonCyan)
                }
            }

            // Name & Subtitle
            VStack(alignment: .leading, spacing: 3) {
                Text(profile?.name ?? "Sarah Johnson")
                    .font(.system(size: 17, weight: .bold, design: .rounded))
                    .foregroundColor(.white)

                Text("San Francisco, CA")
                    .font(.caption)
                    .foregroundColor(.secondary)

                // Glowing Biometrics Active Pill
                HStack(spacing: 5) {
                    Image(systemName: "faceid")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(Theme.emerald)

                    Text("Biometrics Active")
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .foregroundColor(Theme.emerald)
                }
                .padding(.horizontal, 9)
                .padding(.vertical, 3)
                .background(Theme.emerald.opacity(0.12))
                .clipShape(Capsule())
                .overlay(Capsule().stroke(Theme.emerald.opacity(0.35), lineWidth: 0.8))
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
                        .fill(Color.black.opacity(0.2))
                        .frame(width: 36, height: 36)
                    Image(systemName: "plus")
                        .font(.system(size: 17, weight: .heavy))
                        .foregroundColor(.black)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("START NEW STYLE SESSION ✨")
                        .font(.system(size: 14, weight: .heavy, design: .rounded))
                        .foregroundColor(.black)
                        .tracking(0.3)

                    Text("Generate New Look")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.black.opacity(0.70))
                }

                Spacer()
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Theme.neonCyan, lineWidth: 2.5)
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Theme.neonGradient)
                    )
            )
            .shadow(color: Theme.neonCyan.opacity(0.55), radius: 12)
        }
    }
}

struct SessionCardView: View {
    let session: LookSession

    /// Returns a formatted date string and a status string/color pair
    private var sessionMeta: (date: String, status: String, statusColor: Color) {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        let dateStr = formatter.string(from: session.createdAt)

        // Derive a fake "days left" / "completed" status from date
        let daysSince = Calendar.current.dateComponents([.day], from: session.createdAt, to: Date()).day ?? 0
        if session.looks.isEmpty {
            let daysLeft = max(0, 7 - daysSince)
            if daysLeft == 0 {
                return (dateStr, "Expired", Theme.crimson)
            }
            return (dateStr, "\(daysLeft) days left", Theme.warmAmber)
        } else {
            return (dateStr, "✅ Completed", Theme.emerald)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header Row: Title + TOP PICK badge (Neon Cyan per design)
            HStack(alignment: .center) {
                Text(session.title)
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.white)

                Spacer()

                if let best = session.bestLook {
                    Text("TOP PICK: \(String(format: "%.1f", best.score))/10")
                        .font(.system(size: 11, weight: .black, design: .rounded))
                        .foregroundColor(.black)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(
                            Capsule().fill(
                                LinearGradient(
                                    colors: [Theme.neonCyan, Theme.electricBlue],
                                    startPoint: .leading, endPoint: .trailing
                                )
                            )
                        )
                        .shadow(color: Theme.neonCyan.opacity(0.5), radius: 6)
                }
            }

            // Horizontal Thumbnail Deck (real images or placeholder blocks)
            if !session.looks.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(session.looks) { look in
                            if let img = look.image {
                                ZStack(alignment: .bottomTrailing) {
                                    Image(uiImage: img)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(width: 76, height: 96)
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
                // Placeholder item-style blocks matching the design screenshot
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(0..<4, id: \.self) { _ in
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Theme.surfaceDark)
                                .frame(width: 76, height: 96)
                                .overlay(
                                    Image(systemName: "tshirt")
                                        .foregroundColor(.secondary)
                                )
                        }
                    }
                }
            }

            // Footer: Occasion pill + Date pill + Status pill + Total Looks
            let meta = sessionMeta
            HStack(spacing: 6) {
                // Occasion badge
                HStack(spacing: 4) {
                    Image(systemName: session.occasion.icon)
                        .font(.system(size: 9))
                    Text(session.occasion.rawValue.uppercased())
                        .font(.system(size: 9, weight: .bold, design: .rounded))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(session.occasion.color.opacity(0.85))
                .clipShape(Capsule())

                // Date pill (dark)
                Text(meta.date)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.12))
                    .clipShape(Capsule())

                // Status pill
                Text(meta.status)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(meta.statusColor)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(meta.statusColor.opacity(0.15))
                    .clipShape(Capsule())

                Spacer()

                Text("Total Looks: \(session.looks.count)")
                    .font(.caption2.bold())
                    .foregroundColor(.secondary)
            }

            // Tags
            Text(session.tagsFormatted)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.white.opacity(0.6))
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
