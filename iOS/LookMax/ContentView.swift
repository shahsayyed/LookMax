import SwiftUI

struct ContentView: View {
    @State private var profile: UserProfile? = UserProfile.load()
    @StateObject private var storage = SessionStorageManager.shared
    @State private var showingOnboarding = false
    @State private var showingCreateSession = false
    @State private var selectedSession: LookSession?
    @State private var selectedTab: Int = 0
    @State private var showingGlobalComparison = false
    @State private var savedLooksHidden = false
    @State private var dailyInspirationHidden = false

    var body: some View {
        NavigationView {
            ZStack(alignment: .bottom) {
                Theme.oledBlack.ignoresSafeArea()

                VStack(spacing: 0) {
                    // Top Bar / Brand
                    HStack {
                        Text("STYLED")
                            .font(.system(size: 15, weight: .black, design: .rounded))
                            .foregroundColor(.white)
                            .tracking(2.5)

                        Spacer()

                        Button(action: {
                            HapticManager.light()
                            showingOnboarding = true
                        }) {
                            Group {
                                if let p = profile,
                                   let data = p.photoDataList.first,
                                   let img = UIImage(data: data) {
                                    Image(uiImage: img)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(width: 34, height: 34)
                                        .clipShape(Circle())
                                        .overlay(Circle().stroke(Theme.neonCyan, lineWidth: 1.5))
                                } else {
                                    Image(systemName: "person.circle.fill")
                                        .font(.system(size: 28, weight: .medium))
                                        .foregroundColor(.white.opacity(0.8))
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 8)
                    .padding(.bottom, 6)

                    // Profile Header Banner
                    ProfileHeaderBanner(profile: $profile, showingOnboarding: $showingOnboarding)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 12)

                    // Sticky Start New Style Session CTA
                    StartSessionCTA(onAction: { showingCreateSession = true })
                        .padding(.horizontal, 16)
                        .padding(.bottom, 16)

                    // Main Content
                    if storage.sessions.isEmpty {
                        EmptySessionsView(onNew: { showingCreateSession = true })
                            .safeAreaInset(edge: .bottom, spacing: 0) {
                                bottomTabBar
                            }
                    } else {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 14) {
                                // ─── My Style Sessions ───
                                HStack {
                                    Text("MY STYLE SESSIONS")
                                        .font(.system(size: 12, weight: .bold, design: .rounded))
                                        .foregroundColor(.secondary)
                                        .tracking(0.8)
                                    Spacer()
                                }
                                .padding(.horizontal, 20)
                                .padding(.top, 4)

                                ForEach(storage.sessions) { session in
                                    SessionCardView(session: session)
                                        .onTapGesture {
                                            HapticManager.light()
                                            selectedSession = session
                                        }
                                        .padding(.horizontal, 16)
                                        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                            Button(role: .destructive) {
                                                HapticManager.medium()
                                                withAnimation {
                                                    SessionStorageManager.shared.deleteSession(session)
                                                }
                                            } label: {
                                                Label("Delete", systemImage: "trash.fill")
                                            }
                                        }
                                }

                                // ─── Saved Looks ───
                                HStack {
                                    Text("Saved Looks")
                                        .font(.system(size: 18, weight: .bold))
                                        .foregroundColor(.white)

                                    Spacer()

                                    Button(action: {
                                        withAnimation { savedLooksHidden.toggle() }
                                        HapticManager.light()
                                    }) {
                                        HStack(spacing: 2) {
                                            Text(savedLooksHidden ? "Show" : "Hide")
                                                .font(.subheadline)
                                                .foregroundColor(.secondary)
                                            Image(systemName: savedLooksHidden ? "chevron.down" : "chevron.right")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                }
                                .padding(.horizontal, 20)
                                .padding(.top, 10)

                                if !savedLooksHidden {
                                    // Placeholder saved looks row
                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 12) {
                                            ForEach(0..<3) { _ in
                                                RoundedRectangle(cornerRadius: 12)
                                                    .fill(Theme.surfaceDark)
                                                    .frame(width: 90, height: 110)
                                                    .overlay(
                                                        Image(systemName: "bookmark.fill")
                                                            .foregroundColor(Theme.neonCyan.opacity(0.4))
                                                            .font(.title2)
                                                    )
                                            }
                                        }
                                        .padding(.horizontal, 20)
                                    }
                                }

                                // ─── Daily Inspiration ───
                                HStack {
                                    Text("Daily Inspiration")
                                        .font(.system(size: 18, weight: .bold))
                                        .foregroundColor(.white)

                                    Spacer()

                                    Button(action: {
                                        withAnimation { dailyInspirationHidden.toggle() }
                                        HapticManager.light()
                                    }) {
                                        HStack(spacing: 2) {
                                            Text(dailyInspirationHidden ? "Show" : "Hide")
                                                .font(.subheadline)
                                                .foregroundColor(.secondary)
                                            Image(systemName: dailyInspirationHidden ? "chevron.down" : "chevron.right")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                }
                                .padding(.horizontal, 20)
                                .padding(.top, 6)

                                if !dailyInspirationHidden {
                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 12) {
                                            ForEach(0..<4) { _ in
                                                RoundedRectangle(cornerRadius: 12)
                                                    .fill(Theme.cardDark)
                                                    .frame(width: 130, height: 80)
                                                    .overlay(
                                                        Image(systemName: "sparkles")
                                                            .foregroundColor(Theme.neonCyan.opacity(0.4))
                                                            .font(.title2)
                                                    )
                                            }
                                        }
                                        .padding(.horizontal, 20)
                                    }
                                }
                            }
                            .padding(.bottom, 40)
                        }
                        .safeAreaInset(edge: .bottom, spacing: 0) {
                            bottomTabBar
                        }
                    }
                }
            }
            .navigationBarHidden(true)
            .sheet(isPresented: $showingOnboarding) {
                ProfileOnboardingView(profile: $profile)
            }
            .sheet(isPresented: $showingCreateSession) {
                CreateSessionSheet { newSession in
                    storage.addSession(newSession)
                    selectedSession = newSession
                }
            }
            .sheet(item: $selectedSession) { session in
                SessionDetailView(session: binding(for: session), profile: profile)
            }
            .sheet(isPresented: $showingGlobalComparison) {
                if let sess = storage.sessions.first(where: { $0.looks.count >= 2 }),
                   let l1 = sess.looks.first,
                   let l2 = sess.looks.last {
                    BeforeAfterComparisonView(look1: l1, look2: l2, occasion: sess.occasion)
                }
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            if profile == nil { showingOnboarding = true }
            seedSampleDataIfNeeded()
        }
    }

    private func binding(for session: LookSession) -> Binding<LookSession> {
        guard let idx = storage.sessions.firstIndex(where: { $0.id == session.id }) else {
            return .constant(session)
        }
        return $storage.sessions[idx]
    }

    // MARK: - 5-Tab Bottom Bar (matches design: Home, Discover, Sessions, Wardrobe, Profile)
    private var bottomTabBar: some View {
        HStack(spacing: 0) {
            tabItem(icon: "house.fill", title: "Home", tab: 0)
            tabItem(icon: "safari.fill", title: "Discover", tab: 1)
            tabItem(icon: "list.bullet.rectangle.fill", title: "Sessions", tab: 2)
            tabItem(icon: "tshirt.fill", title: "Wardrobe", tab: 3)
            tabItem(icon: "person.fill", title: "Profile", tab: 4)
        }
        .padding(.horizontal, 8)
        .padding(.top, 10)
        // Add safe area padding for devices with home indicator
        .padding(.bottom, UIApplication.shared.windows.first?.safeAreaInsets.bottom ?? 20)
        .background(
            Rectangle()
                .fill(Theme.cardDark.opacity(0.97))
                .background(.ultraThinMaterial)
                .overlay(
                    Rectangle()
                        .frame(height: 0.5)
                        .foregroundColor(Theme.cardBorder),
                    alignment: .top
                )
                .ignoresSafeArea(edges: .bottom)
        )
    }

    private func tabItem(icon: String, title: String, tab: Int) -> some View {
        let isSelected = selectedTab == tab
        return Button(action: {
            HapticManager.light()
            selectedTab = tab
            if tab == 1 { showingGlobalComparison = true }
        }) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 20, weight: isSelected ? .bold : .regular))
                    .foregroundColor(isSelected ? Theme.neonCyan : Color(white: 0.45))

                Text(title)
                    .font(.system(size: 10, weight: isSelected ? .semibold : .regular))
                    .foregroundColor(isSelected ? Theme.neonCyan : Color(white: 0.45))
            }
            .frame(maxWidth: .infinity)
        }
    }

    private func seedSampleDataIfNeeded() {
        if storage.sessions.isEmpty {
            let session1 = LookSession(title: "Business Meeting", occasion: .businessMeeting, createdAt: Date(), looks: [], tags: ["Formal", "Confident", "Sharp"])
            let session2 = LookSession(title: "Date Night", occasion: .dateNight, createdAt: Date().addingTimeInterval(-86400 * 2), looks: [], tags: ["Romantic", "Stylish", "Elegant"])
            storage.addSession(session2)
            storage.addSession(session1)
        }
    }
}

#Preview {
    ContentView()
}
