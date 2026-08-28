import SwiftUI

struct ContentView: View {
    @State private var profile: UserProfile? = UserProfile.load()
    @StateObject private var storage = SessionStorageManager.shared
    @State private var showingOnboarding = false
    @State private var showingCreateSession = false
    @State private var selectedSession: LookSession?
    @State private var selectedTab: Int = 0
    @State private var showingGlobalComparison = false

    var body: some View {
        NavigationView {
            ZStack(alignment: .bottom) {
                Theme.oledBlack.ignoresSafeArea()

                VStack(spacing: 0) {
                    // Top Bar / Brand
                    HStack {
                        Text("STYLED")
                            .font(.system(size: 13, weight: .black, design: .rounded))
                            .foregroundColor(.white)
                            .tracking(2.0)

                        Spacer()

                        Button(action: {
                            HapticManager.light()
                            showingOnboarding = true
                        }) {
                            Image(systemName: "person.crop.circle.badge.checkmark")
                                .font(.system(size: 18))
                                .foregroundColor(Theme.neonCyan)
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
                    } else {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 14) {
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
                                }
                            }
                            .padding(.bottom, 100)
                        }
                    }
                }

                // Bottom Tab Bar
                bottomTabBar
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
            if profile == nil {
                showingOnboarding = true
            }
            seedSampleDataIfNeeded()
        }
    }

    private func binding(for session: LookSession) -> Binding<LookSession> {
        guard let idx = storage.sessions.firstIndex(where: { $0.id == session.id }) else {
            return .constant(session)
        }
        return $storage.sessions[idx]
    }

    // MARK: - Bottom Tab Bar
    private var bottomTabBar: some View {
        HStack(spacing: 0) {
            tabItem(icon: "house.fill", title: "Home", tab: 0)
            tabItem(icon: "slider.horizontal.below.rectangle", title: "Compare", tab: 1)
            tabItem(icon: "bookmark.fill", title: "Saved", tab: 2)
            tabItem(icon: "sparkles", title: "Inspiration", tab: 3)
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 24)
        .background(
            RoundedRectangle(cornerRadius: 24)
                .fill(Theme.cardDark.opacity(0.95))
                .background(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 24)
                        .stroke(Theme.cardBorder, lineWidth: 1)
                )
        )
        .padding(.horizontal, 16)
    }

    private func tabItem(icon: String, title: String, tab: Int) -> some View {
        let isSelected = selectedTab == tab
        return Button(action: {
            HapticManager.light()
            selectedTab = tab
            if tab == 1 {
                showingGlobalComparison = true
            }
        }) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 18, weight: isSelected ? .bold : .medium))
                    .foregroundColor(isSelected ? Theme.neonCyan : .secondary)

                Text(title)
                    .font(.system(size: 10, weight: isSelected ? .bold : .regular))
                    .foregroundColor(isSelected ? Theme.neonCyan : .secondary)
            }
            .frame(maxWidth: .infinity)
        }
    }

    // Initial sample session setup for preview & testing
    private func seedSampleDataIfNeeded() {
        if storage.sessions.isEmpty {
            let session1 = LookSession(
                title: "Business Meeting",
                occasion: .businessMeeting,
                createdAt: Date(),
                looks: [],
                tags: ["Formal", "Confident", "Sharp"]
            )
            let session2 = LookSession(
                title: "Date Night",
                occasion: .dateNight,
                createdAt: Date().addingTimeInterval(-86400 * 2),
                looks: [],
                tags: ["Romantic", "Stylish", "Elegant"]
            )
            storage.addSession(session2)
            storage.addSession(session1)
        }
    }
}

#Preview {
    ContentView()
}
