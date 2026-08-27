import SwiftUI

// MARK: - Sessions Dashboard (Root View)
struct ContentView: View {
    @State private var profile: UserProfile? = UserProfile.load()
    @StateObject private var storage = SessionStorageManager.shared
    @State private var showingOnboarding = false
    @State private var showingCreateSession = false
    @State private var selectedSession: LookSession?

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                ProfileHeaderBanner(profile: $profile, showingOnboarding: $showingOnboarding)
                    .padding(.horizontal)
                    .padding(.top, 12)
                    .padding(.bottom, 8)

                if storage.sessions.isEmpty {
                    EmptySessionsView(onNew: { showingCreateSession = true })
                } else {
                    ScrollView {
                        LazyVStack(spacing: 14) {
                            Button(action: { showingCreateSession = true }) {
                                Label("Start New Style Session", systemImage: "plus.circle.fill")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity).padding()
                                    .background(Color.blue).foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 14))
                            }
                            .padding(.horizontal).padding(.top, 8)

                            ForEach(storage.sessions) { session in
                                SessionCardView(session: session)
                                    .onTapGesture { selectedSession = session }
                                    .padding(.horizontal)
                            }
                        }
                        .padding(.bottom, 30)
                    }
                }
            }
            .navigationTitle("LookMax")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showingOnboarding = true }) {
                        Image(systemName: profile == nil
                              ? "person.crop.circle.badge.plus"
                              : "person.crop.circle.fill")
                    }
                }
            }
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
        }
        .onAppear {
            if profile == nil { showingOnboarding = true }
        }
    }

    private func binding(for session: LookSession) -> Binding<LookSession> {
        guard let idx = storage.sessions.firstIndex(where: { $0.id == session.id }) else {
            return .constant(session)
        }
        return $storage.sessions[idx]
    }
}

#Preview {
    ContentView()
}