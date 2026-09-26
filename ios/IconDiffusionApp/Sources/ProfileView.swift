import SwiftUI

/// 起動して最初に出る画面。**SNS のプロフィール画面そのもの。**
///
/// アイコン生成はここからは見えない。「プロフィールを編集」→ 編集モーダル →
/// アバターをタップ → アイコン生成モーダル、という**実際の SNS と同じ順路**を通る。
/// 生成したアイコンがこの画面でどう見えるか（丸・小さい・投稿の横）が最終的な判定基準なので、
/// 作る画面ではなく**使う画面**を入口に置いた。
struct ProfileView: View {
    @Bindable var studio: IconStudio
    @State private var profile = ProfileStore()
    @State private var editing = false
    @State private var tab = Tab.posts

    enum Tab: String, CaseIterable {
        case posts = "ポスト"
        case media = "メディア"
        case likes = "いいね"
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    headerArea
                    identity
                    metrics
                    tabBar
                    tabContent
                }
            }
            .background(Color(.systemBackground))
            .toolbar { toolbarContent }
            .navigationBarTitleDisplayMode(.inline)
        }
        // SNS の編集画面は全画面のモーダル。押した瞬間に別の場所へ来たと分かる
        .fullScreenCover(isPresented: $editing) {
            ProfileEditView(studio: studio, profile: profile)
        }
        .task { await studio.start() }
    }

    @ToolbarContentBuilder private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .principal) {
            VStack(spacing: 0) {
                Text(profile.displayName).font(.headline)
                Text("\(Self.posts.count) 件のポスト")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: ヘッダー

    private var headerArea: some View {
        ZStack(alignment: .topLeading) {
            LinearGradient(colors: coverColours, startPoint: .topLeading, endPoint: .bottomTrailing)
                .frame(height: 130)
            HStack(alignment: .bottom) {
                AvatarView(image: profile.avatar, size: 88, transparent: false)
                    .overlay(Circle().stroke(Color(.systemBackground), lineWidth: 4))
                Spacer()
                Button("プロフィールを編集") { editing = true }
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 14).padding(.vertical, 7)
                    .overlay(Capsule().stroke(Color(.separator), lineWidth: 1))
            }
            .padding(.horizontal, 16)
            .padding(.top, 86)
        }
        // アバターは 88pt でカバー（130pt）に 44pt 食い込む。下に出る 44pt ぶんだけ
        // 場所を空け、名前は詰めて置く（SNS のプロフィールはここが詰まっている）
        .padding(.bottom, 16)
    }

    /// カバーは選んだ 2 色から作る。アイコンの色を変えると**ページ全体の印象**が動く
    private var coverColours: [Color] {
        studio.mono
            ? [studio.inkColor, studio.inkColor.opacity(0.45)]
            : [studio.inkColor, studio.accentColor]
    }

    private var identity: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(profile.displayName).font(.title2.bold())
            Text("@\(profile.handle)").font(.subheadline).foregroundStyle(.secondary)
            if !profile.bio.isEmpty {
                Text(profile.bio).font(.subheadline).padding(.top, 6)
            }
            Label(joinedText, systemImage: "calendar")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.top, 6)
        }
        .padding(.horizontal, 16)
    }

    private var joinedText: String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "ja_JP")
        formatter.dateFormat = "y年M月"
        return "\(formatter.string(from: profile.joined))からご利用"
    }

    private var metrics: some View {
        HStack(spacing: 18) {
            metric(128, "フォロー中")
            metric(1_204, "フォロワー")
        }
        .padding(.horizontal, 16)
        .padding(.top, 10)
    }

    private func metric(_ value: Int, _ label: String) -> some View {
        HStack(spacing: 4) {
            Text(value.formatted()).font(.subheadline.weight(.semibold))
            Text(label).font(.subheadline).foregroundStyle(.secondary)
        }
    }

    // MARK: タブ

    private var tabBar: some View {
        HStack(spacing: 0) {
            ForEach(Tab.allCases, id: \.self) { item in
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { tab = item }
                } label: {
                    VStack(spacing: 8) {
                        Text(item.rawValue)
                            .font(.subheadline.weight(tab == item ? .semibold : .regular))
                            .foregroundStyle(tab == item ? Color.primary : Color.secondary)
                        Capsule()
                            .fill(tab == item ? Color.accentColor : .clear)
                            .frame(height: 3)
                    }
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity)
            }
        }
        .padding(.top, 14)
        .overlay(alignment: .bottom) { Divider() }
    }

    @ViewBuilder private var tabContent: some View {
        switch tab {
        case .posts:
            VStack(spacing: 0) {
                ForEach(Self.posts, id: \.body) { post in
                    PostRow(avatar: profile.avatar, name: profile.displayName,
                            handle: profile.handle, post: post)
                    Divider()
                }
            }
        case .media, .likes:
            // 作っていないものを「ある」ように見せない
            Text("まだありません")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 60)
        }
    }

    struct Post {
        let elapsed: String
        let body: String
        let replies: Int
        let reposts: Int
        let likes: Int
    }

    static let posts: [Post] = [
        Post(elapsed: "2時間", body: "アイコンを iPhone の中だけで作れるようにした。1 枚 5.4 秒。",
             replies: 3, reposts: 12, likes: 84),
        Post(elapsed: "昨日", body: "輪郭を下書きとして渡すと、被写体が何なのか分かる絵になる。渡さないと形は綺麗でも「何かわからない」になった。",
             replies: 1, reposts: 4, likes: 31),
        Post(elapsed: "3日", body: "アイコンは丸く切り抜かれて、他人のタイムラインの地色の上に載る。正方形で眺めていても良し悪しは分からない。",
             replies: 0, reposts: 2, likes: 26),
    ]
}

/// 投稿 1 件。**生成したアイコンが実寸（40pt・丸）で並ぶ**のがこの行の役目
private struct PostRow: View {
    let avatar: CGImage?
    let name: String
    let handle: String
    let post: ProfileView.Post

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            AvatarView(image: avatar, size: 40, transparent: false)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    Text(name).font(.subheadline.weight(.semibold))
                    Text("@\(handle) · \(post.elapsed)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Text(post.body).font(.subheadline)
                HStack(spacing: 28) {
                    action("bubble.left", post.replies)
                    action("arrow.2.squarepath", post.reposts)
                    action("heart", post.likes)
                }
                .padding(.top, 4)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func action(_ symbol: String, _ count: Int) -> some View {
        HStack(spacing: 5) {
            Image(systemName: symbol)
            Text(count.formatted())
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}

/// 丸いアイコン。**3 つの画面すべてがこれを使う**ので、見え方が食い違わない
struct AvatarView: View {
    let image: CGImage?
    let size: CGFloat
    /// 透過を選んでいるときだけ市松を敷く（透けていることを隠さない）
    var transparent: Bool = false

    var body: some View {
        Group {
            if let image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fill)
                    .background(transparent ? AnyView(Checkerboard()) : AnyView(Color.white))
            } else {
                ZStack {
                    Color(.tertiarySystemFill)
                    Image(systemName: "person.fill")
                        .font(.system(size: size * 0.45))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
    }
}

/// 透過確認用の市松背景
struct Checkerboard: View {
    var body: some View {
        Canvas { context, size in
            let cell: CGFloat = 8
            for row in 0..<Int(size.height / cell) + 1 {
                for column in 0..<Int(size.width / cell) + 1 {
                    if (row + column) % 2 == 0 {
                        context.fill(
                            Path(CGRect(x: CGFloat(column) * cell, y: CGFloat(row) * cell,
                                        width: cell, height: cell)),
                            with: .color(Color(white: 0.88))
                        )
                    }
                }
            }
        }
        .background(Color.white)
    }
}
