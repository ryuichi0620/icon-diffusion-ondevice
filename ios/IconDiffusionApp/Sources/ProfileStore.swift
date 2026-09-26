import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// プロフィールの保存された中身。**3 つの画面（プロフィール / 編集 / アイコン生成）が
/// 同じ値を見る**ので、画面ごとに @AppStorage を持たせず 1 か所に集めた。
///
/// 編集画面は**この値を直接いじらない**。下書きを @State で持ち、「保存」で
/// `commit` する。SNS の編集画面と同じで、キャンセルが本当に取り消しになる。
@MainActor
@Observable
final class ProfileStore {
    var displayName: String
    var handle: String
    var bio: String
    private(set) var avatar: CGImage?
    /// 「2026年8月からご利用」の表示用。初回起動の日を覚えておく
    let joined: Date

    private enum Key {
        static let name = "profile.displayName"
        static let handle = "profile.handle"
        static let bio = "profile.bio"
        static let joined = "profile.joined"
    }

    init() {
        let defaults = UserDefaults.standard
        displayName = defaults.string(forKey: Key.name) ?? "こむぎ"
        handle = defaults.string(forKey: Key.handle) ?? "_komugi0"
        bio = defaults.string(forKey: Key.bio) ?? "オンデバイスでアイコンを作っています。iPhone の中だけで完結する拡散モデルを実験中。"
        if let stored = defaults.object(forKey: Key.joined) as? Date {
            joined = stored
        } else {
            joined = Date()
            defaults.set(joined, forKey: Key.joined)
        }
        avatar = Self.loadAvatar()
    }

    func commit(displayName: String, handle: String, bio: String, avatar: CGImage?) {
        self.displayName = displayName
        self.handle = handle
        self.bio = bio
        let defaults = UserDefaults.standard
        defaults.set(displayName, forKey: Key.name)
        defaults.set(handle, forKey: Key.handle)
        defaults.set(bio, forKey: Key.bio)
        if let avatar, avatar !== self.avatar {
            self.avatar = avatar
            saveAvatar(avatar)
        }
    }

    // MARK: 画像の永続化

    static var avatarURL: URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("avatar.png")
    }

    private static func loadAvatar() -> CGImage? {
        guard let url = avatarURL,
              let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    private func saveAvatar(_ image: CGImage) {
        guard let url = Self.avatarURL, let data = IconStudio.pngData(image) else { return }
        try? data.write(to: url, options: .atomic)
    }

    /// 写真アプリから選んだ画像を CGImage にする。**アイコン生成だけが入口ではない**
    /// ので、SNS と同じく手持ちの写真も選べるようにしてある
    static func decode(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}
