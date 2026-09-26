import PhotosUI
import SwiftUI

/// プロフィール編集モーダル（全画面）。プロフィール画面から presentation で出る。
///
/// **保存済みの値は直接いじらない。** 下書きを持ち、「保存」で確定して閉じ、
/// 「キャンセル」は何も書かずに閉じる。SNS の編集画面と同じ約束。
///
/// アバターをタップすると選択肢が出る。**アイコン生成はそのうちの 1 つ**で、
/// 「ライブラリから選択」と並ぶ。生成専用のアプリではなく、SNS の中の一機能として
/// 置いたときにどう見えるかを確かめるための形。
struct ProfileEditView: View {
    @Bindable var studio: IconStudio
    let profile: ProfileStore

    @Environment(\.dismiss) private var dismiss

    @State private var displayName = ""
    @State private var handle = ""
    @State private var bio = ""
    @State private var avatar: CGImage?

    @State private var showingAvatarOptions = false
    @State private var showingStudio = false
    @State private var pickingPhoto = false
    @State private var photoItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 0) {
                    avatarArea
                    fields
                }
            }
            .background(Color(.systemBackground))
            .navigationTitle("プロフィールを編集")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("キャンセル") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("保存") { save() }
                        .fontWeight(.semibold)
                        .disabled(!hasChanges)
                }
            }
        }
        .onAppear {
            displayName = profile.displayName
            handle = profile.handle
            bio = profile.bio
            avatar = profile.avatar
        }
        // アバターの差し替え口。SNS はここでアクションシートを出す
        .confirmationDialog("プロフィール画像", isPresented: $showingAvatarOptions,
                            titleVisibility: .visible) {
            Button("アイコンを生成…") { showingStudio = true }
            Button("ライブラリから選択") { pickingPhoto = true }
            if avatar != nil {
                Button("削除", role: .destructive) { avatar = nil }
            }
            Button("キャンセル", role: .cancel) {}
        }
        // 生成はさらに一段深いモーダル。**編集を中断せずに戻ってこられる**
        .sheet(isPresented: $showingStudio) {
            IconStudioSheet(studio: studio, avatar: $avatar)
        }
        .photosPicker(isPresented: $pickingPhoto, selection: $photoItem, matching: .images)
        .onChange(of: photoItem) { _, item in
            guard let item else { return }
            Task {
                guard let data = try? await item.loadTransferable(type: Data.self),
                      let image = ProfileStore.decode(data) else { return }
                avatar = image
            }
        }
    }

    // MARK: アバター

    private var avatarArea: some View {
        Button { showingAvatarOptions = true } label: {
            VStack(spacing: 10) {
                AvatarView(image: avatar, size: 96, transparent: studio.background == "none")
                    .overlay(Circle().stroke(Color(.separator), lineWidth: 1))
                    .overlay(alignment: .bottomTrailing) {
                        Image(systemName: "camera.fill")
                            .font(.caption.weight(.bold))
                            .foregroundStyle(.white)
                            .frame(width: 30, height: 30)
                            .background(Circle().fill(Color.accentColor))
                            .overlay(Circle().stroke(Color(.systemBackground), lineWidth: 2))
                    }
                Text("プロフィール画像を変更")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(Color.accentColor)
            }
        }
        .buttonStyle(.plain)
        .padding(.vertical, 24)
    }

    // MARK: 入力欄

    private var fields: some View {
        VStack(spacing: 0) {
            Divider()
            field("表示名") { TextField("表示名", text: $displayName) }
            Divider().padding(.leading, 104)
            field("ユーザー名") {
                HStack(spacing: 2) {
                    Text("@").foregroundStyle(.secondary)
                    TextField("ユーザー名", text: $handle)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                }
            }
            Divider().padding(.leading, 104)
            field("自己紹介") {
                TextField("自己紹介", text: $bio, axis: .vertical).lineLimit(1...5)
            }
            Divider()
        }
    }

    private func field<Content: View>(_ label: String,
                                      @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)
            content().font(.body)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    // MARK: 保存

    private var hasChanges: Bool {
        displayName != profile.displayName || handle != profile.handle
            || bio != profile.bio || avatar !== profile.avatar
    }

    private func save() {
        profile.commit(displayName: displayName, handle: handle, bio: bio, avatar: avatar)
        dismiss()
    }
}
