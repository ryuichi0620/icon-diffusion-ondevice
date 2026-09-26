import IconDiffusionKit
import SwiftUI

/// SNS のプロフィール編集欄を模した 1 画面。**画面遷移は無い。**
///
/// この形にしたのは、このモデルの出口が「SNS のアイコン」だから。アイコンは
/// 丸く切り抜かれ、他人のタイムラインの地色の上に、40pt 程度で載る。
/// 生成の良し悪しはその条件で見ないと分からないので、
///
///   - プレビューは最初から丸（正方形で見ると輪郭の欠けを見落とす）
///   - 候補も丸（選ぶときの条件を、使うときの条件と揃える）
///   - 明るい地と暗い地の両方に並べて見せる（背景 auto の 3:1 はここで効く）
///
/// を既定にしてある。機械判定の点は画面に出さない（人の判定と 10 枚中 3 枚で
/// 食い違ったため）。開発用の欄にだけ残してある。
struct ProfileEditView: View {
    @Bindable var studio: IconStudio

    // 保存済みの値と、編集中の値を分ける。**「保存」と「キャンセル」に意味を
    // 持たせるため。** @AppStorage を直に編集すると打った端から確定してしまい、
    // キャンセルが何も戻さないボタンになる
    @AppStorage("profile.displayName") private var storedName = "えごし"
    @AppStorage("profile.handle") private var storedHandle = "egoshi"
    @AppStorage("profile.bio") private var storedBio = "オンデバイスでアイコンを作っています。"

    @State private var displayName = ""
    @State private var handle = ""
    @State private var bio = ""

    @FocusState private var subjectFocused: Bool
    @State private var savedAvatar: CGImage?
    @State private var toast: String?

    private var avatarImage: CGImage? { studio.selected?.cgImage ?? savedAvatar }

    var body: some View {
        ZStack(alignment: .top) {
            Color(.systemBackground).ignoresSafeArea()
            ScrollView {
                VStack(spacing: 0) {
                    header
                    cover
                    profileFields
                    sectionGap
                    studioSection
                    sectionGap
                    timelinePreview
                    sectionGap
                    DeveloperSection(studio: studio)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 32)
                }
            }
            if let toast {
                Text(toast)
                    .font(.footnote.weight(.medium))
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.top, 52)
                    .transition(.opacity)
            }
        }
        .task {
            displayName = storedName
            handle = storedHandle
            bio = storedBio
            savedAvatar = Self.loadSavedAvatar()
            await studio.start()
        }
        .onChange(of: studio.subject) { _, _ in studio.refreshGlyph() }
    }

    private var sectionGap: some View {
        Rectangle().fill(Color(.secondarySystemBackground)).frame(height: 10)
    }

    // MARK: ヘッダー

    private var header: some View {
        HStack {
            Button("キャンセル") { revert() }
                .disabled(!hasChanges)
            Spacer()
            Text("プロフィールを編集").font(.headline)
            Spacer()
            Button("保存") { save() }
                .fontWeight(.semibold)
                .disabled(!hasChanges)
        }
        .font(.subheadline)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .overlay(alignment: .bottom) { Divider() }
    }

    // MARK: カバーとアバター

    private var cover: some View {
        ZStack(alignment: .bottomLeading) {
            LinearGradient(colors: coverColours, startPoint: .topLeading, endPoint: .bottomTrailing)
                .frame(height: 104)
            avatarBadge
                .padding(.leading, 16)
                .offset(y: 38)
        }
        .padding(.bottom, 46)
    }

    /// カバーは選んだ 2 色から作る。色を変えたときに、アイコンだけでなく
    /// 「自分のページ全体がどう見えるか」が同時に動く
    private var coverColours: [Color] {
        studio.mono
            ? [studio.inkColor, studio.inkColor.opacity(0.45)]
            : [studio.inkColor, studio.accentColor]
    }

    private var avatarBadge: some View {
        Button {
            subjectFocused = true
        } label: {
            avatarCircle(size: 84)
                .overlay(Circle().stroke(Color(.systemBackground), lineWidth: 4))
                .overlay(alignment: .bottomTrailing) {
                    Image(systemName: "sparkles")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.white)
                        .frame(width: 26, height: 26)
                        .background(Circle().fill(Color.accentColor))
                        .overlay(Circle().stroke(Color(.systemBackground), lineWidth: 2))
                        .offset(x: 2, y: 2)
                }
        }
        .buttonStyle(.plain)
    }

    /// SNS と同じ丸抜き。透過のときだけ市松を敷く（透けていることを隠さない）
    @ViewBuilder private func avatarCircle(size: CGFloat) -> some View {
        Group {
            if let avatarImage {
                Image(decorative: avatarImage, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fill)
                    .background(studio.background == "none" ? AnyView(Checkerboard()) : AnyView(Color.white))
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

    // MARK: プロフィール欄

    private var profileFields: some View {
        VStack(spacing: 0) {
            field("表示名") { TextField("表示名", text: $displayName) }
            Divider().padding(.leading, 96)
            field("ユーザー名") {
                HStack(spacing: 2) {
                    Text("@").foregroundStyle(.secondary)
                    TextField("ユーザー名", text: $handle).autocorrectionDisabled()
                }
            }
            Divider().padding(.leading, 96)
            field("自己紹介") {
                TextField("自己紹介", text: $bio, axis: .vertical).lineLimit(1...3)
            }
        }
        .padding(.horizontal, 16)
    }

    private func field<Content: View>(_ label: String,
                                      @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(width: 72, alignment: .leading)
            content()
                .font(.body)
                .textFieldStyle(.plain)
        }
        .padding(.vertical, 12)
    }

    // MARK: アイコンをつくる

    private var studioSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("アイコンをつくる")
                .font(.subheadline.weight(.semibold))
            subjectRow
            colourRow
            backgroundRow
            generateRow
            if !studio.candidates.isEmpty || studio.isBusy { candidateRow }
        }
        .padding(16)
    }

    private var subjectRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("何を描く？（英語）", text: $studio.subject)
                .focused($subjectFocused)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
                .padding(.horizontal, 12).padding(.vertical, 10)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))

            // 表に無い語を打たせないための補完。引けない語は下書きなしに落ちる
            if studio.useControl, studio.resolvedGlyph == nil,
               let glyphs = studio.glyphs {
                let words = glyphs.suggestions(for: studio.subject)
                if !words.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(words, id: \.self) { word in
                                Button(word) { studio.subject = word }
                                    .font(.caption)
                                    .buttonStyle(.bordered)
                                    .buttonBorderShape(.capsule)
                            }
                        }
                    }
                }
            }

            HStack(spacing: 8) {
                // 下書き＝輪郭の下敷き。既定で入れる（実機の絵を人が見た判定で
                // なし 0/5、あり 3/5）。外せるようにしてあるのは比較のため
                Toggle(isOn: $studio.useControl) {
                    Text("形の下書きを使う").font(.footnote)
                }
                .toggleStyle(.switch)
                .fixedSize()
                Spacer()
            }
            Text(studio.controlStatus)
                .font(.caption2)
                .foregroundStyle(studio.resolvedGlyph == nil && studio.useControl ? .orange : .secondary)
        }
    }

    private var colourRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 14) {
                ColorPicker("主色", selection: $studio.inkColor, supportsOpacity: false)
                    .labelsHidden()
                if !studio.mono {
                    ColorPicker("副色", selection: $studio.accentColor, supportsOpacity: false)
                        .labelsHidden()
                }
                Toggle(isOn: $studio.mono) { Text("単色").font(.footnote) }
                    .toggleStyle(.button)
                    .buttonBorderShape(.capsule)
                Spacer()
            }
            colourWarning
        }
    }

    @ViewBuilder private var colourWarning: some View {
        let theme = studio.theme
        if !studio.mono && !theme.isUsable {
            // 2 色は必ず隣り合うので、明るさが近いと見分けられない
            Text(String(format: "2 色が近すぎます（%.1f:1）", theme.pairContrast))
                .font(.caption2).foregroundStyle(.orange)
        } else if !theme.backgroundAchievable {
            // 9 倍離れていないと、どの地を敷いても片方が 3:1 に届かない
            Text(String(format: "明暗の開きが %.1f 倍（地との 3:1 には 9 倍要ります）",
                        theme.luminanceSpread))
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var backgroundRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("背景", selection: $studio.background) {
                Text("自動").tag("auto")
                Text("白").tag("white")
                Text("主色").tag("ink")
                Text("透過").tag("none")
            }
            .pickerStyle(.segmented)
            Text("自動は、線の色に対して 3:1 の明暗差がつく地を選びます")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var generateRow: some View {
        VStack(spacing: 8) {
            Button {
                Task { await studio.generate() }
            } label: {
                Text(studio.isBusy ? "つくっています…" : "\(studio.candidateCount) 枚つくる")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(studio.isBusy || studio.subject.isEmpty)

            if studio.isBusy {
                // 1 枚 5 秒前後 × 4 枚。無反応に見せないための進捗
                ProgressView(value: studio.progress)
            }
            if let failure = studio.failure {
                // シミュレータでは Core AI が無いのでここに出る。実機専用であることを
                // 黙って握りつぶさない
                Text(failure)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var candidateRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                ForEach(0..<studio.candidateCount, id: \.self) { index in
                    candidateCell(index)
                }
            }
            if studio.selected != nil {
                Button {
                    studio.saveToPhotos()
                    flash("写真に保存しました")
                } label: {
                    Label("写真に保存", systemImage: "square.and.arrow.down")
                        .font(.footnote)
                }
                .buttonStyle(.bordered)
                .buttonBorderShape(.capsule)
            }
        }
    }

    /// 候補は**生成した順**に並べる。点の高い順にしないのは、その点が人の判定と
    /// 10 枚中 3 枚で食い違ったから。並べ替えると「上ほど良い」と誤解させる
    @ViewBuilder private func candidateCell(_ index: Int) -> some View {
        let candidate = studio.candidates.first { $0.id == index }
        let chosen = studio.selectedID == index
        Button {
            if candidate != nil { studio.select(index) }
        } label: {
            Group {
                if let candidate {
                    Image(decorative: candidate.cgImage, scale: 1)
                        .resizable()
                        .interpolation(.high)
                        .aspectRatio(contentMode: .fill)
                        .background(studio.background == "none" ? AnyView(Checkerboard()) : AnyView(Color.white))
                } else {
                    Color(.tertiarySystemFill)
                }
            }
            .frame(maxWidth: .infinity)
            .aspectRatio(1, contentMode: .fit)
            .clipShape(Circle())
            .overlay(Circle().stroke(chosen ? Color.accentColor : Color(.separator),
                                     lineWidth: chosen ? 3 : 1))
        }
        .buttonStyle(.plain)
        .disabled(candidate == nil)
    }

    // MARK: タイムラインでの見え方

    /// **明るい地と暗い地の両方に載せて見せる。** アイコンは自分のページではなく
    /// 他人のタイムラインで見られるもので、そこの地色はこちらでは選べない。
    /// 背景「透過」を選ぶと、ここで両方に溶けるのが見える
    private var timelinePreview: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("タイムラインでの見え方")
                .font(.subheadline.weight(.semibold))
            timelineRow(dark: false)
            timelineRow(dark: true)
        }
        .padding(16)
    }

    private func timelineRow(dark: Bool) -> some View {
        HStack(alignment: .top, spacing: 10) {
            avatarCircle(size: 40)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Text(displayName.isEmpty ? "表示名" : displayName)
                        .font(.footnote.weight(.semibold))
                    Text("@\(handle)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text(bio.isEmpty ? "投稿のテキストが 1 行ここに入ります。" : bio)
                    .font(.caption)
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .background(dark ? Color(white: 0.09) : Color(white: 0.97),
                    in: RoundedRectangle(cornerRadius: 12))
        .foregroundStyle(dark ? Color.white : Color.black)
        .environment(\.colorScheme, dark ? .dark : .light)
    }

    // MARK: 雑用

    /// 保存していない編集があるか。無ければ「保存」も「キャンセル」も押せない
    private var hasChanges: Bool {
        displayName != storedName || handle != storedHandle || bio != storedBio
            || studio.selected != nil
    }

    private func save() {
        storedName = displayName
        storedHandle = handle
        storedBio = bio
        if studio.selected != nil {
            guard studio.saveAvatar() else {
                flash("保存できませんでした")
                return
            }
            savedAvatar = studio.selected?.cgImage
        }
        flash("プロフィールを保存しました")
    }

    /// 編集を捨てて保存済みに戻す。生成した候補は残す（捨てると作り直しになる）が、
    /// 選択は外すのでアバターは保存済みの絵に戻る
    private func revert() {
        displayName = storedName
        handle = storedHandle
        bio = storedBio
        studio.selectedID = nil
    }

    private func flash(_ text: String) {
        withAnimation { toast = text }
        Task {
            try? await Task.sleep(for: .seconds(1.8))
            withAnimation { toast = nil }
        }
    }

    private static func loadSavedAvatar() -> CGImage? {
        guard let url = IconStudio.avatarURL,
              let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}

/// 開発用。製品の画面からは外したが、実機での検証にはこれが要る。
/// **点は人の判定と食い違う**ので、ここに畳んである
private struct DeveloperSection: View {
    @Bindable var studio: IconStudio
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 8) {
                Button("モデルを準備") { Task { await studio.prewarm() } }
                    .buttonStyle(.bordered)
                    .disabled(studio.isBusy || studio.didPrewarm)
                ForEach(studio.candidates) { candidate in
                    Text(String(format: "候補 %d: 形 %.2f・塊 %d・粗さ %.2f・%@",
                                candidate.id + 1, candidate.report.score,
                                candidate.report.parts, candidate.report.rough,
                                candidate.ok ? "合格" : "不合格"))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Text(studio.status)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.top, 8)
        } label: {
            Text("開発用").font(.caption).foregroundStyle(.secondary)
        }
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
