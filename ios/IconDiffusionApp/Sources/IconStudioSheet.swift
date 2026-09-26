import IconDiffusionKit
import SwiftUI

/// アイコン生成モーダル。編集画面で「アイコンを生成…」を選ぶと出る。
///
/// **出口が SNS のアイコンなので、選ぶときの条件を使うときの条件に揃える。**
///
///   - プレビューも候補も丸（正方形で見ると輪郭の欠けを見落とす）
///   - 明るい地と暗い地の両方に並べる（他人のタイムラインの地色は選べない）
///   - 機械判定の点は出さない（人の判定と 10 枚中 3 枚で食い違った。開発用の欄にだけ）
///
/// 閉じ方は 2 つ。「使う」は選んだ絵を編集画面へ渡すだけで、**保存はしない**
/// （保存するかどうかは編集画面の「保存」が決める）。
struct IconStudioSheet: View {
    @Bindable var studio: IconStudio
    /// 選んだ絵を編集画面へ返す
    let onChoose: (CGImage) -> Void

    @Environment(\.dismiss) private var dismiss
    @FocusState private var subjectFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // 作る前は「何を描く？」が最初に来るべきで、空の丸ではない。
                    // プレビューは見るものが出てきてから現れる
                    if studio.selected != nil { preview }
                    if !studio.candidates.isEmpty || studio.isBusy { candidateRow }
                    subjectSection
                    colourSection
                    backgroundSection
                    generateSection
                    DeveloperSection(studio: studio)
                }
                .padding(16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("アイコンを生成")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("キャンセル") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("使う") {
                        if let image = studio.selected?.cgImage { onChoose(image) }
                        dismiss()
                    }
                    .fontWeight(.semibold)
                    .disabled(studio.selected == nil)
                }
            }
        }
    }

    // MARK: プレビュー

    private var preview: some View {
        VStack(spacing: 14) {
            AvatarView(image: studio.selected?.cgImage, size: 132,
                       transparent: studio.background == "none")
                .overlay(Circle().stroke(Color(.separator), lineWidth: 1))
                .frame(maxWidth: .infinity)
            timelineRow(dark: false)
            timelineRow(dark: true)
            Button {
                studio.saveToPhotos()
            } label: {
                Label("写真に保存", systemImage: "square.and.arrow.down").font(.footnote)
            }
            .buttonStyle(.bordered)
            .buttonBorderShape(.capsule)
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16))
    }

    /// **アイコンは他人のタイムラインで見られる。** そこの地色はこちらでは選べないので、
    /// 明暗の両方に載せて見せる。背景「透過」を選ぶと、ここで片方に溶けるのが見える
    private func timelineRow(dark: Bool) -> some View {
        HStack(spacing: 10) {
            AvatarView(image: studio.selected?.cgImage, size: 36,
                       transparent: studio.background == "none")
            VStack(alignment: .leading, spacing: 2) {
                Text(dark ? "暗い背景のタイムライン" : "明るい背景のタイムライン")
                    .font(.caption.weight(.semibold))
                Text("この大きさで人の目に触れます")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(dark ? Color(white: 0.09) : Color(white: 0.97),
                    in: RoundedRectangle(cornerRadius: 10))
        .foregroundStyle(dark ? Color.white : Color.black)
        .environment(\.colorScheme, dark ? .dark : .light)
    }

    // MARK: 候補

    /// 候補は**生成した順**に並べる。点の高い順にしないのは、その点が人の判定と
    /// 10 枚中 3 枚で食い違ったから。並べ替えると「上ほど良い」と誤解させる
    private var candidateRow: some View {
        HStack(spacing: 10) {
            ForEach(0..<studio.candidateCount, id: \.self) { index in
                candidateCell(index)
            }
        }
    }

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
                        .background(studio.background == "none"
                                    ? AnyView(Checkerboard()) : AnyView(Color.white))
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

    // MARK: 入力

    private var subjectSection: some View {
        section("何を描く？") {
            VStack(alignment: .leading, spacing: 10) {
                TextField("英語で（例: a coffee cup）", text: $studio.subject)
                    .focused($subjectFocused)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .padding(.horizontal, 12).padding(.vertical, 10)
                    .background(Color(.tertiarySystemFill), in: RoundedRectangle(cornerRadius: 10))
                    .onChange(of: studio.subject) { _, _ in studio.refreshGlyph() }

                // 表に無い語を打たせないための補完。引けない語は下書きなしに落ちる
                if studio.useControl, studio.resolvedGlyph == nil, let glyphs = studio.glyphs {
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

                // 下書き＝輪郭の下敷き。既定で入れる（人が見た判定で なし 0/5・あり 3/5）。
                // 外せるようにしてあるのは比較のため
                Toggle(isOn: $studio.useControl) {
                    Text("形の下書きを使う").font(.subheadline)
                }
                Text(studio.controlStatus)
                    .font(.caption2)
                    .foregroundStyle(studio.useControl && studio.resolvedGlyph == nil
                                     ? .orange : .secondary)
            }
        }
    }

    private var colourSection: some View {
        section("色") {
            VStack(alignment: .leading, spacing: 8) {
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

    private var backgroundSection: some View {
        section("背景") {
            VStack(alignment: .leading, spacing: 6) {
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
    }

    private var generateSection: some View {
        VStack(spacing: 8) {
            Button {
                subjectFocused = false
                Task { await studio.generate() }
            } label: {
                Text(buttonLabel).frame(maxWidth: .infinity)
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

    private var buttonLabel: String {
        if studio.isBusy { return "つくっています…" }
        return studio.candidates.isEmpty
            ? "\(studio.candidateCount) 枚つくる" : "もう一度つくる"
    }

    private func section<Content: View>(_ title: String,
                                        @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.subheadline.weight(.semibold))
            content()
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground),
                    in: RoundedRectangle(cornerRadius: 16))
    }
}

/// 開発用。製品の画面からは外したが、実機での検証にはこれが要る。
/// **点は人の判定と食い違う**ので、ここに畳んである
struct DeveloperSection: View {
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
        .padding(.horizontal, 4)
    }
}
