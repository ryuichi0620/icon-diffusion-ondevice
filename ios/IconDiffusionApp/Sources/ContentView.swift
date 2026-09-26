import SwiftUI

/// 起動したら出る画面。**SNS のプロフィール画面。**
///
/// 順路は実際の SNS と同じ 3 層:
///
///   ProfileView（プロフィール）
///     └ ProfileEditView（編集・全画面モーダル）
///         └ IconStudioSheet（アイコン生成・シート）
///
/// 生成の本体は `IconStudio` が持ち、3 つの画面で共有する（モーダルを閉じても
/// 候補が消えない）。以前ここにあった検証用の画面は `IconStudioSheet` の
/// 「開発用」に畳んである。
struct ContentView: View {
    @State private var studio = IconStudio()

    var body: some View {
        ProfileView(studio: studio)
    }
}
