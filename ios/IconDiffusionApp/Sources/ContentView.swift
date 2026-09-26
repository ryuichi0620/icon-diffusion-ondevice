import SwiftUI

/// 起動したら出る唯一の画面。**画面遷移は無い。**
///
/// 中身は `ProfileEditView`（SNS のプロフィール編集欄を模した見た目）で、
/// 生成の本体は `IconStudio` が持つ。以前ここにあった検証用の画面（候補を
/// 並べて点を出すだけのもの）は、`ProfileEditView` の「開発用」に畳んである。
struct ContentView: View {
    @State private var studio = IconStudio()

    var body: some View {
        ProfileEditView(studio: studio)
    }
}
