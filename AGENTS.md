# AGENTS.md

このリポジトリで作業するときの取り決め。

## このプロジェクトは何か

キャッチーでシンプルなアイコンを生成するモデルを作る。最終的に iPhone のオンデバイスで
動かすのが目標。基本方針は **「速くする LoRA は既製品を借り、アイコンらしさの LoRA だけ自分で作る」**。

全体像は [README.md](README.md)、企画の原本は [docs/plan.md](docs/plan.md)。

## いまの優先順位

**モデルを完成させることが先。iOS アプリ化は後回し。**

- 主線は Stage 1 → 2 → 3 → `generate.py`（`training/`）。入出力はフォルダで完結する
- Stage 4（焼き込み）・Stage 5（Core AI 変換）・`ios/` は端末に載せる段の話。いま触らなくてよい
- 完成の判定基準は [docs/verification.md](docs/verification.md) の手順 5

## 文書を書いたら Notion にも追加する

**まとまった文書を作ったら、`docs/` に置くだけでなく Notion のプロジェクトページに
子ページとして追加する。**

- 親ページ: [アイコン生成モデル — 少ステップ拡散モデルのオンデバイス化](https://app.notion.com/p/3c32bbb6de9d813783dbc708f4839a74)
- 既にある子ページ: 検証手順 / 設計メモ / データセット方針 / 実装状況と受け渡し / 実験ログ
- 既存文書を更新したときは、対応する子ページも更新する（新規作成して重複させない）
- Notion の記法は独自なので、書く前に `notion://docs/enhanced-markdown-spec` を読むこと。
  特に表は Markdown のパイプ記法ではなく `<table><tr><td>` を使う

## 変更したら通すもの

```bash
cd training && python3 scripts/smoke_test.py
```

極小モデルで Stage 3 → generate → Stage 4 → Stage 5 を通しで実行する。
ネットワークも実モデルも要らず数秒で終わるので、`training/` に手を入れたら必ず通す。

Swift 側を触ったら `cd ios/IconDiffusionKit && swift test`。

**Core AI まわりを触ったら `./ios/typecheck-coreai.sh` も通す。** `swift test` では
`CoreAIBackend.swift` はビルド対象外（`ICON_DIFFUSION_COREAI` が未定義）なので、
壊れていても気づけない。Core AI は iOS 27 以降なので Xcode 27 が要る。

## 書き方の約束

- **設定は `configs/icon_lora.toml` に集約する。** スクリプトの CLI 引数は「その場で試すための
  上書き」であって、恒久的な値をコードに散らさない
- **重い依存（torch / diffusers）は遅延インポートする。** `--help` と引数チェックが
  torch 無しで通る状態を保つ。`load_torch_stack()` を使うこと
- **確認できていないことは、確認できていないと書く。** Core AI の API は WWDC26 のもので
  実物に未検証なので `TODO(core-ai)` を付け、`CoreAIBackend.swift` は既定でビルド対象外に
  してある。推測で埋めて動くように見せない
- コメントは「何をしているか」ではなく「なぜそうしたか」を書く。特に、素朴な実装だと
  壊れる理由（例: 背景の透過で外周連結を見る理由、BPE で最小ランクを選ぶ理由）
