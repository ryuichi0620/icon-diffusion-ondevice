# 学習データの出所

`docs/dataset.md` の「権利まわり」に従い、生成に使ったサービス・日付・プロンプトを残す。
画像本体はリポジトリに入れないので、ここが唯一の出所の記録になる。

## 生成条件

| 項目 | 内容 |
| --- | --- |
| サービス | OpenAI `gpt-image-2` (Images API) |
| 生成日 | 2026-08-22（64）/ 08-23（+90）/ 08-30（+219）/ 09-06（2 枚目の変奏） |
| サイズ / 品質 | 1024x1024 / `low` |
| 枚数 | **373 被写体 × 2 枚 = 746 枚** |
| 置き場所 | `training/data/raw/<subject>-01.png` と `-02.png` |

2 枚目を足したのは構造を学ばせるため。1 被写体 1 枚では持ち手の数や蓋の重なりを
一度しか見せられない（docs/experiments.md Trial 12）。プロンプトは 1 枚目と同一で、
seed が違うだけの変奏になっている。

市販アイコンセットは一切使っていない。既存ブランドのロゴを模した被写体も含めていない。

## プロンプト

被写体だけを差し替え、スタイル指定は全枚数で同一。

```
A minimal flat vector app icon of {subject}, centered in the frame,
occupying about 70% of the canvas with equal empty margin on all four sides,
on a pure solid white #FFFFFF background.
Use strictly flat solid fills in exactly two colors: deep navy #1F2A44 and coral #FF6B4A.
No outlines, no strokes, no gradients, no shadows, no highlights, no texture,
no 3D, no perspective, no text, no letters. Orthographic straight-on front view.
Bold simple geometric shapes, chunky and readable at small size.
Single object only, nothing else in frame.
```

パレットを 2 色に固定してあるのは `docs/dataset.md` の「揃えるべき軸」に合わせたため。
占有率を数値で書いているのは、パイロットで品質段によって余白が変わったため。

## 被写体

`prompts.example.txt` の 12 被写体のうち **8 つを学習に含め、4 つを意図的に外した**。

外したもの: `a rocket` / `an umbrella` / `a shopping cart` / `a magnifying glass`

全部入れてしまうと、手順 5 の生成結果を見たときに「スタイルを覚えた」のか
「その絵を覚えた」のかを区別できなくなる。外した 4 つが学習時と同じスタイルで
出てくれば、汎化したと言える。

含めた 8 つ: `a coffee cup` `a house key` `a calendar` `a paper airplane`
`a light bulb` `a shopping bag` `a camera` `an envelope`

残り 56 は日用品を中心に、2 色フラットで成立するシルエットのものを選んだ。
一覧は生成スクリプトの `EXTRA` を参照（`training/data/raw/` のファイル名がそのまま被写体名）。

## 利用規約について（2026-08-23 に調査。**判断が要る論点が残っている**）

### 分かったこと

**生成物の権利は利用者に帰属する。** OpenAI は Output の権利を利用者に譲渡する。
この点は複数の情報源で一致しており、争いは無さそう。

**ただし「OpenAI と競合するモデルの開発に Output を使ってはならない」という
制限がある。** 利用規約と Services Agreement の両方にこの趣旨の条項がある。

### 残っている論点

**このプロジェクトがその制限に触れるかは、判断が要る。**

- 作っているのは SD1.5 に載せるアイコン風スタイル LoRA で、汎用画像生成モデルではない
- 個人利用で、商用配布の予定も現時点では無い
- 一方で「画像生成モデルの学習に使っている」ことは事実

「競合するモデル」の解釈次第で、白とも黒とも読める。**法務判断が必要な場合は
専門家に確認すること。** ここで断定しない。

### 調査の限界（2026-08-27 に再調査して更新）

`openai.com/policies/row-terms-of-use/` は自動アクセスでは取得できない
（WebFetch は 403、実 Chrome の headless も Cloudflare の challenge で弾かれる）。
**規約ページそのものは開けていない**が、条項の文言は複数の独立ソースで一致して
引用されており、以下が原文と考えてよい:

> Use Output to develop models that compete with OpenAI.

（利用規約の「What you cannot do」節の禁止事項リストの一項。裏づけ:
Microsoft Q&A [learn.microsoft.com/en-us/answers/questions/1920477](https://learn.microsoft.com/en-us/answers/questions/1920477/are-there-any-restrictions-on-the-handling-of-outp)
と OpenAI 公式コミュニティ [community.openai.com/t/863706](https://community.openai.com/t/use-openai-for-generate-specific-task-instructions-for-ft/863706)
が同一文言を規約由来として引用。2026-08-27 確認）

最終確認はブラウザで
[openai.com/policies/row-terms-of-use/](https://openai.com/policies/row-terms-of-use/)
を人が開けば 1 分で済む。文言の存在自体はもう疑っていない。

### 補足（2026-08-27）

- **Azure OpenAI 経由なら適用されるのは Microsoft の規約**（OpenAI の規約ではない）。
  将来データを作り直す場合の選択肢として記録しておく
- 姉妹プロジェクト handdrawn-avatar-ondevice は **CC0 素材から学習データを作る**ため、
  この論点が最初から存在しない。アイコン側も同様の道（権利がクリーンな素材から
  データセットを再構築する）に切り替えれば論点ごと消せる

### 判断（2026-08-30・プロジェクトオーナー）

**A: 個人利用の範囲として現状維持。** 学習データは gpt-image-2 の出力を使い続ける。
配布・商用化を検討する段になったら、この判断を見直すこと（その時点での選択肢は
専門家への確認、または CC0 方式でのデータ再構築）。

### 影響を受ける範囲

学習データ 746 枚すべてが gpt-image-2 の出力。論点が黒と判断された場合、
**データセットを作り直す必要がある**（別サービス、または自前で描く）。
`docs/dataset.md` の「権利まわり」も合わせて見直すこと。
