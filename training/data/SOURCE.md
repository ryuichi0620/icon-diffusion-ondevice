# 学習データの出所

`docs/dataset.md` の「権利まわり」に従い、生成に使ったサービス・日付・プロンプトを残す。
画像本体はリポジトリに入れないので、ここが唯一の出所の記録になる。

## 生成条件

| 項目 | 内容 |
| --- | --- |
| サービス | OpenAI `gpt-image-2` (Images API) |
| 生成日 | 2026-08-22 |
| サイズ / 品質 | 1024x1024 / `low` |
| 枚数 | 64 被写体 × 1 枚 |
| 置き場所 | `training/data/raw/<subject>-01.png` |

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

## 利用規約について

**未確認。** `docs/dataset.md` は「生成に使ったサービスの利用規約で『出力を学習に
使ってよいか』を確認しておくこと」と書いているが、この時点では OpenAI の規約原文に
当たっていない。学習を回す前に確認し、結果をここに書くこと。

確認すべき点は 2 つ:

- 生成物の権利が利用者に帰属するか
- その生成物を別のモデルの学習データとして使ってよいか
