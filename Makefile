# 各ステージの入口。実体は training/scripts と ios/IconDiffusionKit にある。
.DEFAULT_GOAL := help
PY      ?= python3
TRAINING := training
INPUT   ?= data/raw

.PHONY: help setup dataset train preview fuse export check-export test clean

help:  ## このヘルプを表示する
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  例: make dataset INPUT=~/Downloads/icons_raw"

setup:  ## Python の依存を入れる
	cd $(TRAINING) && $(PY) -m pip install -r requirements.txt

dataset:  ## Stage 1: アイコン画像を 512x512 + captions に整形する
	cd $(TRAINING) && $(PY) scripts/01_prepare_dataset.py --input $(INPUT)

train:  ## Stage 2: スタイル LoRA を学習する
	cd $(TRAINING) && ./scripts/02_train_lora.sh

preview:  ## Stage 3: LCM を重ねて 4 ステップ生成を確認する（最初の山）
	cd $(TRAINING) && $(PY) scripts/03_preview_lcm.py

fuse:  ## Stage 4: LoRA を UNet に焼き込む
	cd $(TRAINING) && $(PY) scripts/04_fuse_lora.py

check-export:  ## Stage 5 の前段: Core AI 無しで 3 モデルの形だけ確認する
	cd $(TRAINING) && $(PY) scripts/05_export_coreai.py --check-only

export:  ## Stage 5: Core AI 形式（3 モデル）へ変換する
	cd $(TRAINING) && $(PY) scripts/05_export_coreai.py

test:  ## Stage 6: Swift 側のテスト（Core AI もモデルも不要）
	cd ios/IconDiffusionKit && swift test

clean:  ## 生成物を消す（data/raw は残す）
	rm -rf $(TRAINING)/data/processed $(TRAINING)/outputs/* $(TRAINING)/.vendor
	rm -rf ios/IconDiffusionKit/.build
