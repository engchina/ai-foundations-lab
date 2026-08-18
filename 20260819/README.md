# PDF / 画像レイアウト比較ラボ

PDF または画像をアップロードし、OCI Document Understanding、MinerU、dots.mocr、Unstructured、Docling、PyMuPDF、YOLOv10、PP-DocLayoutV3 のレイアウト解析結果をページごとに比較するための Gradio アプリです。React ビューアは `pdf-jsonl-viewer` と同じ発想で、JSONL の行をクリックすると対応する bbox をページ上で強調表示します。

## セットアップ

Python は 3.12 を推奨します。仮想環境は `uv` でプロジェクト直下の `.venv` に作成します。既存の `.venv` が別の Python や Conda 由来で作成されている場合は、削除してから作り直してください。

最小構成では Gradio、PyMuPDF、React ビューアだけをインストールします。OCI、MinerU、dots.mocr、Unstructured、Docling、YOLOv10、PP-DocLayoutV3 などの重いエンジンは必要なものだけ追加します。

```bash
rm -rf .venv
uv --cache-dir /tmp/uv-cache-pdf-layout-lab venv --python 3.12 .venv
source .venv/bin/activate
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -r requirements.txt
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python pip

cd viewer
npm install
npm run build
cd ..

cp .env.example .env
```

OCI、MinerU、dots.mocr、Unstructured、Docling、YOLOv10、PP-DocLayoutV3 は追加の依存関係やモデルが必要です。未設定の場合、アプリ上では該当エンジンが無効として表示されます。

## WSL Ubuntu でのデプロイ

Windows 上の WSL Ubuntu で動かす場合は、Ubuntu 24.04 など Python 3.12 を使える環境を推奨します。Python 3.12 がない場合は、OS を更新するか pyenv などで Python 3.12 を用意してください。`uv` も事前にインストールし、`uv --version` が実行できる状態にします。

```bash
sudo apt update
sudo apt install -y git build-essential curl python3.12 python3.12-venv

python3.12 --version
uv --version
```

React ビューアのビルドには Node.js `20.19` 以上、または `22.12` 以上が必要です。Ubuntu 標準パッケージの `nodejs` が古い場合は、nvm、Volta、NodeSource などで新しい Node.js を入れてください。

```bash
node --version
npm --version
```

リポジトリを配置したディレクトリで、バックエンドと React ビューアをセットアップします。

```bash
rm -rf .venv
uv --cache-dir /tmp/uv-cache-pdf-layout-lab venv --python 3.12 .venv
source .venv/bin/activate
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -r requirements.txt
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python pip

cd viewer
npm install
npm run build
cd ..

cp .env.example .env
```

WSL 内だけで使う場合は既定の `127.0.0.1:7860` のままで起動できます。Windows 側のブラウザから開く場合も、通常は `http://127.0.0.1:7860` でアクセスできます。

```bash
source .venv/bin/activate
.venv/bin/python app.py
```

同じ LAN 上の別端末からアクセスさせたい場合は、`.env` で host を `0.0.0.0` に変更し、Windows Defender Firewall で該当ポートを許可してください。

```bash
PDF_LAYOUT_LAB_HOST=0.0.0.0
PDF_LAYOUT_LAB_PORT=7860
```

## macOS でのデプロイ

macOS では Homebrew で Python 3.12、uv、Node.js を用意します。Apple Silicon / Intel Mac のどちらでも同じ手順で進められます。

```bash
brew install python@3.12 uv node

python3.12 --version
uv --version
node --version
npm --version
```

React ビューアのビルドには Node.js `20.19` 以上、または `22.12` 以上が必要です。`node --version` が古い場合は `brew upgrade node` で更新してください。

リポジトリを配置したディレクトリで、仮想環境を作成して依存関係を入れます。

```bash
rm -rf .venv
uv --cache-dir /tmp/uv-cache-pdf-layout-lab venv --python 3.12 .venv
source .venv/bin/activate
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -r requirements.txt
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python pip

cd viewer
npm install
npm run build
cd ..

cp .env.example .env
```

ローカル利用では既定の `127.0.0.1:7860` のままで十分です。起動後、ブラウザで `http://127.0.0.1:7860` を開きます。

```bash
source .venv/bin/activate
.venv/bin/python app.py
```

LAN 内の別端末からアクセスする場合は、`.env` で host を `0.0.0.0` に変更し、macOS のファイアウォール設定で Python からの受信接続を許可してください。

```bash
PDF_LAYOUT_LAB_HOST=0.0.0.0
PDF_LAYOUT_LAB_PORT=7860
```

## 追加エンジンのインストール

以下のコマンドは、この README があるプロジェクト直下で実行してください。`uv` で `.venv/bin/python` を明示しているため、仮想環境を有効化していない場合でも同じ環境へ入ります。

### まとめてインストール

```bash
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[dots,oci,unstructured,docling,pp-doclayout,paddle,yolo,mineru,dev]' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

PyTorch CPU index と PyPI の両方を使うため、`uv` の index 解決では `--index-strategy unsafe-best-match` を指定します。これを省くと、PyTorch CPU index 側に古い `tqdm` などが見つかった時点で PyPI の新しい互換バージョンが候補から外れ、MinerU などの依存解決に失敗する場合があります。

### 個別にインストール

| エンジン | コマンド | 追加設定 |
| --- | --- | --- |
| dots.mocr | `uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[dots]' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match` | 既定では Transformers でローカル実行します。必要に応じて `DOTS_MOCR_MODEL` にローカル重みディレクトリを指定します。 |
| OCI Document Understanding | `uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[oci]'` | `.env` に `OCI_COMPARTMENT_ID`、必要に応じて `OCI_CONFIG_FILE` と `OCI_PROFILE` を設定します。 |
| Unstructured | `uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[unstructured]'` | high-res partition 用の依存が入ります。環境によっては OCR 関連のシステムパッケージが別途必要です。 |
| Docling | `uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[docling]' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match` | CPU 実行では `DOCLING_DEVICE=cpu` を使います。初回実行時にモデルや追加データの取得が発生する場合があります。 |
| PP-DocLayoutV3 | `uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[pp-doclayout]'` | CPU 向けに PaddleOCR の `onnxruntime` engine で実行します。既定は `PP_DOCLAYOUT_MODEL=PP-DocLayoutV3`、`PP_DOCLAYOUT_ENGINE=onnxruntime` です。 |
| PP-DocLayoutV3 (PaddlePaddle) | `uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[paddle]'` | PaddlePaddle の `paddle_static` / `paddle_dynamic` engine で実行したい場合に使います。 |
| YOLOv10 DocLayNet | `uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[yolo]' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match` | `YOLOV10_MODEL_PATH=models/yolov10x_best.pt` に重みファイルを配置するか、`.env` でパスを変更します。 |
| MinerU / MinerU2.5-Pro | `uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[mineru]' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match` | CPU 実行向けに `mineru[core]` を入れ、`MINERU_BACKEND=pipeline` を使います。初回実行時にモデル取得が発生する場合があります。 |

### dots.mocr

dots.mocr は既定で外部 API ではなく、同じ Python プロセス内の Transformers 推論としてローカル実行します。

```bash
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[dots]' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

既定設定:

```bash
DOTS_MOCR_BACKEND=transformers
DOTS_MOCR_MODEL=rednote-hilab/dots.mocr
DOTS_MOCR_DEVICE=auto
DOTS_MOCR_DEVICE_MAP=
DOTS_MOCR_TORCH_DTYPE=auto
DOTS_MOCR_ATTN_IMPLEMENTATION=auto
DOTS_MOCR_MAX_NEW_TOKENS=24000
```

`DOTS_MOCR_MODEL` には Hugging Face の model id、または事前にダウンロードしたローカル重みディレクトリを指定できます。ローカルディレクトリを使う場合は、例として `./weights/DotsMOCR` のように指定します。初回実行時にモデルを取得する場合、キャッシュは `.runs/_cache/dots_mocr` 配下へ保存します。

`DOTS_MOCR_ATTN_IMPLEMENTATION=auto` では、CUDA と `flash_attn` が利用できる場合だけモデル既定の flash attention を使い、それ以外は `sdpa` に切り替えます。`flash_attention_2` を明示する場合は、実行環境に合う `flash_attn` を別途インストールしてください。

すでに同じマシンで vLLM / SGLang などの OpenAI 互換サーバーを起動している場合だけ、API モードに切り替えます。

```bash
DOTS_MOCR_BACKEND=api
DOTS_MOCR_BASE_URL=http://127.0.0.1:8000/v1
DOTS_MOCR_MODEL=rednote-hilab/dots.mocr
```

### OCI Document Understanding

OCI config/profile 認証を使います。`/root/.oci/config` の `DEFAULT` profile を使う場合は、`.env` に次のように設定します。

```bash
OCI_CONFIG_FILE=/root/.oci/config
OCI_PROFILE=DEFAULT
OCI_COMPARTMENT_ID=ocid1.compartment.oc1..example
OCI_DOCUMENT_LANGUAGE=auto
```

`OCI_COMPARTMENT_ID` には Document Understanding を実行する compartment の OCID を指定してください。`OCI_DOCUMENT_LANGUAGE=auto` または空の場合は `language` を指定せず、OCI 側の判定に任せます。日本語だけを明示したい場合は `OCI_DOCUMENT_LANGUAGE=JA` を指定します。日本語 PDF に `en` を強制すると OCR 結果が英数字の断片になる場合があります。

アプリは起動時ではなく解析時に OCI API を呼び出します。profile や key file を読み込めない場合は、エンジン状態に日本語の確認メッセージを表示します。OCI の日本語 OCR では `confidence=-1` が返る場合があるため、これは「信頼度なし」として扱い、信頼度フィルタでは除外しません。

同期 `AnalyzeDocument` では入力 PDF のページ数制限があるため、アプリは指定された解析ページだけを一時 PDF に切り出してから OCI へ送信します。複数ページを指定した場合は 5 ページ単位で batch 実行し、OCI 応答のページ番号を元 PDF のページ番号へ戻して JSONL とビューアへ出力します。

OCI Document Understanding の同期入力は 5 ページ以下、かつ 8 MB 以下です。画像アップロードやスキャン PDF では、一時 PDF が元画像より大きくなる場合があります。その場合、アプリは単一ページの画像入力へ自動的に切り替え、必要に応じて JPEG 圧縮します。それでも 8 MB を超える場合は、ページ画像 DPI を下げて再実行してください。

### Docling

CPU だけで Docling を実行する場合は、Docling 公式の案内に合わせて PyTorch CPU index を併用します。

```bash
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[docling]' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

既定設定:

```bash
DOCLING_DEVICE=cpu
DOCLING_NUM_THREADS=4
DOCLING_DO_OCR=true
DOCLING_DO_TABLE_STRUCTURE=true
```

アプリ内では `AcceleratorOptions(device=CPU)` を明示し、OCR は CPU で扱いやすい RapidOCR を使います。Docling、PyTorch、OCR 関連のキャッシュは `.runs/_cache/docling` 配下へ保存します。

### PP-DocLayoutV3

CPU だけで実行する場合は、PaddleOCR の `onnxruntime` engine を使います。

```bash
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[pp-doclayout]'
```

既定設定:

```bash
PP_DOCLAYOUT_MODEL=PP-DocLayoutV3
PP_DOCLAYOUT_ENGINE=onnxruntime
PP_DOCLAYOUT_MODEL_SOURCE=BOS
```

`PP_DOCLAYOUT_MODEL_SOURCE=BOS` は、Hugging Face に接続できない環境でも PaddleOCR 公式モデルを取得しやすくするための設定です。PaddlePaddle の `paddle_static` engine を使う場合は `uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[paddle]'` を実行し、`PP_DOCLAYOUT_ENGINE=paddle_static` に変更してください。

### MinerU

MinerU は既存の JSON 出力取り込み、またはローカル CLI 実行に対応しています。CPU だけで実行する場合は、現行 MinerU CLI の `-b pipeline` を使います。

CPU 実行用に MinerU をインストールする場合:

```bash
uv --cache-dir /tmp/uv-cache-pdf-layout-lab pip install --python .venv/bin/python -e '.[mineru]' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

この extra は `mineru[core]` を使います。`mineru[all]` は vLLM などの GPU/VLM 加速用依存も含むため、CPU だけで確認する場合は避けてください。

既存出力を取り込む場合:

```bash
MINERU_OUTPUT_DIR=/path/to/mineru/output
```

CLI を実行する場合:

```bash
MINERU_COMMAND=auto
MINERU_BACKEND=pipeline
```

`MINERU_COMMAND=auto` の場合は、`.venv/bin/mineru`、`mineru`、`.venv/bin/magic-pdf`、`magic-pdf` の順に検出します。旧 `magic-pdf` CLI を使う場合は `MINERU_METHOD=auto` を `-m auto` として渡します。追加オプションが必要な場合は `MINERU_EXTRA_ARGS` に指定してください。

MinerU のモデルや周辺ライブラリのキャッシュは、ホームディレクトリ権限に依存しないように `.runs/_cache/mineru` 配下へ保存します。初回実行はモデル取得のため時間がかかりますが、2 回目以降は同じキャッシュを再利用します。

MinerU の `model.json`、`middle.json`、`content_list.json`、`content_list_v2.json` のいずれかを検出して、bbox 比較ビューアへ取り込みます。

### YOLOv10 の重み

`moured/YOLOv10-Document-Layout-Analysis` の `yolov10x_best.pt` を利用する場合は、例えば次のように配置します。

```bash
mkdir -p models
# yolov10x_best.pt を models/yolov10x_best.pt に保存
```

保存場所を変える場合は `.env` を更新します。

```bash
YOLOV10_MODEL_PATH=/path/to/yolov10x_best.pt
```

## 起動

```bash
.venv/bin/python app.py
```

既定では `http://127.0.0.1:7860` で起動します。

## 操作

1. PDF または画像ファイルをアップロードします。
2. 解析するページ範囲を指定します。既定は `1` です。画像ファイルは 1 ページとして扱います。
3. 使用するエンジンを選択します。OCI Document Understanding の直後に MinerU / MinerU2.5-Pro が表示されます。
4. `解析を実行` を押します。
5. React ビューアでページとエンジンを切り替え、JSONL 行をクリックして bbox を確認します。

## dots.mocr のプロンプト

dots.mocr では公式 `prompt_layout_all_en` をそのまま使用します。長い独自プロンプトには置き換えません。

## 出力

- `results.json`: 正規化済みの全結果
- `results.jsonl`: `pdf-jsonl-viewer` 互換フィールドを含む JSONL
- `viewer-data.json`: React ビューア用データ

正規化レコードには `id`, `engine`, `page`, `seq_no`, `bbox`, `coord_system`, `page_width`, `page_height`, `category`, `text`, `confidence`, `raw_type`, `raw` が含まれます。

## テスト

外部エンジンに接続しない単体テストは、プロジェクトの基本依存関係だけで実行できます。

```bash
.venv/bin/python -m unittest discover -s tests
```
