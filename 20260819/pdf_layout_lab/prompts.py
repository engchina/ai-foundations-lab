"""dots.mocr 公式プロンプト。

公式リポジトリの dots_ocr/utils/prompts.py にある prompt_layout_all_en を、
長い独自プロンプトへ置き換えずに利用するための定義です。
"""

DOTS_MOCR_PROMPT_SOURCE = "https://github.com/studio-dots-ai/dots.ocr/blob/master/dots_ocr/utils/prompts.py"

PROMPT_LAYOUT_ALL_EN = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.
5. Final Output: The entire output must be a single JSON object.
"""

# Picture 切り出し画像の種類を判定する独自プロンプト（dots.mocr 公式ではない）。
# 一語で答えず説明を付けることが多いので、応答からキーワードを拾う前提
PROMPT_PICTURE_KIND = """Classify this image. Answer with exactly one word from this list: flowchart, table, chart, screenshot, photo, logo, other.
- flowchart: boxes or shapes connected by arrows or lines that show a process or flow.
- table: rows and columns of cells.
- chart: bar, line, or pie chart.
- screenshot: a captured application, web page, or form UI.
- photo: a photograph.
- logo: a logo, stamp, seal, or QR/barcode.
"""

# Picture 切り出し画像から Mermaid を生成する独自プロンプト（dots.mocr 公式ではない）
PROMPT_PICTURE_MERMAID = """Convert this image into a Mermaid flowchart that reproduces its structure (boxes, callouts, arrows, reading order).
Rules:
- Output only one ```mermaid code block, starting with `graph TD`.
- Node labels must be the original text from the image, no translation.
- If the image is not a diagram, flowchart, screen flow, or process (for example a photo, logo, or decoration), output exactly NONE.
"""
