# BallonsTranslator — OCR 模組開發說明書

## 這份說明書的用途

這份文件是給新對話的 Claude 看的，用來快速理解這個專案的背景、架構、以及目前的開發狀態，避免每次都要重新解釋。

---

## 專案簡介

**BallonsTranslator** 是一個漫畫自動翻譯工具，流程是：
1. 用 YOLOv8 偵測漫畫頁面上的文字框
2. 用 OCR 讀取每個文字框的日文原文
3. 用翻譯器（ChatGPT 等）翻譯成繁中
4. 用修復器（lama）塗白原文，渲染翻譯後的文字

知識庫內上傳了核心程式碼（檔名格式為 `路徑_路徑_檔名.py`），還有 `project_tree.txt`（完整檔案樹）可以參考。

---

## 核心資料結構：TextBlock

定義在 `utils/textblock.py`，是整個 pipeline 的資料載體：

```python
TextBlock
├── xyxy              [x1, y1, x2, y2]  框的像素座標
├── lines             文字行多邊形點
├── vertical          bool  直排/橫排
├── text              List[str]  OCR 辨識原文
├── translation       str  翻譯後文字
├── font_size         float  渲染字體大小
└── _detected_font_size  float  自動計算字體大小
```

---

## Pipeline 流程

```
module_manager._imgtrans_pipeline()
  │
  ├─ 1. TextDetector（YOLOv8）
  │       → 回傳 List[TextBlock]，每個框有 xyxy 座標
  │
  ├─ 2. OCR（OCRLlm）← 主要開發對象
  │       → 填入 blk.text（原文）
  │
  ├─ 3. Translator（ChatGPT 等）
  │       → 填入 blk.translation
  │
  └─ 4. Inpainter（lama）
          → 塗白 + 渲染翻譯文字
```

---

## OCRLlm 架構（modules/ocr/ocr_llm.py）

這是本次主要開發的檔案，目前的邏輯：

### Plan A / B / C / D

```
Plan A：原圖 → _build_grid_img() → LLM → _parse_fullpage_result()
Plan B：塗黑圖 → 同上（繞過 safety filter）
Plan C：切片模式，每個框單獨裁切送 LLM（最慢但最可靠）
Plan D：Grok fallback（C 被擋時單框備援）
```

### 關鍵設計：Grid 拼圖

Plan A/B 不再送原圖給 LLM，改為：
1. 把所有 TextBlock 裁切出來
2. 用 **bin packing** 拼成一張接近正方形的拼圖（`_build_grid_img`）
3. 同時記錄 `visual_order[視覺index] = orig_idx`
4. LLM 按右到左、上到下的視覺順序讀圖，回傳 `index` 0,1,2...
5. `_parse_fullpage_result` 用 `visual_order` 把視覺 index 轉回原始框 index

### Grid 拼圖格式

- 每個格子保持原始裁切比例，不縮放
- 左側有黑色 label 欄（寬度依解析度比例）
- 格子間有固定間距（gap）
- 行高取該行最高框的高度（同行 x 軸對齊）

### Prompt 設計（_GRID_PROMPT）

告訴 LLM：
- 圖片是格子拼圖
- 閱讀順序：右到左（欄），上到下（欄內）
- index 0 在最右欄第 0 行
- 回傳格式：`[{"index": N, "direction": "v"/"h", "original": "...", "translation": "..."}]`

### 錯誤處理

- `_explain_error()`：把 HTTP 錯誤碼轉成中文說明
- Plan A 失敗會 log 原因再進 Plan B，以此類推
- C 階段單框失敗才填 `●●●`

---

## 支援的 API

- **主 API**：Gemini（GeminiClient）或 OpenAI 相容 API（OpenAICompatClient）
- **備援 API**：Grok（也用 OpenAICompatClient，base_url = api.x.ai）

---

## 已知問題 / 待改進

- Grid 拼圖目前有 debug 存檔功能（`grid_debug_xxx.jpg`），正式版應移除
- `_sort_blk_reading_order` 排序邏輯現在其實不影響正確性（visual_order 已處理），可以考慮簡化
- Plan B 的塗黑圖現在也走 grid 模式，跟 Plan A 邏輯相同，差別只在圖片

---

## 開發注意事項

1. **TextBlock 不能用 `==` 比較**，會觸發 `deprecated_attributes` 錯誤，要用 `id(blk)` 做識別
2. **`blk_list` 是 in-place 修改**，`_ocr_blk_list` 直接改傳入的 list
3. **`current_imgname`** 要在呼叫前設好，log 用來顯示目前處理的圖片名稱
4. **`self.logger`** 用 loguru，有 `success` / `warning` / `error` / `debug` 等級

---

## 檔案對照表

知識庫內的檔案命名規則：`路徑用底線連接`

| 知識庫檔名 | 原始路徑 |
|-----------|---------|
| `modules_ocr_ocr_llm.py` | `modules/ocr/ocr_llm.py` |
| `modules_ocr_base.py` | `modules/ocr/base.py` |
| `modules_base.py` | `modules/base.py` |
| `utils_textblock.py` | `utils/textblock.py` |
| `utils_registry.py` | `utils/registry.py` |
| `ui_module_manager.py` | `ui/module_manager.py` |
| `modules_textdetector_base.py` | `modules/textdetector/base.py` |
| `modules_textdetector_detector_yolov8.py` | `modules/textdetector/detector_yolov8.py` |
| `modules_translators_base.py` | `modules/translators/base.py` |

