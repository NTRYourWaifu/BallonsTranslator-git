# BallonsTranslator（修改版）

基於 [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) 的個人修改版本，針對 LLM OCR、多層備援與工作流程進行了深度優化。

> [!IMPORTANT]
> **如打算公開分享本工具的機翻結果，且沒有有經驗的譯者進行過完整的翻譯或校對，請在顯眼位置注明機翻。**

---

## 與原版的主要差異

### LLM OCR 多層備援機制

本版最核心的改動：以 LLM（Gemini / OpenAI 相容 API）取代傳統 OCR，並同時完成翻譯，不需要獨立的翻譯器模組。

| Plan | 模式 | 說明 |
|------|------|------|
| **A** | 原圖全頁 | 將所有文字框裁切並拼成 Grid 圖，一次送 LLM 辨識全頁 |
| **B** | 切片（主 API） | Plan A 失敗後，每框單獨裁切，ThreadPoolExecutor 並行送主 API |
| **C** | 切片（備援 API） | Plan B 失敗的框改用備援 API（Grok）重試 |

失敗判定：HTTP 錯誤、安全過濾、JSON 解析失敗、框數不符均視為失敗。

### OCR 狀態列

底部狀態列即時顯示各備援層的成功 / 失敗計數：

| 圖示 | 顏色 | 意義 |
|------|------|------|
| ✓ | 綠 | Plan A 成功（全頁一次到位） |
| ✂ | 橘 | Plan B 成功（切片，主 API） |
| ◆ | 粉 | Plan C 成功（切片，備援 API） |
| ✕ | 紅 | 最終放棄（所有 Plan 均失敗） |

### YOLOv8 文字偵測器

- 可使用自訓練的 YOLOv8 模型進行文字框偵測
- 搭配 CTD 補充偵測，修正方向判斷異常的框
- 偵測結果填入 `TextBlock.obs`，後續由 `resolve_blk_style()` 統一計算字體大小與方向

### 其他功能

- **Google Lens OCR**：透過 Google Lens API 進行辨識（作為替代 OCR 引擎）
- **GPU 顯存限制**：啟動時自動將 PyTorch 顯存上限設為 80%（可在 `launch.py` 中調整）
- **彩色分級 Logger**：SUCCESS 綠色、EXPENSIVE 粉色，方便追蹤 OCR 流程

### 介面簡化

- 移除左側「暫pause」與「從上次繼續」按鈕，僅保留「停止」與「從當前頁繼續」
- 停止 / 暫停時自動清理 GPU 顯存

---

## 安裝與使用

### 方式一：Windows 整合包

從 [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) 或 [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) 下載 `BallonsTranslator_dev_src_with_gitpython.7z`，解壓後將本 repo 的修改檔案覆蓋進去，執行 `launch_win.bat` 啟動。

### 方式二：從原始碼執行

```bash
# 需要 Python <= 3.12 與 Git
git clone https://github.com/NTRYourWaifu/BallonsTranslator-git.git
cd BallonsTranslator-git
python launch.py
```

首次執行會自動安裝依賴項與下載模型。

---

## LLM OCR 設定

在設定面板中選擇 `llm_ocr`，填入以下參數：

### 必填

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `gemini_api_key` | 主 API 金鑰（Gemini 或 OpenAI 相容） | — |
| `model` | 主模型名稱 | `gemini-3.1-flash-lite-preview` |

### 備援（選填）

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `grok_api_key` | 備援 API 金鑰（xAI Grok） | — |
| `fallback_model` | 備援模型名稱 | `grok-3-mini-fast` |

### 效能調整

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `max_workers` | Plan B 切片並行數 | `5` |
| `delay` | 請求間隔（秒，`0` = 不限速） | `0.0` |
| `font_size_scale` | 字體大小整體縮放係數 | `1.0` |

### 逾時上限

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `plan_a_retry_seconds` | Plan A 過載重試上限 | `120` 秒 |
| `plan_b_retry_seconds` | Plan B 試探框逾時上限 | `300` 秒 |
| `plan_c_retry_seconds` | Plan C 試探框逾時上限 | `60` 秒 |

> **注意**：`delay > 0` 時，`max_workers` 的並行效益會大幅下降，API 有速率限制時才建議設 delay。

---

## 快捷鍵

| 按鍵 | 功能 |
|------|------|
| `A` / `D` | 翻頁 |
| `T` | 文字編輯模式 |
| `P` | 畫板模式 |
| `Ctrl+Z` / `Ctrl+Y` | 復原 / 重做 |
| `Ctrl+A` | 選取全部文字框 |
| `Ctrl+F` | 當前頁查找 |
| `Ctrl+G` | 全域查找 |
| `Ctrl++` / `Ctrl+-` | 縮放畫布 |
| `0`–`9` | 調整嵌字 / 原圖透明度 |
| `Ctrl+B` / `Ctrl+U` / `Ctrl+I` | 粗體 / 底線 / 斜體 |

---

## 已知限制

- Plan A 的 Grid 拼圖對超長橫向文字框效果較差（LLM 視野有限）
- `delay > 0` 時並行效益消失，實際等同串行
- 手寫字體字體大小估算偏差較大（render 結果與 LLM 回傳 font_size 差距超過閾值會警告但不修正）

---

## 致謝

- 原始專案由 [dmMaze](https://github.com/dmMaze/BallonsTranslator) 開發
- 後端依賴 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)
- Sugoi 翻譯器作者：[mingshiba](https://www.patreon.com/mingshiba)
- Google Lens OCR 實現參考社群貢獻

## 授權

本專案遵循 [GPL-3.0](LICENSE) 授權。
