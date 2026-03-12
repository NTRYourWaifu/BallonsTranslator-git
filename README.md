# BallonsTranslator（修改版）

# 以下均為AI生成摘要
基於 [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) 的個人修改版本，針對 LLM OCR、顯存管理與工作流程進行了多項優化。

> [!IMPORTANT]
> **如打算公開分享本工具的機翻結果，且沒有有經驗的譯者進行過完整的翻譯或校對，請在顯眼位置注明機翻。**

---

## 與原版的主要差異

### 新增功能

- **LLM OCR 多層備援機制**
  - Plan A：Gemini 原圖全頁辨識
  - Plan B：黑底圖全頁重試（提高對比度）
  - Plan C：切片逐框辨識（主 API → Grok 備援）
  - Plan D：Grok 全頁備援
  - 支援 Gemini 與 OpenAI 相容 API 作為主要 OCR 引擎

- **OCR 狀態列**（底部即時顯示）
  - ✓ 綠：Plan A 成功 ｜ ▲ 黃：Plan B 成功 ｜ ✂ 橘：Plan C 切片成功 ｜ ♦ 粉：Grok 備援成功 ｜ ✕ 紅：失敗

- **YOLOv8 文字偵測器**：可使用自訓練的 YOLOv8 模型進行文字偵測

- **Google Lens OCR**：透過 Google Lens API 進行文字辨識

- **GPU 顯存限制**：啟動時自動將 PyTorch 顯存上限設為 80%（可在 `launch.py` 中調整）

### 介面優化

- 簡化左側工具列：移除「暫停」與「從上次繼續」按鈕，僅保留「停止」與「從當前頁繼續」
- 停止 / 暫停時自動清理 GPU 顯存

### 其他改進

- 增強 Logger：新增彩色分級日誌（SUCCESS 綠色、EXPENSIVE 粉色），方便追蹤 OCR 流程
- 各模組的繁體中文介面註解

---

## 安裝與使用

### Windows

從 [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) 或 [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) 下載 `BallonsTranslator_dev_src_with_gitpython.7z`，解壓後將本 repo 的修改檔案覆蓋進去，執行 `launch_win.bat` 啟動。

### 從原始碼執行

```bash
# 安裝 Python <= 3.12 與 Git
git clone https://github.com/NTRYourWaifu/BallonsTranslator-git.git
cd BallonsTranslator-
python launch.py
```

首次執行會自動安裝依賴項與下載模型。

---

## LLM OCR 設定

在設定面板中選擇 `llm_ocr`，並填入以下參數：

| 參數 | 說明 |
|------|------|
| API 提供者 | Gemini 或 OpenAI 相容 API |
| API Key | 主要 API 金鑰 |
| Model | 建議 `gemini-3.1-flash-lite-preview` |
| 備援 API Key | Grok 備援金鑰（選填） |
| 啟用切片 | Plan C 切片備援開關 |

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
| `0-9` | 調整嵌字 / 原圖透明度 |
| `Ctrl+B` / `Ctrl+U` / `Ctrl+I` | 粗體 / 底線 / 斜體 |

---

## 致謝

- 原始專案由 [dmMaze](https://github.com/dmMaze/BallonsTranslator) 開發
- 後端依賴 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)
- Sugoi 翻譯器作者：[mingshiba](https://www.patreon.com/mingshiba)
- Google Lens OCR 實現參考社群貢獻

## 授權

本專案遵循 [GPL-3.0](LICENSE) 授權。
