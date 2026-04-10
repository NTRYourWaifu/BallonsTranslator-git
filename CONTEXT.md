# 專案上下文：氣球翻譯器 (改進版)

## 1. 專案概述
本專案是基於原始「氣球翻譯器 (BalloonsTranslator)」進行深度改造的專業漫畫翻譯工具。核心目標是透過現代 AI 技術（尤其是大語言模型 LLM）提升翻譯的脈絡準確度，並優化 UI 互動與批量處理效率。

## 2. 核心技術亮點

### 2.1 算法融合文字檢測 (Detection Fusion)
*   **模型組合**：結合 **YOLOv8**（擅長大範圍氣泡偵測）與 **CTD (Comic Text Detector)**（擅長精確文字定位）。
*   **融合策略**：
    *   以 YOLOv8 為主要框架進行初篩與欄位合併。
    *   啟用 **CTD 補充模式**：針對 YOLO 容易漏掉的橫排標題、旁白或邊緣小字進行補抓與替換。
    *   透過面積佔比與排列方向（Vertical/Horizontal）判定，自動選擇最精確的檢測框。

### 2.2 LLM OCR 與翻譯策略 (Plan A/B/C)
放棄傳統 OCR 引擎，全面對接大模型 API（Gemini 1.5 Pro/Flash, Grok 等）：
*   **Plan A (Grid 拼圖模式)**：將一頁中的所有文字框裁切、標號並拼成一張「Grid 拼圖」。
    *   *優點*：LLM 能一次看到整頁的影像內容與相對位置，提供具備**上下文感感知**的翻譯，且大幅節省 API 呼叫次數。
*   **Plan B (Slice 切片模式)**：若拼圖解析失敗，則退回到單個文字框裁切傳送。
*   **Plan C (Fallback 備援模式)**：當主模型（如 Gemini）觸發安全過濾或過載時，自動切換至備援模型（如 Grok）。

### 2.3 影像修復 (Inpainting)
*   集成 LaMa、AOT-GAN 等先進修復模型，在 OCR 翻譯的同時同步去除原文，實現「無痕」文字替換。

## 3. 系統架構與多執行緒

### 3.1 非同步管線 (Async Pipeline)
系統採用雙軌並行模式，以最大化硬體與網路利用率：
*   **主迴圈 (Main Thread/Pipeline)**：負責 GPU 密集型任務（Detection + Inpaint）。
*   **背景 Worker (Multi-Worker)**：
    *   根據設定啟動數個 **OCR Workers**。
    *   利用 `Queue` 機制接手主迴圈處理完的頁面。
    *   負責網路密集型任務（API Call + JSON 解析），支援 `start_delay` 錯開請求以避免 API 限速。

### 3.2 顯存管理 (VRAM Management)
*   支援 **Low VRAM 模式**：階段性載入/卸載模型（先跑檢測修復 -> 卸載 -> 跑 OCR），讓低顯存設備也能運作。

## 4. 目錄結構指南
*   `modules/textdetector/`：YOLOv8 與 CTD 融合邏輯 (`detector_yolov8.py`)。
*   `modules/ocr/`：核心 LLM 驅動邏輯 (`ocr_llm.py`)，包含拼圖與重試機制。
*   `modules/inpaint/`：影像修復演算法。
*   `ui/module_manager.py`：負責串接整個 Pipeline 與管理多 Worker 執行緒。
*   `ui/batch_queue_panel.py`：批量任務面板與 OCR 成功率統計。
*   `ui/scenetext_manager.py`：處理畫面上文字項目的自動排版 (`layout_textblk`) 與樣式渲染。

## 5. 開發狀態與目標
*   **已完成**：UI 改造、LLM 三階段策略、YOLO+CTD 融合、多執行緒排隊系統。
*   **重點關注**：
    *   API 回傳 JSON 的解析魯棒性。
    *   自動排版（Auto-layout）在不同語系間的適應性。
    *   大規模批量處理時的錯誤恢復機制。
