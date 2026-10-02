# PageLifecycle FSM 設計文件

目的：用顯式狀態機取代散在 counter / signal / if 分支裡的隱性狀態，消除「emit 順序的時序合約」與「靠守衛補救殘留 signal」這類治標寫法。

## a) 執行緒契約

| 來源 | 動作 | 機制 | 規則 |
|---|---|---|---|
| Worker thread | 改 `_state[imgname]` | `QMutex` 同步 | `mark_done` 內部一律持鎖寫入；寫完釋鎖才 emit transition |
| Worker thread → 任何 slot | `transition` signal | `Qt.QueuedConnection` 強制 | 跨執行緒一律 queued，接收方在自己 thread 的 event loop 處理 |
| 主執行緒 runner | 跑 FONT_CALCULATED / DONE | 訂閱 `transition` 對應 stage | 跑完才 `mark_done(下一 stage)`，runner 自己負責推進 |
| `mark_done` 重入 | 同頁同 stage 再次呼叫 | 直接 noop + WARN log | 避免殘留 signal 把已 DONE 的頁拉回 |
| 同頁逆向 stage | 例：DONE 後收到 OCRED | reject + ERROR log | 狀態只能 monotonic 推進 |

**Stage 擁有者（owner）契約**

| Stage | Owner thread | Mark 的時機 |
|---|---|---|
| `PENDING` | lifecycle.initialize() | 建構即設定 |
| `DETECTED` | detect worker | textdetector.detect() 回傳後 |
| `INPAINTED` | inpaint worker | inpainter.inpaint() 回傳後（或 mask 為空時 noop mark） |
| `OCRED` | OCR worker | `_do_ocr_page` 回傳後 |
| `TRANSLATED` | OCR worker（與 OCR 同 worker） | `translator.translate_textblk_lst` 回傳後 |
| `FONT_CALCULATED` | **主執行緒 runner**（訂閱 TRANSLATED） | `calc_font_size_by_render` 跑完 + 顏色/stroke 補做完 |
| `DONE` | **主執行緒 runner**（訂閱 FONT_CALCULATED） | UI 刷新 + saveImg 派工完成 |

關鍵點：**worker 只 mark 到 TRANSLATED 就回去抓下一頁，不等 font_calc**。並發不退化；FONT_CALCULATED 由主執行緒 runner 自己 mark，不會有「worker emit 順序」問題。

## b) PageStage 最終定名（採建議 2 精簡）

`PENDING → DETECTED → INPAINTED → OCRED → TRANSLATED → FONT_CALCULATED → DONE`

7 個。砍掉建議的 `UI_REFRESHED + SAVED` 兩個獨立 stage（這兩件事順序固定、由同一個主執行緒 runner 連續做、外界沒人需要中間態），合併為 `DONE` 的 side effect。

## c) Transition signal payload schema（採建議 3）

```python
Signal transition(imgname: str, prev: PageStage, new: PageStage, payload: dict)
```

| Stage | payload 內容 |
|---|---|
| `PENDING` | `{}` |
| `DETECTED` | `{'blk_count': int, 'has_mask': bool}` |
| `INPAINTED` | `{'has_inpainted': bool}` |
| `OCRED` | `{'blk_count_with_text': int}` |
| `TRANSLATED` | `{}` |
| `FONT_CALCULATED` | `{'font_sizes': dict[int, float]}`（每框最終 px） |
| `DONE` | `{'save_path': str | None}` |

下游 handler 從 payload 取「事件當下的快照值」，**不再去 `imgtrans_proj` 讀可變共享狀態**——這跟「task local 參數」精神一致。`blk_list` 物件本身的修改仍經 `imgtrans_proj.pages`（在 worker 寫入時尚未 emit transition，所以下游讀到的是寫入後的值），但任何「event 當下的數量、是否為空、最終字體值」一律走 payload。

## d) 空頁與 stage noop 處理（採方案 B）

**所有頁 expected_stages 都一樣**（靜態，依 `enable_*` 旗組合決定），**不依賴 detect 結果動態調整**。

當某頁對某 stage 無事可做（例：empty blk → FONT_CALCULATED 沒框可算）：
- runner 直接呼 `mark_done(stage, payload={...empty...})`，事件序列完整
- log 顯示 `noop`，偵錯時清楚知道「跑到這個 stage 但沒事做」

例：純圖頁（detect 後 blk_list 為空）
1. `DETECTED` mark with `{'blk_count': 0}`
2. `INPAINTED` 正常跑
3. `OCRED` runner 看到 0 框 → noop mark
4. `TRANSLATED` runner 看到 0 框 → noop mark
5. `FONT_CALCULATED` runner 看到 0 框 → noop mark with `{'font_sizes': {}}`
6. `DONE` runner 看到 0 框 → 走「純圖頁存 inpainted」分支，不做 UI 刷新

`is_all_done()` 判定極簡：`all(s.current_stage == DONE for s in self._state.values())`。

## 不變式（invariants）保證表

| 不變式 | 由什麼保證 |
|---|---|
| 字體 calc 前 translation 已寫入 | FONT_CALCULATED runner 訂閱 TRANSLATED，只在 transition 觸發後跑 |
| 顏色寫回必須在 stroke 之前 | FONT_CALCULATED runner 內部順序硬編，不再散在 handler |
| pipeline 完成時所有頁 UI 都刷完 | `all_done` 由 lifecycle 判定，不再由 progress signal 觸發 |
| 範圍外 signal 不影響 UI | lifecycle 一開始只放本批的頁，殘留 signal 對應 imgname 不在 `_state` 中，`mark_done` 直接 reject |
| Stage 單調遞增 | `mark_done` 內 `assert new > current`，逆向直接 reject |
