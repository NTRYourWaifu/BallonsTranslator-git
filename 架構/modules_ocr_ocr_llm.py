import math
import numpy as np
import cv2
import base64
import json
import time
import requests
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from qtpy.QtCore import Signal, QObject

from .base import register_OCR, OCRBase, TextBlock


# ── 統計事件類型 ──────────────────────────────────────────────
class OcrEventType:
    PLAN_A_OK  = 'plan_a_ok'   # 綠  原圖全頁成功
    PLAN_A2_OK = 'plan_a2_ok'  # 黃  黑圖全頁成功
    SLICE_OK   = 'slice_ok'    # 橘  切片（主API）成功
    GROK_OK    = 'grok_ok'     # 粉  Grok 成功（切片或全頁）
    ERROR      = 'error'       # 紅  最終放棄此頁


class OcrStatsSignals(QObject):
    event = Signal(str)


# ── 圖片工具 ──────────────────────────────────────────────────
def _img_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


# ── API 客戶端 ────────────────────────────────────────────────
class GeminiClient:
    BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

    def __init__(self, api_key: str, model: str = 'gemini-3.1-flash-lite-preview'):
        self.api_key = api_key
        self.model = model

    def ocr(self, img_b64: str, prompt: str) -> str:
        url = self.BASE_URL.format(model=self.model) + f'?key={self.api_key}'
        payload = {
            'contents': [{'parts': [
                {'text': prompt},
                {'inline_data': {'mime_type': 'image/jpeg', 'data': img_b64}}
            ]}],
            'generationConfig': {'temperature': 0.0, 'response_mime_type': 'application/json'},
            'safetySettings': [
                {'category': 'HARM_CATEGORY_HARASSMENT',        'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_HATE_SPEECH',       'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
            ]
        }
        resp = requests.post(url, json=payload, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            if 'promptFeedback' in data and 'blockReason' in data['promptFeedback']:
                raise RuntimeError(f"Blocked: {data['promptFeedback']['blockReason']}")
            return data['candidates'][0]['content']['parts'][0]['text'].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Gemini 回應異常: {data}") from e


class OpenAICompatClient:
    def __init__(self, api_key: str, model: str = 'gpt-4o',
                 base_url: str = 'https://api.openai.com/v1'):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')

    def ocr(self, img_b64: str, prompt: str, timeout: int = 45) -> str:
        url = f'{self.base_url}/chat/completions'
        headers = {'Authorization': f'Bearer {self.api_key}',
                   'Content-Type': 'application/json'}
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}}
            ]}],
            'response_format': {'type': 'json_object'}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f'OpenAI API 回應異常: {data}') from e


# ── 主模組 ────────────────────────────────────────────────────
@register_OCR('llm_ocr')
class OCRLlm(OCRBase):
    params = {
        'provider': {
            'type': 'selector',
            'options': ['Gemini', 'OpenAI / 相容 API'],
            'value': 'Gemini',
            'description': '選擇 API 提供商'
        },
        'api_key':       {'value': '', 'description': 'API 金鑰'},
        'model':         {'value': 'gemini-3.1-flash-lite-preview',
                          'description': '建議：gemini-3.1-flash-lite-preview'},
        'base_url':      {'value': '', 'description': '僅 OpenAI 相容 API 使用'},
        'delay':         {'value': '0.0', 'description': '請求間隔（付費版可設為 0）'},
        'max_workers':   {'value': '5', 'description': '切片模式並行數（建議 5~10）'},
        'font_size_ratio': {'value': '0.8', 'description': '字體大小係數（0.5~1.2）'},
        'fallback_api_key': {'value': '', 'description': '備援 Grok API 金鑰'},
        'fallback_model':   {'value': 'grok-4.20-beta-0309-non-reasoning',
                             'description': '備援模型，需支援 vision'},
        'use_fallback_only':  {'type': 'checkbox', 'value': False,
                               'description': '全部走備援 API'},
        'enable_masked_plan': {'type': 'checkbox', 'value': True,
                               'description': 'Plan B：原圖被擋後送塗黑版'},
        'enable_slice_plan':  {'type': 'checkbox', 'value': True,
                               'description': 'Plan C：B 被擋後切片逐框，被擋送 Plan D Grok'},
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.client = None
        self.last_request_time = 0
        self.page_counter = 0
        self.current_imgname = ''
        self.stats_signals = OcrStatsSignals()
        self._build_client()

    def _fmt_imgname(self, name: str) -> str:
        """超過9字元縮寫：前3字元 + ... + 後3字元（不含副檔名部分補上副檔名）
        例：01_37639451_p0.webp → 01_...p0.webp"""
        if len(name) <= 9:
            return name
        ext_idx = name.rfind('.')
        ext = name[ext_idx:] if ext_idx != -1 else ''          # .webp
        stem = name[:ext_idx] if ext_idx != -1 else name       # 01_37639451_p0
        return f'{stem[:3]}...{stem[-3:]}{ext}'                 # 01_...p0.webp

    # ── properties ───────────────────────────────────────────
    @property
    def provider(self) -> str:   return self.params['provider']['value']
    @property
    def api_key(self) -> str:    return self.params['api_key']['value']
    @property
    def model(self) -> str:      return self.params['model']['value']
    @property
    def base_url(self) -> str:   return self.params['base_url']['value']
    @property
    def delay(self) -> float:
        try: return float(self.params['delay']['value'])
        except: return 0.0
    @property
    def max_workers(self) -> int:
        try: return int(self.params['max_workers']['value'])
        except: return 5
    @property
    def font_size_ratio(self) -> float:
        try: return float(self.params['font_size_ratio']['value'])
        except: return 0.8
    @property
    def fallback_api_key(self) -> str:  return self.params['fallback_api_key']['value']
    @property
    def fallback_model(self) -> str:    return self.params['fallback_model']['value']
    @property
    def use_fallback_only(self) -> bool:   return bool(self.params['use_fallback_only']['value'])
    @property
    def enable_masked_plan(self) -> bool:  return bool(self.params['enable_masked_plan']['value'])
    @property
    def enable_slice_plan(self) -> bool:   return bool(self.params['enable_slice_plan']['value'])

    # ── 統計事件 ──────────────────────────────────────────────
    def _emit(self, event_type: str):
        self.stats_signals.event.emit(event_type)

    # ── 客戶端 ────────────────────────────────────────────────
    def _build_client(self):
        if not self.api_key: return
        if self.provider == 'Gemini':
            self.client = GeminiClient(api_key=self.api_key, model=self.model)
        else:
            self.client = OpenAICompatClient(
                api_key=self.api_key, model=self.model, base_url=self.base_url)

    def _build_fallback_client(self):
        if not self.fallback_api_key: return None
        return OpenAICompatClient(
            api_key=self.fallback_api_key,
            model=self.fallback_model,
            base_url='https://api.x.ai/v1')

    # ── 速率控制 ──────────────────────────────────────────────
    def _respect_delay(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()

    # ── 圖片工具 ──────────────────────────────────────────────
    def _make_masked_img(self, img: np.ndarray,
                          blk_list: List[TextBlock]) -> np.ndarray:
        """黑圖只保留合併後的文字框區域（Plan B 用）。"""
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            mask[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(img, img, mask=mask)

    # ── 字體大小 ──────────────────────────────────────────────
    def _auto_font_size(self, blk: TextBlock, text: str) -> float:
        x1, y1, x2, y2 = blk.xyxy
        box_area = max((x2 - x1) * (y2 - y1), 1)
        char_count = max(len(text.replace(' ', '').replace('\n', '')), 1)
        font_size_px = math.sqrt(box_area / char_count) * self.font_size_ratio
        return max(8.0, min(72.0, font_size_px * 0.75))

    def _apply_font_size(self, blk: TextBlock, text: str):
        fs = self._auto_font_size(blk, text)
        blk.font_size = fs
        blk._detected_font_size = fs

    # ── 底層 API 呼叫 ─────────────────────────────────────────
    # ── 錯誤碼中文對照 ───────────────────────────────────────
    @staticmethod
    def _explain_error(err: str) -> str:
        if '429' in err or 'exhausted' in err.lower() or 'quota' in err.lower():
            return '配額耗盡/限速'
        if '401' in err or 'unauthorized' in err.lower() or 'api key' in err.lower():
            return 'API金鑰無效'
        if '403' in err or 'forbidden' in err.lower():
            return '無存取權限'
        if '500' in err or 'internal' in err.lower():
            return 'API伺服器內部錯誤'
        if '503' in err or 'unavailable' in err.lower():
            return 'API服務暫時不可用'
        if 'timeout' in err.lower() or 'timed out' in err.lower():
            return '請求逾時'
        if 'connection' in err.lower() or 'network' in err.lower():
            return '網路連線失敗'
        if 'json' in err.lower() or 'decode' in err.lower():
            return 'API回傳格式異常'
        if 'Blocked' in err or 'PROHIBITED_CONTENT' in err:
            return '安全過濾器擋住'
        return f'未知錯誤({err[:30]})'

    def _call_ocr(self, img: np.ndarray, custom_prompt: str = None) -> str:
        """回傳: 原始字串 | 'BLOCKED_BY_SAFETY' | 'ERR:原因'"""
        if self.use_fallback_only:
            return 'BLOCKED_BY_SAFETY'
        if self.client is None:
            return 'ERR:未設定API金鑰'
        img_b64 = _img_to_base64(img)
        target_prompt = custom_prompt
        max_retries = 3
        for attempt in range(max_retries):
            self._respect_delay()
            try:
                return self.client.ocr(img_b64, target_prompt)
            except Exception as e:
                err = str(e)
                if '429' in err or 'exhausted' in err.lower():
                    wait = 3 * (attempt + 1)
                    self.logger.warning(f"限速，暫停 {wait}s 重試 ({attempt+1}/{max_retries})...")
                    time.sleep(wait)
                elif 'Blocked' in err or 'PROHIBITED_CONTENT' in err:
                    return 'BLOCKED_BY_SAFETY'
                else:
                    reason = self._explain_error(err)
                    self.logger.error(f"API錯誤：{reason}")
                    return f'ERR:{reason}'
        reason = '配額耗盡/限速，重試耗盡'
        self.logger.error(reason)
        return f'ERR:{reason}'

    def _call_ocr_grok(self, img: np.ndarray, prompt: str, log_prefix: str,
                        silent: bool = False):
        """回傳: str（成功）| ''（失敗）。silent=True 時不印log，由呼叫方處理。"""
        client = self._build_fallback_client()
        if client is None:
            return ''
        try:
            return client.ocr(_img_to_base64(img), prompt, timeout=120)
        except Exception as e:
            err = str(e)
            code = re.search(r'(\d{3})', err)
            code_str = code.group(1) if code else err[:40]
            if not silent:
                self.logger.error(f"{log_prefix} Grok 失敗: {code_str}")
            return code_str if silent else ''

    # ── 解析全頁結果（index-based，不需座標配對）────────────
    def _parse_fullpage_result(self, response_text: str,
                                blk_list: List[TextBlock],
                                visual_order: list = None) -> bool:
        try:
            clean = re.sub(r'```json\s*|\s*```', '', response_text).strip()
            results = json.loads(clean)
            if not isinstance(results, list) or not results:
                return False

            matched = 0
            for item in results:
                if 'original' not in item:
                    continue
                visual_idx = item.get('index')
                if visual_idx is None or not isinstance(visual_idx, int):
                    continue
                if visual_order is not None:
                    if visual_idx < 0 or visual_idx >= len(visual_order):
                        self.logger.warning(f"LLM 回傳 index={visual_idx} 超出視覺順序範圍，跳過")
                        continue
                    idx = visual_order[visual_idx]
                else:
                    idx = visual_idx
                if idx < 0 or idx >= len(blk_list):
                    self.logger.warning(f"orig index={idx} 超出範圍（共{len(blk_list)}框），跳過")
                    continue
                blk = blk_list[idx]
                blk.text = [item['original']]
                blk.translation = item.get('translation', '')
                llm_dir = item.get('direction', '').lower()
                if llm_dir == 'v':
                    blk.vertical = True
                elif llm_dir == 'h':
                    blk.vertical = False
                self._apply_font_size(blk, item['original'])
                matched += 1

            self.logger.debug(f"全頁解析：{matched}/{len(blk_list)} 框成功")
            return matched > 0
        except Exception as e:
            self.logger.warning(f"全頁解析失敗: {e}")
            return False

    # ── 全頁模式（單次） ──────────────────────────────────────
    _GRID_CELL = 192   # 每個格子的邊長（px）
    _GRID_PAD  = 28    # 格子間距（px），需夠大放 index 編號

    _GRID_PROMPT = (
        "This image is a grid of manga text box crops.\n"
        "The grid is arranged RIGHT-TO-LEFT in columns, TOP-TO-BOTTOM within each column.\n"
        "index 0 is at the TOP of the RIGHTMOST column, then goes DOWN, then moves LEFT to the next column.\n"
        "Each cell contains one text box. Read the Japanese text in each cell and translate to Traditional Chinese.\n"
        "\n"
        "Output ONLY a valid JSON array, one entry per cell, ordered by index:\n"
        '[{"index": 0, "direction": "v or h", "original": "...", "translation": "..."}, ...]\n'
        'direction: "v"=vertical/tategumi, "h"=horizontal/yokogumi.\n'
        'If a cell has no readable text: {"index": N, "direction": "v", "original": "", "translation": ""}\n'
        "Total cells: {n}"
    )

    def _build_grid_img(self, img: np.ndarray,
                        blk_list: List[TextBlock]) -> np.ndarray:
        import math
        h_img, w_img = img.shape[:2]

        label_w = max(20, int(w_img * 0.018))  # 左側編號欄寬
        gap = max(6, int(w_img * 0.005))        # 格子間距

        # 裁切所有框，保持原始比例
        crops = []
        for blk in blk_list:
            bx1, by1, bx2, by2 = blk.xyxy
            px1 = max(0, bx1 - 4); py1 = max(0, by1 - 4)
            px2 = min(w_img, bx2 + 4); py2 = min(h_img, by2 + 4)
            crop = img[py1:py2, px1:px2]
            crops.append(crop if crop.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8))

        # 目標寬度 = sqrt(所有格子面積總和)，盡量接近正方形
        total_area = sum((c.shape[1] + label_w) * c.shape[0] for c in crops)
        target_w = max(int(math.sqrt(total_area)), 100)

        # Bin packing：框按高度由大到小排，逐行填充
        order = sorted(range(len(crops)), key=lambda i: -crops[i].shape[0])
        rows = []       # list of list of (orig_idx, crop)
        row_ws = []     # 每行目前總寬
        row_hs = []     # 每行最大高

        for orig_idx in order:
            crop = crops[orig_idx]
            ch, cw = crop.shape[:2]
            fw = cw + label_w  # 含 label 的格子寬

            placed = False
            for r in range(len(rows)):
                if row_ws[r] + gap + fw <= target_w or len(rows[r]) == 0:
                    rows[r].append((orig_idx, crop))
                    row_ws[r] += (gap if row_ws[r] > 0 else 0) + fw
                    row_hs[r] = max(row_hs[r], ch)
                    placed = True
                    break
            if not placed:
                rows.append([(orig_idx, crop)])
                row_ws.append(fw)
                row_hs.append(ch)

        # canvas 尺寸（含 label_w）
        canvas_w = max(row_ws) + gap
        canvas_h = sum(row_hs) + (len(rows) + 1) * gap
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 貼圖：LLM 和 debug 看同一張，左側黑底寫編號
        y_cursor = gap
        for r, row in enumerate(rows):
            x_cursor = gap
            for orig_idx, crop in row:
                ch, cw = crop.shape[:2]
                row_h = row_hs[r]

                # 黑色 label 欄
                canvas[y_cursor:y_cursor+row_h, x_cursor:x_cursor+label_w] = 0
                # 裁切圖靠左上貼
                canvas[y_cursor:y_cursor+ch,
                       x_cursor+label_w:x_cursor+label_w+cw] = crop
        
                x_cursor += label_w + cw + gap
            y_cursor += row_hs[r] + gap

        # 建立視覺順序映射：LLM 按右到左、上到下讀
        # visual_order[視覺index] = orig_idx
        visual_order = []
        for r, row in enumerate(rows):
            for orig_idx, _ in reversed(row):
                visual_order.append(orig_idx)

        self.logger.debug(f"grid img: {len(crops)}框 → {len(rows)}行, canvas={canvas_w}x{canvas_h}")
        self.logger.debug(f"visual_order: {visual_order}")

        # DEBUG：存到專案根目錄
        safe_name = self.current_imgname.replace('/', '_').replace('\\', '_') or 'unknown'
        debug_path = f"grid_debug_{safe_name}.jpg"
        cv2.imwrite(debug_path, canvas)
        self.logger.warning(f"grid debug 圖已存至 {debug_path}")

        return canvas, visual_order

    def _run_fullpage(self, img: np.ndarray, blk_list: List[TextBlock]) -> str:
        grid_img, visual_order = self._build_grid_img(img, blk_list)
        prompt = self._GRID_PROMPT.replace('{n}', str(len(blk_list)))
        resp = self._call_ocr(grid_img, custom_prompt=prompt)
        if resp == 'BLOCKED_BY_SAFETY':
            return '安全過濾器擋住'
        if resp.startswith('ERR:'):
            return resp[4:]
        if not resp:
            return 'API無回應'
        if self._parse_fullpage_result(resp, blk_list, visual_order):
            return 'ok'
        return 'JSON解析失敗'

    # ── 切片模式 ──────────────────────────────────────────────
    _SLICE_PROMPT = (
        "Extract the Japanese text from this image and translate to Traditional Chinese. "
        "Output ONLY valid JSON: {\"original\": \"...\", \"translation\": \"...\"}. "
        "If the image contains NO TEXT, output an empty JSON {}."
    )

    def _process_single_blk(self, idx: int, blk: TextBlock,
                             cropped: np.ndarray, log_prefix: str):
        used_grok = False
        resp = self._call_ocr(cropped, custom_prompt=self._SLICE_PROMPT)

        if resp == 'BLOCKED_BY_SAFETY':
            resp = self._call_ocr_grok(cropped, self._SLICE_PROMPT, log_prefix,
                                       silent=True)
            if not resp or resp.isdigit():
                self.logger.error(f"{log_prefix} 切片 {idx+1} 失敗: {resp or '?'}")
                self._emit(OcrEventType.ERROR)
                return idx, None
            used_grok = True
            used_grok = True

        if not resp:
            self._emit(OcrEventType.ERROR)
            return idx, None

        try:
            clean = re.sub(r'```json\s*|\s*```', '', resp).strip()
            if not clean or clean in ('{}', '[]'):
                self._emit(OcrEventType.ERROR)
                return idx, None
            data = json.loads(clean)
            if isinstance(data, list):
                data = data[0] if data else {}
            if isinstance(data, dict) and data.get('original'):
                blk.text = [data['original']]
                blk.translation = data.get('translation', '')
                self._apply_font_size(blk, data['original'])
                self._emit(OcrEventType.GROK_OK if used_grok else OcrEventType.SLICE_OK)
                return idx, blk
        except Exception as e:
            self.logger.warning(f"{log_prefix} 切片 {idx+1} 解析失敗: {e}")

        self._emit(OcrEventType.ERROR)
        return idx, None

    def _run_slice_plan(self, img: np.ndarray, blk_list: List[TextBlock],
                         log_prefix: str):
        h, w = img.shape[:2]
        pad = 12
        tasks = []
        for i, blk in enumerate(blk_list):
            bx1, by1, bx2, by2 = blk.xyxy
            px1 = max(0, bx1-pad);  py1 = max(0, by1-pad)
            px2 = min(w, bx2+pad);  py2 = min(h, by2+pad)
            if px1 >= px2 or py1 >= py2:
                continue
            tasks.append((i, blk, img[py1:py2, px1:px2]))

        results_map = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {
                ex.submit(self._process_single_blk, i, blk, crop, log_prefix): i
                for i, blk, crop in tasks
            }
            for f in as_completed(futs):
                idx, rb = f.result()
                results_map[idx] = rb

        # 以原始 blk_list 為基準，逐一決定結果
        final = []
        task_indices = {t[0] for t in tasks}
        for i, blk in enumerate(blk_list):
            if i not in task_indices:
                # 座標無效，保留原框填 ●●●
                blk.text = ['●●●']
                blk.translation = '●●●'
                final.append(blk)
            elif results_map.get(i) is not None:
                # OCR 成功
                final.append(results_map[i])
            else:
                # OCR 失敗，保留原框填 ●●●
                blk.text = ['●●●']
                blk.translation = '●●●'
                final.append(blk)

        blk_list.clear()
        blk_list.extend(final)

    # ── 閱讀順序排序（日漫：右→左欄，欄內上→下）────────────
    def _sort_blk_reading_order(self, blk_list: List[TextBlock]) -> List[TextBlock]:
        if not blk_list:
            return blk_list

        def cx(b): return (b.xyxy[0] + b.xyxy[2]) / 2
        def cy(b): return (b.xyxy[1] + b.xyxy[3]) / 2

        img_h_approx = max(b.xyxy[3] for b in blk_list)

        # 欄寬門檻：用框寬中位數，但至少是圖片高度的 3%
        widths = sorted([(b.xyxy[2] - b.xyxy[0]) for b in blk_list])
        median_w = widths[len(widths) // 2]
        col_thresh = max(median_w * 0.75, img_h_approx * 0.03)

        # 貪婪分欄：x 中心點相近的框歸同一欄
        cols = []
        for blk in sorted(blk_list, key=cx, reverse=True):
            placed = False
            for col in cols:
                rep_cx = sum(cx(b) for b in col) / len(col)
                if abs(cx(blk) - rep_cx) <= col_thresh:
                    col.append(blk)
                    placed = True
                    break
            if not placed:
                cols.append([blk])



        # 欄內按 y 排序後，若相鄰兩框 y 距離超過圖片高度 20%，視為跨區塊，強制拆欄
        gap_thresh = img_h_approx * 0.20
        split_cols = []
        for col in cols:
            col_sorted = sorted(col, key=cy)
            current = [col_sorted[0]]
            for blk in col_sorted[1:]:
                if cy(blk) - cy(current[-1]) > gap_thresh:
                    split_cols.append(current)
                    current = [blk]
                else:
                    current.append(blk)
            split_cols.append(current)



        # 拆欄後：先按欄的 y 中心分群（區塊），再各區塊內按 cx 由右到左排
        # 用圖片高度的 20% 當區塊分界門檻（同 gap_thresh）
        split_cols.sort(key=lambda col: sum(cy(b) for b in col) / len(col))

        # 把 cy 相近的欄歸到同一「區塊」
        blocks = []
        current_block = [split_cols[0]]
        for col in split_cols[1:]:
            prev_cy = sum(cy(b) for b in current_block[-1]) / len(current_block[-1])
            this_cy = sum(cy(b) for b in col) / len(col)
            if this_cy - prev_cy > gap_thresh:
                blocks.append(current_block)
                current_block = [col]
            else:
                current_block.append(col)
        blocks.append(current_block)



        # 每個區塊內按 cx 由右到左排
        result = []
        for bi, block in enumerate(blocks):
            block.sort(key=lambda col: -sum(cx(b) for b in col) / len(col))

            for col in block:
                result.extend(col)

        return result

    # ── 主流程 ────────────────────────────────────────────────
    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock],
                       *args, **kwargs):
        if self.client is None or not blk_list:
            return

        self.page_counter += 1
        lp = f"[{self._fmt_imgname(self.current_imgname)}]"

        # 按漫畫閱讀順序排序（右→左欄，欄內上→下）
        # 只影響傳給 LLM 的 index 對應，不改變框本身
        sorted_blks = self._sort_blk_reading_order(blk_list)


        # Plan A：原圖全頁
        result = self._run_fullpage(img, sorted_blks)
        if result == 'ok':
            self._emit(OcrEventType.PLAN_A_OK)
            self.logger.success(f"{lp} Plan A 成功")
            return
        self.logger.warning(f"{lp} Plan A 失敗（原因：{result}），嘗試 Plan B...")

        # Plan B：塗黑全頁
        if self.enable_masked_plan:
            masked = self._make_masked_img(img, blk_list)
            result = self._run_fullpage(masked, sorted_blks)
            if result == 'ok':
                self._emit(OcrEventType.PLAN_A2_OK)
                self.logger.success(f"{lp} Plan B 成功")
                return
            self.logger.warning(f"{lp} Plan B 失敗（原因：{result}），嘗試 Plan C...")
        else:
            self.logger.warning(f"{lp} Plan B 已停用，直接進 Plan C...")

        # Plan C：切片（含切片層級的 Grok 備援）
        if self.enable_slice_plan:
            self.logger.info(f"{lp} 進入 Plan C 切片模式")
            self._run_slice_plan(img, blk_list, lp)
            return  # 切片完成，不論成敗都結束

        # Plan D：Grok 全頁備援（僅切片停用時）
        if self.fallback_api_key:
            self.logger.warning(f"{lp} Plan D Grok 備援...")
            grok_img = (self._make_masked_img(img, blk_list)
                        if self.enable_masked_plan else img)
            h, w = grok_img.shape[:2]
            resp = self._call_ocr_grok(grok_img, self._PAGE_PROMPT, lp)
            if resp and self._parse_fullpage_result(resp, w, h, blk_list):
                self._emit(OcrEventType.GROK_OK)
                return
            self.logger.error(f"{lp} Plan D 也失敗，此頁放棄")
            self._emit(OcrEventType.ERROR)
        else:
            self.logger.error(f"{lp} 所有方案失敗，此頁放棄")
            self._emit(OcrEventType.ERROR)

    def ocr_img(self, img: np.ndarray) -> str:
        return self._call_ocr(img)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ('provider', 'api_key', 'model', 'base_url'):
            self._build_client()
            self.page_counter = 0