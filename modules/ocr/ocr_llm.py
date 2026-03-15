import os
import math
import numpy as np
import cv2
import base64
import json
import time
import requests
import re
import threading
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from qtpy.QtCore import Signal, QObject

from .base import register_OCR, OCRBase, TextBlock


# ── 統計事件類型 ──────────────────────────────────────────────
class OcrEventType:
    PLAN_A_OK  = 'plan_a_ok'   # 綠  原圖全頁成功
    FONT_WARN  = 'plan_a2_ok'  # 黃  [借用舊版信號] 字型異常警告 (過大或過小)
    SLICE_OK   = 'slice_ok'    # 橘  Plan B 切片（主API）成功
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
        'debug_log': {'type': 'checkbox', 'value': False,
                      'description': '輸出 OCR 流程 log（Plan A/B/C 結果、解析失敗等）'},
        'debug_font_log': {'type': 'checkbox', 'value': False,
                           'description': '輸出每框字型推算過程（原文行數、fs 計算路徑）'},
        'fallback_api_key': {'value': '', 'description': '備援 Grok API 金鑰'},
        'fallback_model':   {'value': 'grok-4.20-beta-0309-non-reasoning',
                             'description': '備援模型，需支援 vision'},
        'disable_plan_a': {'type': 'checkbox', 'value': False,
                           'description': '測試用：停用 Plan A（強制跳到 Plan B）'},
        'disable_plan_b': {'type': 'checkbox', 'value': False,
                           'description': '測試用：停用 Plan B（強制跳到 Plan C）'},
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.client = None
        self.last_request_time = 0
        
        # 速率控制屬性初始化 (從 GeminiClient 移至此處)
        self._api_lock = threading.Lock()
        self._last_api_time = 0.0
        
        self.page_counter = 0
        self.current_imgname = ''
        self.current_img_dir = ''
        self.stats_signals = OcrStatsSignals()
        self._build_client()

    def _fmt_imgname(self, name: str) -> str:
        """超過9字元縮寫：前3字元 + ... + 後3字元（不含副檔名部分補上副檔名）"""
        if len(name) <= 9:
            return name
        ext_idx = name.rfind('.')
        ext = name[ext_idx:] if ext_idx != -1 else ''
        stem = name[:ext_idx] if ext_idx != -1 else name
        return f'{stem[:3]}...{stem[-3:]}{ext}'

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
    def debug_log(self) -> bool: return bool(self.params['debug_log']['value'])
    @property
    def debug_font_log(self) -> bool:
        return bool(self.params.get('debug_font_log', {}).get('value', False))
    @property
    def fallback_api_key(self) -> str:  return self.params['fallback_api_key']['value']
    @property
    def fallback_model(self) -> str:    return self.params['fallback_model']['value']
    @property
    def disable_plan_a(self) -> bool:   return bool(self.params['disable_plan_a']['value'])
    @property
    def disable_plan_b(self) -> bool:   return bool(self.params['disable_plan_b']['value'])

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
        with self._api_lock:
            elapsed = time.time() - self._last_api_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last_api_time = time.time()

    # ── 字體大小 ──────────────────────────────────────────────

    @staticmethod
    def _count_lines(text: str, fs: float, main_axis: float) -> int:
        """
        給定字型大小 fs 和主軸長度，推算原文會換幾行/欄。

        邏輯：
          1. 按 \\n 切成硬換行段落
          2. 每段字數 × fs > main_axis 時自動換行，累加行數
          3. 回傳總行/欄數

        text     : 原文（日文 OCR 結果）
        fs       : 估計字型大小（像素）
        main_axis: 橫排 = box_w，直排 = box_h
        """
        if fs <= 0 or main_axis <= 0:
            return 1
        chars_per_line = max(1, int(main_axis / fs))
        lines = text.split('\n')
        total = 0
        for seg in lines:
            seg_len = max(len(re.sub(r'\s', '', seg)), 1)
            total += math.ceil(seg_len / chars_per_line)
        return max(total, 1)

    def _calc_fs_from_source(self, blk: TextBlock, src_text: str) -> float:
        """
        用 OCR 原文（日文）推算合理字型大小。

        步驟：
          1. 以 _detected_font_size 為初始 fs 猜測值
             若無則用框的短邊 / 2 當起點
          2. 迭代兩輪：
             用 fs 算出行/欄數 → 用行/欄數反推 fs
          3. 乘以 font_size_ratio 回傳

        橫排：主軸 = box_w，副軸 = box_h，fs = box_h / 行數
        直排：主軸 = box_h，副軸 = box_w，fs = box_w / 欄數
        """
        x1, y1, x2, y2 = blk.xyxy
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        orig_w = getattr(blk, '_detected_font_size', 0)

        if blk.vertical:
            main_axis  = box_h   # 每欄沿 Y 方向延伸
            cross_axis = box_w   # 欄數沿 X 方向
        else:
            main_axis  = box_w   # 每行沿 X 方向延伸
            cross_axis = box_h   # 行數沿 Y 方向

        # 初始 fs 猜測
        if 0 < orig_w < cross_axis * 0.95:
            fs_guess = orig_w
        else:
            fs_guess = cross_axis / 2.0

        # 迭代兩輪收斂
        fs = fs_guess
        for _ in range(2):
            n = self._count_lines(src_text, fs, main_axis)
            fs = cross_axis / n

        return max(8.0, min(250.0, fs * self.font_size_ratio))

    def _auto_font_size(self, blk: TextBlock, text: str = '') -> float:
        """
        計算字型大小。

        優先使用 _calc_fs_from_source（原文反推），
        fallback 到舊的面積開根號估算。
        """
        x1, y1, x2, y2 = blk.xyxy
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        orig_w = getattr(blk, '_detected_font_size', 0)

        # 有原文 → 用原文行數反推（B-1 新邏輯）
        if text:
            return self._calc_fs_from_source(blk, text)

        # 無原文 fallback：沿用舊邏輯
        # （_detected_font_size 可信時直接用，否則面積開根號估算）
        if 0 < orig_w < box_w * 0.8:
            return max(8.0, min(250.0, orig_w * self.font_size_ratio))

        return max(8.0, min(250.0,
            math.sqrt(box_w * box_h) / 4 * self.font_size_ratio
        ))

    def _apply_font_size(self, blk: TextBlock, text: str = ''):
        orig_w = getattr(blk, '_detected_font_size', 0)
        fs = self._auto_font_size(blk, text)
        x1, y1, x2, y2 = blk.xyxy
        box_w = x2 - x1
        box_h = y2 - y1

        is_abnormal = fs < 12.0 or fs > 65.0

        if is_abnormal:
            self.logger.warning(
                f"⚠️字型異常: fs={fs:.1f} "
                f"(orig_w={orig_w:.1f}, box={box_w}x{box_h}, "
                f"vert={blk.vertical}) | text={text[:12]!r}"
            )
            self._emit(OcrEventType.FONT_WARN)

        if self.debug_font_log:
            # 重新跑一次 _count_lines 拿行數，僅供 log 顯示用
            if text and blk.vertical:
                n = self._count_lines(text, fs / self.font_size_ratio, box_h)
                axis_info = f"box_h={box_h} → {n}欄 → fs={box_w/n:.1f}"
            elif text:
                n = self._count_lines(text, fs / self.font_size_ratio, box_w)
                axis_info = f"box_w={box_w} → {n}行 → fs={box_h/n:.1f}"
            else:
                axis_info = 'no_text→fallback'
            path = 'src→lines' if text else ('orig_w' if 0 < orig_w < box_w * 0.8 else 'fallback')
            self.logger.debug(
                f"[font] {path} | orig_w={orig_w:.1f} {axis_info}"
                f" ×ratio={self.font_size_ratio} → fs={fs:.1f}"
                f" {'⚠️' if is_abnormal else '✓'}"
                f" | vert={blk.vertical} xyxy={blk.xyxy}"
                f" | text={text[:12]!r}"
            )

        blk.font_size = fs

    # ── 底層 API 呼叫 ─────────────────────────────────────────
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
        if self.client is None:
            return 'ERR:未設定API金鑰'
        img_b64 = _img_to_base64(img)
        target_prompt = custom_prompt
        
        # 固定遞增重試等待
        max_retries = 3
        for attempt in range(max_retries):
            self._respect_delay()
            try:
                return self.client.ocr(img_b64, target_prompt)
            except Exception as e:
                err = str(e)
                if '429' in err or 'exhausted' in err.lower() or 'quota' in err.lower():
                    wait = 1.5 * (attempt + 1)
                    self.logger.warning(f"限速，暫停 {wait:.1f}s 重試 ({attempt+1}/{max_retries})...")
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

            return matched > 0
        except Exception as e:
            self.logger.warning(f"全頁解析失敗: {e}")
            return False

    # ── 全頁模式（單次） ──────────────────────────────────────
    _GRID_CELL = 192
    _GRID_PAD  = 28

    _GRID_PROMPT = (
        "This image is a grid of manga text box crops.\n"
        "Each cell has a black label on its LEFT side showing its index number.\n"
        "Use the label numbers to identify each cell — do NOT count or guess order.\n"
        "Read the Japanese text in each cell and translate to Traditional Chinese.\n"
        "\n"
        "Translation rules:\n"
        "- Translate the original text directly. Do NOT add any parenthetical notes, explanations, or romanizations.\n"
        "- Output ONLY a valid JSON array, one entry per cell:\n"
        '[{"index": 0, "direction": "v or h", "original": "...", "translation": "..."}, ...]\n'
        'direction: "v"=vertical/tategumi, "h"=horizontal/yokogumi.\n'
        'If a cell has no readable text: {"index": N, "direction": "v", "original": "", "translation": ""}\n'
        "Total cells: {n}"
    )

    def _build_grid_img(self, img: np.ndarray,
                        blk_list: List[TextBlock]) -> np.ndarray:
        h_img, w_img = img.shape[:2]
        label_w = max(20, int(w_img * 0.018))
        gap = max(6, int(w_img * 0.005))

        crops = []
        for blk in blk_list:
            bx1, by1, bx2, by2 = blk.xyxy
            px1 = max(0, bx1 - 4); py1 = max(0, by1 - 4)
            px2 = min(w_img, bx2 + 4); py2 = min(h_img, by2 + 4)
            crop = img[py1:py2, px1:px2]
            crops.append(crop if crop.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8))

        total_area = sum((c.shape[1] + label_w) * c.shape[0] for c in crops)
        target_w = max(int(math.sqrt(total_area)), 100)

        order = sorted(range(len(crops)), key=lambda i: -crops[i].shape[0])
        rows = []
        row_ws = []
        row_hs = []

        for orig_idx in order:
            crop = crops[orig_idx]
            ch, cw = crop.shape[:2]
            fw = cw + label_w

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

        canvas_w = max(row_ws) + gap
        canvas_h = sum(row_hs) + (len(rows) + 1) * gap
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        visual_order = []
        for r, row in enumerate(rows):
            for orig_idx, _ in reversed(row):
                visual_order.append(orig_idx)
        orig_to_visual = {orig_idx: vi for vi, orig_idx in enumerate(visual_order)}

        font_thick = 1
        y_cursor = gap
        for r, row in enumerate(rows):
            row_actual_w = sum(label_w + crop.shape[1] for _, crop in row) + gap * (len(row) - 1)
            x_cursor = canvas_w - gap - row_actual_w

            for orig_idx, crop in row:
                ch, cw = crop.shape[:2]
                row_h = row_hs[r]

                canvas[y_cursor:y_cursor+row_h, x_cursor:x_cursor+label_w] = 0
                vi = orig_to_visual[orig_idx]
                label = str(vi)
                font_scale = max(0.3, min(label_w / 40, row_h * 0.6 / 20))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                tx = x_cursor + max(0, (label_w - tw) // 2)
                ty = y_cursor + (row_h + th) // 2
                cv2.putText(canvas, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

                canvas[y_cursor:y_cursor+ch, x_cursor+label_w:x_cursor+label_w+cw] = crop

                cx1, cy1 = x_cursor, y_cursor
                cx2, cy2 = x_cursor + label_w + cw, y_cursor + row_h
                cv2.rectangle(canvas, (cx1,     cy1),     (cx2,     cy2),     (0,   0,   0), 3)
                cv2.rectangle(canvas, (cx1 + 2, cy1 + 2), (cx2 - 2, cy2 - 2), (0, 220, 255), 2)

                x_cursor += label_w + cw + gap
            y_cursor += row_hs[r] + gap

        img_dir = self.current_img_dir or os.path.dirname(self.current_imgname) or '.'
        img_name = os.path.basename(self.current_imgname) or 'unknown'
        
        debug_dir = os.path.join(img_dir, 'ocr_debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        file_stem = os.path.splitext(img_name)[0]
        safe_name = f"{file_stem}_grid.jpg"
        debug_path = os.path.join(debug_dir, safe_name)
        
        cv2.imwrite(debug_path, canvas)

        return canvas, visual_order

    def _run_fullpage(self, img: np.ndarray, blk_list: List[TextBlock]):
        if self.disable_plan_a:
            return 'SKIP:disabled', 0
        grid_img, visual_order = self._build_grid_img(img, blk_list)
        prompt = self._GRID_PROMPT.replace('{n}', str(len(blk_list)))
        resp = self._call_ocr(grid_img, custom_prompt=prompt)
        if resp == 'BLOCKED_BY_SAFETY':
            return '安全過濾器擋住', 0
        if resp.startswith('ERR:'):
            return resp[4:], 0
        if not resp:
            return 'API無回應', 0
        matched = self._parse_fullpage_result(resp, blk_list, visual_order)
        if matched:
            return 'ok', matched
        return 'JSON解析失敗', 0

    # ── 切片模式 ──────────────────────────────────────────────
    _SLICE_PROMPT = (
        "Extract the Japanese text from this image and translate to Traditional Chinese. "
        "Line breaks: use the original text's line breaks as a loose reference — keep a line break in the translation only if it fits the natural meaning or rhythm. Do NOT add extra line breaks. "
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
        if self.disable_plan_b:
            return 'SKIP:disabled', 0, len(blk_list), list(range(len(blk_list)))
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
            futs = {}
            for i, (orig_idx, blk, crop) in enumerate(tasks):
                futs[ex.submit(self._process_single_blk, orig_idx, blk, crop, log_prefix)] = orig_idx
                # 在分配任務時加入固定延遲，完美錯開併發流量
                if i < len(tasks) - 1:
                    time.sleep(0.3)

            for f in as_completed(futs):
                idx, rb = f.result()
                results_map[idx] = rb

        final = []
        task_indices = {t[0] for t in tasks}
        for i, blk in enumerate(blk_list):
            if i not in task_indices:
                final.append(blk)
            elif results_map.get(i) is not None:
                final.append(results_map[i])
            else:
                final.append(blk)

        ok_count = sum(1 for i in range(len(blk_list)) if results_map.get(i) is not None)
        fail_count = len(blk_list) - ok_count
        failed_indices = [i for i in range(len(blk_list)) if results_map.get(i) is None]

        blk_list.clear()
        blk_list.extend(final)
        return 'ok', ok_count, fail_count, failed_indices

    # ── 閱讀順序排序（日漫：右→左欄，欄內上→下）────────────
    def _sort_blk_reading_order(self, blk_list: List[TextBlock]) -> List[TextBlock]:
        if not blk_list:
            return blk_list

        def cx(b): return (b.xyxy[0] + b.xyxy[2]) / 2
        def cy(b): return (b.xyxy[1] + b.xyxy[3]) / 2

        img_h_approx = max(b.xyxy[3] for b in blk_list)
        widths = sorted([(b.xyxy[2] - b.xyxy[0]) for b in blk_list])
        median_w = widths[len(widths) // 2]
        col_thresh = max(median_w * 0.75, img_h_approx * 0.03)

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

        split_cols.sort(key=lambda col: sum(cy(b) for b in col) / len(col))

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

        sorted_blks = self._sort_blk_reading_order(blk_list)

        # Plan A：原圖全頁
        result, matched = self._run_fullpage(img, sorted_blks)
        if result == 'ok':
            self._emit(OcrEventType.PLAN_A_OK)
            self.logger.success(f"{lp} Plan A 成功（{matched}/{len(blk_list)} 框）")
            return
        self.logger.warning(f"{lp} Plan A：{result}")

        # Plan B：切片（含切片層級的 Grok 備援）
        result, ok, fail, failed_indices = self._run_slice_plan(img, blk_list, lp)
        if result == 'ok':
            total = ok + fail
            if fail == 0:
                self.logger.success(f"{lp} Plan B 成功（{ok}/{total} 框）")
                return
            else:
                self.logger.warning(f"{lp} Plan B 完成（{ok}/{total} 框，{fail} 框失敗）")
        else:
            self.logger.warning(f"{lp} Plan B：{result}")

        # Plan C：Grok 切片補救（只處理 Plan B 失敗的框）
        if self.fallback_api_key and failed_indices:
            h, w = img.shape[:2]
            pad = 12
            grok_client = self._build_fallback_client()
            ok, fail = 0, 0
            for i in failed_indices:
                blk = blk_list[i]
                bx1, by1, bx2, by2 = blk.xyxy
                crop = img[max(0,by1-pad):min(h,by2+pad),
                           max(0,bx1-pad):min(w,bx2+pad)]
                try:
                    resp = grok_client.ocr(_img_to_base64(crop), self._SLICE_PROMPT, timeout=120)
                    clean = re.sub(r'```json\s*|\s*```', '', resp).strip()
                    data = json.loads(clean)
                    if isinstance(data, list): data = data[0] if data else {}
                    if isinstance(data, dict) and data.get('original'):
                        blk.text = [data['original']]
                        blk.translation = data.get('translation', '')
                        self._apply_font_size(blk, data['original'])
                        self._emit(OcrEventType.GROK_OK)
                        ok += 1
                        continue
                except Exception as e:
                    self.logger.warning(f"{lp} Plan C 框{i+1} 失敗: {self._explain_error(str(e))}")
                blk.text = ['●●●']
                blk.translation = '●●●'
                fail += 1
                self._emit(OcrEventType.ERROR)
            if fail == 0:
                self.logger.success(f"{lp} Plan C 成功（{ok}/{len(failed_indices)} 框）")
            else:
                self.logger.warning(f"{lp} Plan C 完成（{ok}/{len(failed_indices)} 框，{fail} 框失敗）")
            return
            
        for i in failed_indices:
            blk_list[i].text = ['●●●']
            blk_list[i].translation = '●●●'
        self.logger.error(f"{lp} 所有方案失敗，此頁放棄")
        self._emit(OcrEventType.ERROR)

    def ocr_img(self, img: np.ndarray) -> str:
        return self._call_ocr(img)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ('provider', 'api_key', 'model', 'base_url'):
            self._build_client()
            self.page_counter = 0