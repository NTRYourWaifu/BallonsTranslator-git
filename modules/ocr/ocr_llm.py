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
        'prompt':        {'value': 'Hybrid Mode Enabled. (This prompt is overridden internally)',
                          'description': '提示詞已由程式內部接管'},
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
    def prompt(self) -> str:     return self.params['prompt']['value']
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
    def _call_ocr(self, img: np.ndarray, custom_prompt: str = None) -> str:
        """回傳: 原始字串 | '' | 'BLOCKED_BY_SAFETY'"""
        if self.use_fallback_only:
            return 'BLOCKED_BY_SAFETY'
        if self.client is None:
            return ''
        img_b64 = _img_to_base64(img)
        target_prompt = custom_prompt or self.prompt
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
                    self.logger.error(f"API 錯誤: {e}")
                    return ''
        self.logger.error("重試耗盡。")
        return ''

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

    # ── 解析全頁結果 ──────────────────────────────────────────
    def _parse_fullpage_result(self, response_text: str,
                                img_w: int, img_h: int,
                                blk_list: List[TextBlock]) -> bool:
        try:
            clean = re.sub(r'```json\s*|\s*```', '', response_text).strip()
            results = json.loads(clean)
            if not isinstance(results, list):
                return False

            valid_blks, used_ids = [], set()
            for item in results:
                if 'box_2d' not in item or 'original' not in item:
                    continue
                ymin, xmin, ymax, xmax = item['box_2d']
                gx1 = int(xmin * img_w / 1000)
                gy1 = int(ymin * img_h / 1000)
                gx2 = int(xmax * img_w / 1000)
                gy2 = int(ymax * img_h / 1000)

                overlapping = [
                    blk for blk in blk_list
                    if id(blk) not in used_ids
                    and (gx1-30) <= (blk.xyxy[0]+blk.xyxy[2])/2 <= (gx2+30)
                    and (gy1-30) <= (blk.xyxy[1]+blk.xyxy[3])/2 <= (gy2+30)
                ]
                if overlapping:
                    main = max(overlapping,
                               key=lambda b: (b.xyxy[2]-b.xyxy[0])*(b.xyxy[3]-b.xyxy[1]))
                    main.text = [item['original']]
                    main.translation = item.get('translation', '')
                    self._apply_font_size(main, item['original'])
                    valid_blks.append(main)
                    for b in overlapping:
                        used_ids.add(id(b))

            if valid_blks:
                blk_list.clear()
                blk_list.extend(valid_blks)
                return True
            return False
        except Exception as e:
            self.logger.warning(f"全頁解析失敗: {e}")
            return False

    # ── 全頁模式（單次） ──────────────────────────────────────
    _PAGE_PROMPT = (
        "Detect all text bubbles in this manga page. "
        "Extract original Japanese and translate to Traditional Chinese. "
        "Output ONLY valid JSON: "
        "[{\"box_2d\": [ymin, xmin, ymax, xmax], "
        "\"original\": \"...\", \"translation\": \"...\"}]. "
        "Coordinates must be normalized (0-1000)."
    )

    def _run_fullpage(self, img: np.ndarray, blk_list: List[TextBlock]) -> str:
        """回傳 'ok' | 'blocked' | 'failed'"""
        h, w = img.shape[:2]
        resp = self._call_ocr(img, custom_prompt=self._PAGE_PROMPT)
        if resp == 'BLOCKED_BY_SAFETY':
            return 'blocked'
        if not resp:
            return 'failed'
        if self._parse_fullpage_result(resp, w, h, blk_list):
            return 'ok'
        return 'failed'

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

    # ── 主流程 ────────────────────────────────────────────────
    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock],
                       *args, **kwargs):
        if self.client is None or not blk_list:
            return

        self.page_counter += 1
        lp = f"[{self._fmt_imgname(self.current_imgname)}]"

        # Plan A：原圖全頁
        result = self._run_fullpage(img, blk_list)
        if result == 'ok':
            self._emit(OcrEventType.PLAN_A_OK)
            self.logger.success(f"{lp} Plan A 成功")
            return

        # Plan B：塗黑全頁
        if self.enable_masked_plan:
            masked = self._make_masked_img(img, blk_list)
            result = self._run_fullpage(masked, blk_list)
            if result == 'ok':
                self._emit(OcrEventType.PLAN_A2_OK)
                self.logger.warning(f"{lp} Plan B 成功")
                return

        # Plan C：切片（含切片層級的 Grok 備援）
        if self.enable_slice_plan:
            self._run_slice_plan(img, blk_list, lp)
            return  # 切片完成，不論成敗都結束（失敗框已保留、已計error）

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