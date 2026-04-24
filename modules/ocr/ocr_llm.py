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
from utils.textblock import resolve_blk_style


# ── 統計事件類型 ──────────────────────────────────────────────
class OcrEventType:
    PLAN_A_OK  = 'plan_a_ok'   # 綠  原圖全頁成功
    FONT_WARN  = 'plan_a2_ok'  # 黃△ 字型嚴重過小（雙條件同時成立）
    SLICE_OK   = 'slice_ok'    # 橘  Plan B 切片（主API）成功
    GROK_OK    = 'grok_ok'     # 粉  Grok 成功（切片或全頁）
    ERROR      = 'error'       # 紅  最終放棄此頁
    DET_WARN   = 'det_warn'    # 黃■ 字型疑似過小（單條件成立）
    BOX_OOB    = 'box_oob'    # 紅矩 對話框凸出圖片範圍


class OcrStatsSignals(QObject):
    event = Signal(str, str)   # (event_type, imgname)


# ── 圖片工具 ──────────────────────────────────────────────────
def _img_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def _fmt_api_error(data: dict) -> str:
    """從 API 回應 dict 提取最有用的錯誤摘要，避免印出整個 JSON 噪音"""
    if not isinstance(data, dict):
        return str(data)[:120]
    # Gemini / Google 格式: {"error": {"code": 429, "message": "..."}}
    err = data.get('error')
    if isinstance(err, dict):
        code = err.get('code', '?')
        msg  = err.get('message', err.get('status', 'unknown'))
        return f"[{code}] {str(msg)[:120]}"
    if isinstance(err, str):
        return f"error={err[:120]}"
    # OpenAI 格式: {"error": {"message": "...", "type": "...", "code": "..."}}
    # （同上，已被上面 isinstance(err, dict) 涵蓋）
    # 未知格式：只印 keys + 前80字
    return f"keys={list(data.keys())} | {str(data)[:80]}"


# ── API 客戶端 ────────────────────────────────────────────────
class GeminiClient:
    BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

    def __init__(self, api_key: str, model: str = 'gemini-3.1-flash-lite-preview',
                 thinking_budget: int = 0, thinking_level: bool = False,
                 thinking_level_gemini3: str = 'medium'):
        self.api_key = api_key
        self.model = model
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level              # bool（gemma-4：True=high, False=不啟用）
        self.thinking_level_gemini3 = thinking_level_gemini3  # minimal/low/medium/high（gemini-3.x 專用）

    _GEMMA4_MODELS  = {'gemma-4-31b-it', 'gemma-4-26b-a4b-it'}
    _GEMINI3_MODELS = {'gemini-flash-latest', 'gemini-3-flash-preview', 'gemini-3.1-flash-lite-preview'}

    def ocr(self, img_b64: str, prompt: str, timeout: int = 120) -> str:
        url = self.BASE_URL.format(model=self.model) + f'?key={self.api_key}'
        is_gemma4   = self.model in self._GEMMA4_MODELS
        is_gemini3  = self.model in self._GEMINI3_MODELS

        if is_gemma4:
            # gemma-4：不傳 thinkingConfig = 不啟用思考；傳 high = 啟用
            gen_config = {'temperature': 0.0, 'response_mime_type': 'application/json'}
            if self.thinking_level:
                gen_config['thinkingConfig'] = {'thinkingLevel': 'high'}
        elif is_gemini3:
            # Gemini 3.x 原生思考模型：thinkingLevel（low/medium/high）
            # none = 不傳 thinkingConfig，讓模型使用預設
            gen_config = {'temperature': 1.0, 'response_mime_type': 'application/json'}
            if self.thinking_level_gemini3 and self.thinking_level_gemini3 != 'none':
                gen_config['thinkingConfig'] = {'thinkingLevel': self.thinking_level_gemini3}
        else:
            gen_config = {'temperature': 0.0, 'response_mime_type': 'application/json'}
            # 2.5 系列：thinkingBudget 整數（0=關閉）
            if 'gemini-2.5' in self.model and self.thinking_budget > 0:
                gen_config['thinkingConfig'] = {'thinkingBudget': self.thinking_budget}

        payload = {
            'contents': [{'parts': [
                {'text': prompt},
                {'inline_data': {'mime_type': 'image/jpeg', 'data': img_b64}}
            ]}],
            'generationConfig': gen_config,
            'safetySettings': [
                {'category': 'HARM_CATEGORY_HARASSMENT',        'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_HATE_SPEECH',       'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
            ]
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            err_data = resp.json() if resp.content else {}
            if resp.status_code not in (503, 500):
                import logging; logging.getLogger(__name__).error(f"[GeminiClient] {resp.status_code} {self.model} => {_fmt_api_error(err_data)}")
            raise RuntimeError(f"API Error {resp.status_code}: {_fmt_api_error(err_data)}")
        data = resp.json()
        try:
            if 'promptFeedback' in data and 'blockReason' in data['promptFeedback']:
                raise RuntimeError(f"Blocked: {data['promptFeedback']['blockReason']}")
            return data['candidates'][0]['content']['parts'][0]['text'].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Gemini 回應異常: {_fmt_api_error(data)}") from e


class OpenAICompatClient:
    def __init__(self, api_key: str, model: str = 'gpt-4o',
                 base_url: str = 'https://api.openai.com/v1'):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')

    def ocr(self, img_b64: str, prompt: str, timeout: int = 45,
            allow_array: bool = False) -> str:
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
        }
        if not allow_array:
            payload['response_format'] = {'type': 'json_object'}
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            import logging; logging.getLogger(__name__).error(f"[OpenAICompatClient] {resp.status_code} {self.model} => {_fmt_api_error(resp.json() if resp.content else {})}")
            raise RuntimeError(f"API Error {resp.status_code}: {_fmt_api_error(resp.json() if resp.content else {})}")
        data = resp.json()
        try:
            return data['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f'OpenAI API 回應異常: {_fmt_api_error(data)}') from e


# ── 主模組 ────────────────────────────────────────────────────
@register_OCR('llm_ocr')
class OCRLlm(OCRBase):
    _MODEL_OPTIONS = [
        'gemini-flash-latest',
        'gemini-3-flash-preview',
        'gemini-3.1-flash-lite-preview',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
        'gemma-4-31b-it',
        'gemma-4-26b-a4b-it',
        'grok-4.20-0309-reasoning',
        'grok-4.20-0309-non-reasoning',
        'grok-4-1-fast-reasoning',
        'grok-4-1-fast-non-reasoning',
        '',  # 空 = 不使用（模型名稱錯誤）
    ]

    params = {
        'gemini_api_key': {'value': '', 'description': 'Gemini API 金鑰（用於 gemini-* 模型）'},
        'grok_api_key':   {'value': '', 'description': 'xAI API 金鑰（用於 grok-* 模型）'},
        'model': {
            'type': 'selector',
            'options': _MODEL_OPTIONS,
            'value': 'gemini-3.1-flash-lite-preview',
            'description': '主模型（空 = 不使用）'
        },
        'thinking_budget': {'value': '1024',
                            'description': '主模型 Gemini 2.5 思考模式 token 上限（0=關閉；建議 512~2048；最大 24576）',
                            'controlled_by': 'model',
                            'visible_for_models': ['gemini-2.5']},
        'thinking_level_gemini3': {'type': 'selector',
                            'options': ['minimal', 'low', 'medium', 'high'],
                            'value': 'medium',
                            'description': '主模型 Gemini 3.x 思考深度（minimal=最低延遲；high=最深推理）',
                            'controlled_by': 'model',
                            'visible_for_models': ['gemini-3', 'gemini-flash-latest']},
        'thinking_level':  {'type': 'checkbox', 'value': False,
                            'description': '主模型 Gemma 4 開啟思考模式（勾選=high，不勾=不啟用）',
                            'controlled_by': 'model',
                            'visible_for_models': ['gemma-4']},
        'fallback_model': {
            'type': 'selector',
            'options': _MODEL_OPTIONS,
            'value': 'grok-4.20-0309-non-reasoning',
            'description': '備援模型（空 = 不使用備援）'
        },
        'fallback_thinking_budget': {'value': '1024',
                            'description': '備援模型 Gemini 2.5 思考模式 token 上限（0=關閉；建議 512~2048；最大 24576）',
                            'controlled_by': 'fallback_model',
                            'visible_for_models': ['gemini-2.5']},
        'fallback_thinking_level_gemini3': {'type': 'selector',
                            'options': ['minimal', 'low', 'medium', 'high'],
                            'value': 'medium',
                            'description': '備援模型 Gemini 3.x 思考深度（minimal=最低延遲；high=最深推理）',
                            'controlled_by': 'fallback_model',
                            'visible_for_models': ['gemini-3', 'gemini-flash-latest']},
        'fallback_thinking_level': {'type': 'checkbox', 'value': False,
                            'description': '備援模型 Gemma 4 開啟思考模式（勾選=high，不勾=不啟用）',
                            'controlled_by': 'fallback_model',
                            'visible_for_models': ['gemma-4']},
        'delay':         {'value': '0.0', 'description': '請求間隔（付費版可設為 0）'},
        'max_workers':   {'value': '5', 'description': '切片模式並行數（建議 5~10）'},
        'font_size_scale': {'value': '1.0', 'description': '字體大小縮放（0.5 縮小一半；1.0 不縮放；1.2 放大 20%）'},
        'char_scale_table': {'value': '3:0.04:0.40, 6:0.06:0.45, 12:0.08:0.50, 22:0.10:0.55', 'description': '字數縮放表：「字數:佔比閾值:縮放係數」用逗號分隔。字數落在對應段時，佔比超過該段閾值才縮，縮放係數直接乘在原字體上'},
        'handwritten_size_ratio': {'value': '1.5', 'description': '手寫字：實際渲染大小與 LLM 估算大小的比值超過此倍數（過大）或低於此倍數的倒數（過小）才警告（預設 1.5）'},
        'save_grid_debug': {'type': 'checkbox', 'value': False,
                            'description': '將每頁送給 LLM 的 grid 拼圖存到專案目錄下的 ocr_debug/ 資料夾'},
        'plan_a_retry_seconds': {'value': '120',
                                 'description': 'Plan A 過載/逾時重試上限（秒，預設 120）'},
        'plan_b_retry_seconds': {'value': '300',
                                 'description': 'Plan B 試探框過載/逾時重試上限（秒，預設 300）'},
        'plan_c_retry_seconds': {'value': '60',
                                 'description': 'Plan C 試探框過載/逾時重試上限（秒，預設 60）'},
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
        self.stop_flag = False
        self._build_client()

    def _fmt_imgname(self, name: str) -> str:
        """超過9字元縮寫：前3字元 + ... + 後3字元（不含副檔名部分補上副檔名）"""
        if len(name) <= 9:
            return name
        ext_idx = name.rfind('.')
        ext = name[ext_idx:] if ext_idx != -1 else ''
        stem = name[:ext_idx] if ext_idx != -1 else name
        return f'{stem[:3]}...{stem[-3:]}{ext}'

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def _is_gemini(model: str) -> bool:
        return model.startswith('gemini') or model.startswith('gemma')

    @staticmethod
    def _xai_base_url() -> str:
        return 'https://api.x.ai/v1'

    # ── properties ───────────────────────────────────────────
    @property
    def gemini_api_key(self) -> str: return self.params['gemini_api_key']['value']
    @property
    def grok_api_key(self) -> str:   return self.params['grok_api_key']['value']
    @property
    def model(self) -> str:      return self.params['model']['value']
    @property
    def delay(self) -> float:
        try: return float(self.params['delay']['value'])
        except: return 0.0
    @property
    def max_workers(self) -> int:
        try: return int(self.params['max_workers']['value'])
        except: return 5
    @property
    def font_size_scale(self) -> float:
        try: return float(self.params['font_size_scale']['value'])
        except: return 1.0
    @property
    def char_scale_table(self) -> list:
        """解析 '3:0.04:0.40, 6:0.06:0.45' → [(3,0.04,0.40),(6,0.06,0.45),...]"""
        try:
            raw = self.params['char_scale_table']['value']
            result = []
            for seg in raw.split(','):
                parts = [p.strip() for p in seg.strip().split(':')]
                if len(parts) == 3:
                    result.append((int(parts[0]), float(parts[1]), float(parts[2])))
            return sorted(result, key=lambda x: x[0])
        except:
            return [(3, 0.04, 0.40), (6, 0.06, 0.45), (12, 0.08, 0.50), (22, 0.10, 0.55)]
    @property
    def handwritten_size_ratio(self) -> float:
        try: return float(self.params['handwritten_size_ratio']['value'])
        except: return 1.5
    @property
    def save_grid_debug(self) -> bool:
        return bool(self.params.get('save_grid_debug', {}).get('value', False))
    @property
    def fallback_model(self) -> str:    return self.params['fallback_model']['value']
    @property
    def plan_a_retry_seconds(self) -> int:
        try: return int(self.params['plan_a_retry_seconds']['value'])
        except: return 120
    @property
    def plan_b_retry_seconds(self) -> int:
        try: return int(self.params['plan_b_retry_seconds']['value'])
        except: return 300
    @property
    def plan_c_retry_seconds(self) -> int:
        try: return int(self.params['plan_c_retry_seconds']['value'])
        except: return 60
    @property
    def disable_plan_a(self) -> bool:   return bool(self.params['disable_plan_a']['value'])
    @property
    def disable_plan_b(self) -> bool:   return bool(self.params['disable_plan_b']['value'])
    @property
    def thinking_budget(self) -> int:
        try: return int(self.params['thinking_budget']['value'])
        except: return 0
    @property
    def thinking_level(self) -> bool:
        v = self.params.get('thinking_level', {}).get('value', False)
        if isinstance(v, bool): return v
        return str(v).lower().strip() == 'true'
    @property
    def thinking_level_gemini3(self) -> str:
        try: return str(self.params['thinking_level_gemini3']['value'])
        except: return 'medium'
    @property
    def fallback_thinking_budget(self) -> int:
        try: return int(self.params['fallback_thinking_budget']['value'])
        except: return 0
    @property
    def fallback_thinking_level(self) -> bool:
        v = self.params.get('fallback_thinking_level', {}).get('value', False)
        if isinstance(v, bool): return v
        return str(v).lower().strip() == 'true'
    @property
    def fallback_thinking_level_gemini3(self) -> str:
        try: return str(self.params['fallback_thinking_level_gemini3']['value'])
        except: return 'medium'

    # ── 統計事件 ──────────────────────────────────────────────
    def _emit(self, event_type: str, imgname: str = ''):
        self.stats_signals.event.emit(event_type, imgname or self.current_imgname)

    # ── 客戶端 ────────────────────────────────────────────────
    def _api_key_for(self, model: str) -> str:
        return self.gemini_api_key if self._is_gemini(model) else self.grok_api_key

    def _build_client(self):
        if not self.model: return
        key = self._api_key_for(self.model)
        if not key: return
        if self._is_gemini(self.model):
            self.client = GeminiClient(api_key=key, model=self.model,
                                       thinking_budget=self.thinking_budget,
                                       thinking_level=self.thinking_level,
                                       thinking_level_gemini3=self.thinking_level_gemini3)
        else:
            self.client = OpenAICompatClient(api_key=key, model=self.model, base_url=self._xai_base_url())

    def _build_fallback_client(self):
        if not self.fallback_model: return None
        key = self._api_key_for(self.fallback_model)
        if not key: return None
        if self._is_gemini(self.fallback_model):
            return GeminiClient(api_key=key, model=self.fallback_model,
                                thinking_budget=self.fallback_thinking_budget,
                                thinking_level=self.fallback_thinking_level,
                                thinking_level_gemini3=self.fallback_thinking_level_gemini3)
        return OpenAICompatClient(api_key=key, model=self.fallback_model, base_url=self._xai_base_url())

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

    def _call_ocr(self, img: np.ndarray, custom_prompt: str = None,
                  allow_array: bool = False,
                  max_overload_seconds: int = 120,
                  max_timeout_seconds: int = 120,
                  client=None) -> str:
        """回傳: 原始字串 | 'BLOCKED_BY_SAFETY' | 'ERR:原因'"""
        _client = client if client is not None else self.client
        if _client is None:
            return 'ERR:未設定API金鑰'
        img_b64 = _img_to_base64(img)
        target_prompt = custom_prompt

        # 重試策略：503/500 每5s重試，累計超過 max_overload_seconds 放棄
        # 429/timeout：5→10→15s 遞增，累計超過 max_timeout_seconds 放棄
        _STEP_WAITS = [5.0, 10.0, 15.0]

        overload_elapsed  = 0.0
        overload_attempt  = 0
        ratelimit_elapsed = 0.0
        ratelimit_attempt = 0
        timeout_elapsed   = 0.0
        timeout_attempt   = 0
        _retry_printed    = False  # 是否用 \r 模式印過，成功時需補換行

        while True:
            if self.stop_flag:
                if _retry_printed: print()
                return 'ERR:已停止'
            self._respect_delay()
            _t0 = time.monotonic()
            try:
                if isinstance(_client, OpenAICompatClient):
                    result = _client.ocr(img_b64, target_prompt, timeout=120, allow_array=allow_array)
                else:
                    result = _client.ocr(img_b64, target_prompt, timeout=120)
                if _retry_printed: print()  # 補換行讓後續 log 不亂
                return result
            except Exception as e:
                _elapsed = time.monotonic() - _t0
                err = str(e)
                if '503' in err or 'unavailable' in err.lower() or '500' in err or 'internal server' in err.lower():
                    overload_elapsed += _elapsed + 5.0
                    overload_attempt += 1
                    if overload_elapsed > max_overload_seconds:
                        print()  # 結束覆寫行
                        break
                    max_attempts_display = max(1, int(max_overload_seconds / 5))
                    if overload_attempt == 1:
                        self.logger.error(f"API服務不可用 ({err[:120]})，開始重試（上限 {max_overload_seconds}s）...")
                    _retry_printed = True
                    print(f"\r[WARNING] API服務不可用，5s 後重試 ({overload_attempt}/{max_attempts_display} 已等 {overload_elapsed:.0f}s/{max_overload_seconds}s)...   ", end='', flush=True)
                    time.sleep(5.0)
                elif '429' in err or 'exhausted' in err.lower() or 'quota' in err.lower():
                    wait = _STEP_WAITS[min(ratelimit_attempt, len(_STEP_WAITS) - 1)]
                    ratelimit_elapsed += _elapsed + wait
                    ratelimit_attempt += 1
                    if ratelimit_elapsed > 120:
                        print()
                        break
                    _retry_printed = True
                    print(f"\r[WARNING] 限速，{wait:.0f}s 後重試 ({ratelimit_attempt} 已等 {ratelimit_elapsed:.0f}s/120s)...   ", end='', flush=True)
                    time.sleep(wait)
                elif 'timeout' in err.lower() or 'timed out' in err.lower():
                    wait = _STEP_WAITS[min(timeout_attempt, len(_STEP_WAITS) - 1)]
                    timeout_elapsed += _elapsed + wait
                    timeout_attempt += 1
                    if timeout_elapsed > max_timeout_seconds:
                        print()
                        break
                    _retry_printed = True
                    print(f"\r[WARNING] 請求逾時，{wait:.0f}s 後重試 ({timeout_attempt} 已等 {timeout_elapsed:.0f}s/{max_timeout_seconds}s)...   ", end='', flush=True)
                    time.sleep(wait)
                elif 'Blocked' in err or 'PROHIBITED_CONTENT' in err:
                    return 'BLOCKED_BY_SAFETY'
                else:
                    reason = self._explain_error(err)
                    self.logger.error(f"API錯誤：{reason}")
                    return f'ERR:{reason}'
        reason = 'API重試耗盡'
        self.logger.error(f"{reason}（overload={overload_attempt}次 ratelimit={ratelimit_attempt}次 timeout={timeout_attempt}次 最後錯誤={err[:120]}）")
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

    @staticmethod
    def _clean_translation(t: str) -> str:
        """清理 LLM 在 translation 欄位摻入的推理垃圾：括號英文說明、字面 \\n 等。"""
        if not t:
            return t
        # 移除括號內全英文/數字的說明段落，例如 (or ...) / (Double check...) / (It's an...)
        t = re.sub(r'\s*\([A-Za-z0-9\'",.\s\-\?\!]+\)', '', t)
        # 移除字面的 \n（非真正換行，而是文字「\n」）
        t = re.sub(r'\\n', '', t)
        # 移除 "Final JSON" / "double check" / "Note:" 等英文垃圾行
        t = re.sub(r'(?i)(final json|double check|note:|translation:|original:).*', '', t)
        return t.strip()

    # ── gemma-4 markdown bullet 格式解析 ─────────────────────
    def _parse_gemma4_markdown(self, text: str,
                               blk_list: List[TextBlock],
                               visual_order: list = None) -> bool:
        """
        解析 gemma-4 輸出的 markdown bullet 格式，支援多種變體：
          格式A: *   **Cell 0:**  (縮排 bullet + bold)
                     *   Text: ...  /  *   Translation: ...
          格式B: - Index 0:
                     - Text: ...  /  - Translation: ...
          格式C: *   Cell 0: "文字" (Vertical)  (一行式，無子欄位)
                     *   Cell 0: "文字" -> "翻譯"
        """
        # 統一用寬鬆 pattern：忽略行首任意 bullet/bold 符號（*、-、空白），
        # 匹配 Cell N: / Index N: / **Cell N:** / *   **Cell 0:** 等所有變體
        cell_blocks = re.split(
            r'(?:^|\n)[*\-\s]*\*{0,2}\s*(?:[Cc]ell|[Ii]ndex)\s+(\d+)\s*\*{0,2}\s*:',
            text
        )
        # split 結果：[前置文字, index, 內容, index, 內容, ...]
        matched = 0
        i = 1
        while i + 1 < len(cell_blocks):
            try:
                visual_idx = int(cell_blocks[i])
                block_text = cell_blocks[i + 1]
            except (ValueError, IndexError):
                i += 2
                continue

            def _extract(pattern, s=block_text):
                m = re.search(pattern, s, re.IGNORECASE | re.DOTALL)
                return m.group(1).strip() if m else ''

            # 格式A/B：有 Text: 欄位
            original = _extract(r'[Tt]ext\s*:\s*"?(.+?)"?\s*(?=\n|$)')
            # 格式C：`"文字" (Vertical)` 或 `"文字" ->` 一行式，直接在 block_text 第一行
            if not original:
                original = _extract(r'^\s*"([^"]+)"')
            direction   = _extract(r'[Dd]irection\s*[:\(].*?([vh])')
            # 格式C direction 可能是 `(Vertical)` 或 `(Horizontal)`
            if not direction:
                direction = 'v' if re.search(r'[Vv]ertical', block_text) else (
                            'h' if re.search(r'[Hh]orizontal', block_text) else '')
            handwritten = _extract(r'[Hh]andwritten\s*:\s*(true|false)')
            font_raw    = _extract(r'[Ff]ont\s*size\s*:\s*~?(\d+)')
            # 格式A/B：Translation: 欄位
            translation = _extract(r'[Tt]ranslation\s*:\s*"?(.+?)"?\s*(?=\n\s*[-\*\d]|\n\s*\w+\s*:|\Z)')
            # 格式C：`"原文" -> "翻譯"` 或 `"原文" -> 翻譯`
            if not translation:
                translation = _extract(r'->\s*"?(.+?)"?\s*(?=\n|\(|\Z)')

            if not original:
                i += 2
                continue

            if visual_order is not None:
                if visual_idx < 0 or visual_idx >= len(visual_order):
                    i += 2
                    continue
                idx = visual_order[visual_idx]
            else:
                idx = visual_idx
            if idx < 0 or idx >= len(blk_list):
                i += 2
                continue

            blk = blk_list[idx]
            blk.text = [original]
            blk.translation = self._clean_translation(translation)

            if isinstance(blk.obs, dict):
                from utils.textblock import RawObservations
                blk.obs = RawObservations(**{k: v for k, v in blk.obs.items() if k in RawObservations.__dataclass_fields__})

            if direction.lower() == 'v':
                blk.obs.ocr_says_vertical = True
            elif direction.lower() == 'h':
                blk.obs.ocr_says_vertical = False
            blk.obs.ocr_src_text = original
            blk.obs.ocr_is_handwritten = handwritten.lower() == 'true'
            blk.obs.ocr_font_size_px = float(font_raw) if font_raw else -1
            matched += 1
            i += 2

        return matched > 0

    # ── 解析全頁結果（index-based，不需座標配對）────────────
    def _parse_fullpage_result(self, response_text: str,
                                blk_list: List[TextBlock],
                                visual_order: list = None,
                                is_gemma4: bool = False) -> bool:
        try:
            clean = re.sub(r'```json\s*|\s*```', '', response_text).strip()
            # gemma-4 thinking 輸出格式：<|channel>thought\n...\n<channel|>[最終答案]
            # 若有 <channel|> 標記，只取後面的部分
            if is_gemma4 and '<channel|>' in clean:
                clean = clean.split('<channel|>', 1)[-1].strip()
                self.logger.warning(f'[gemma4 debug] 偵測到 thinking 標記，已切割至 <channel|> 後，前200字: {repr(clean[:200])}')
            # 若回應不是以 [ 開頭，先嘗試抓 [...] 區塊
            if not clean.startswith('['):
                # 要求 [ 後面緊接 { 或空白+{，避免抓到 "[ and ends with ]" 這種描述文字
                m = re.search(r'(\[\s*\{.*\}\s*\])', clean, re.DOTALL)
                if m:
                    clean = m.group(1)
                else:
                    # gemma-4 fallback：從 markdown bullet 格式抽取
                    self.logger.warning(f'[gemma4 debug] 找不到 [...] 區塊，原始回應全文:\n{clean}')
                    parsed = self._parse_gemma4_markdown(clean, blk_list, visual_order)
                    return parsed

            # 移除 JSON 字串值內的原始控制字元（\x00-\x1f，排除合法的 \n \r \t）
            clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)
            # 將 JSON 字串值內的裸換行符轉成合法的 \n 逸脫（LLM 偶爾直接換行而非輸出 \\n）
            clean = re.sub(r'"((?:[^"\\]|\\.)*)"',
                           lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r') + '"',
                           clean)
            # raw_decode 只消費第一個合法 JSON 值，忽略尾端垃圾（3 Flash 已知會在陣列後多輸出 ] / "} 等片段）
            try:
                results, _ = json.JSONDecoder().raw_decode(clean.lstrip())
            except json.JSONDecodeError:
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
                trans = item.get('translation', '')
                blk.translation = self._clean_translation(trans) if is_gemma4 else trans

                # 確保 obs 是物件而不是 dict
                if isinstance(blk.obs, dict):
                    from utils.textblock import RawObservations
                    blk.obs = RawObservations(**{k: v for k, v in blk.obs.items() if k in RawObservations.__dataclass_fields__})

                llm_dir = item.get('direction', '').lower()
                if llm_dir == 'v':
                    blk.obs.ocr_says_vertical = True
                elif llm_dir == 'h':
                    blk.obs.ocr_says_vertical = False
                blk.obs.ocr_src_text = item['original']
                blk.obs.ocr_is_handwritten = bool(item.get('is_handwritten', False))
                blk.obs.ocr_font_size_px = float(item.get('font_size_px', -1))
                matched += 1

            return matched > 0
        except Exception as e:
            self.logger.warning(f"全頁解析失敗: {e} | clean全文:\n{clean}")
            if is_gemma4:
                # JSON 壞掉（[ and ends with ] 或 ... 省略號等），用原始回應跑 markdown 解析
                return self._parse_gemma4_markdown(response_text, blk_list, visual_order)
            return False

    # ── 全頁模式（單次） ──────────────────────────────────────
    _GRID_CELL = 192
    _GRID_PAD  = 28

    _GRID_PROMPT = (
        "This image is a grid of manga text box crops.\n"
        "Each cell is bordered and has a black label on its LEFT side showing its index number.\n"
        "CRITICAL: Each bordered cell is ONE independent text box. Do NOT merge text from different cells, even if they look visually related.\n"
        "Use the label numbers to identify each cell — do NOT count or guess order.\n"
        "Read the Japanese text in each cell and translate to Traditional Chinese.\n"
        "\n"
        "Translation rules:\n"
        "- Translate the original text directly. Do NOT add any parenthetical notes, explanations, or romanizations.\n"
        "- Output EXACTLY one JSON entry per cell. The total number of entries MUST equal the total cells count.\n"
        "- Output ONLY a valid JSON array, one entry per cell:\n"
        '[{"index": 0, "direction": "v or h", "is_handwritten": false, "font_size_px": 24, "original": "...", "translation": "..."}, ...]\n'
        'direction: "v"=vertical/tategumi, "h"=horizontal/yokogumi.\n'
        'is_handwritten: true if the text appears to be hand-drawn or artistic lettering (not a standard printed font).\n'
        'font_size_px: estimate the height of a single character in pixels.\n'
        'If a cell has no readable text: {"index": N, "direction": "v", "is_handwritten": false, "font_size_px": 0, "original": "", "translation": ""}\n'
        "Total cells: {n}"
    )

    # 供無原生 JSON 格式強制的模型使用（如 Grok），加強純輸出約束
    _GRID_PROMPT_STRICT = (
        "This image is a grid of manga text box crops.\n"
        "Each cell is bordered and has a black label on its LEFT side showing its index number.\n"
        "CRITICAL: Each bordered cell is ONE independent text box. Do NOT merge text from different cells, even if they look visually related.\n"
        "Use the label numbers to identify each cell — do NOT count or guess order.\n"
        "Read the Japanese text in each cell and translate to Traditional Chinese.\n"
        "\n"
        "Translation rules:\n"
        "- Translate the original text directly. Do NOT add any parenthetical notes, explanations, or romanizations.\n"
        "- Output EXACTLY one JSON entry per cell. The total number of entries MUST equal the total cells count.\n"
        "- Output ONLY a valid JSON array. Do NOT wrap it in markdown. Do NOT add any text before or after.\n"
        "- Your response MUST start with [ and end with ].\n"
        '[{"index": 0, "direction": "v or h", "is_handwritten": false, "font_size_px": 24, "original": "...", "translation": "..."}, ...]\n'
        'direction: "v"=vertical/tategumi, "h"=horizontal/yokogumi.\n'
        'is_handwritten: true if the text appears to be hand-drawn or artistic lettering (not a standard printed font).\n'
        'font_size_px: estimate the height of a single character in pixels.\n'
        'If a cell has no readable text: {"index": N, "direction": "v", "is_handwritten": false, "font_size_px": 0, "original": "", "translation": ""}\n'
        "Total cells: {n}"
    )

    # Gemma-4 專屬提示詞：強制禁止思考文字混入 JSON 欄位
    _GRID_PROMPT_GEMMA4 = (
        "This image is a grid of manga text box crops.\n"
        "Each cell is bordered and has a black label on its LEFT side showing its index number.\n"
        "CRITICAL: Each bordered cell is ONE independent text box. Do NOT merge text from different cells.\n"
        "Use the label numbers to identify each cell — do NOT count or guess order.\n"
        "Read the Japanese text in each cell and translate to Traditional Chinese.\n"
        "\n"
        "STRICT OUTPUT RULES — you MUST follow these exactly:\n"
        "1. Output ONLY a valid JSON array. Nothing else.\n"
        "2. The 'original' field: ONLY the Japanese text from that cell. Nothing else.\n"
        "3. The 'translation' field: ONLY the Traditional Chinese translation. "
        "NO English. NO explanations. NO notes. NO 'Refining translations'. "
        "NO alternative translations. NO parenthetical remarks. ONLY Chinese characters.\n"
        "4. Do NOT put any thinking, reasoning, or analysis into any JSON field.\n"
        "5. If you are unsure of a translation, output your best guess directly in Chinese — do NOT write your reasoning.\n"
        "\n"
        "Output format (one entry per cell):\n"
        '[{"index": 0, "direction": "v", "is_handwritten": false, "font_size_px": 24, "original": "日文原文", "translation": "繁體中文翻譯"}, ...]\n'
        'direction: "v"=vertical, "h"=horizontal.\n'
        'is_handwritten: true only if hand-drawn lettering.\n'
        'font_size_px: estimate character height in pixels.\n'
        'No readable text in a cell: {"index": N, "direction": "v", "is_handwritten": false, "font_size_px": 0, "original": "", "translation": ""}\n'
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

        if self.save_grid_debug:
            debug_dir = os.path.join(img_dir, 'ocr_debug')
            os.makedirs(debug_dir, exist_ok=True)
            file_stem = os.path.splitext(img_name)[0]
            debug_path = os.path.join(debug_dir, f"{file_stem}_grid.jpg")
            cv2.imwrite(debug_path, canvas)

        return canvas, visual_order

    def _run_fullpage(self, img: np.ndarray, blk_list: List[TextBlock]):
        if self.disable_plan_a:
            return 'SKIP:disabled', 0
        return self._run_fullpage_impl(img, blk_list, client=None)

    def _run_fullpage_impl(self, img: np.ndarray, blk_list: List[TextBlock], client=None):
        """全頁模式核心邏輯，可傳入 client 供 Plan C 使用 fallback model。"""
        _client = client if client is not None else self.client
        is_gemma4 = isinstance(_client, GeminiClient) and _client.model in GeminiClient._GEMMA4_MODELS
        if is_gemma4:
            base_prompt = self._GRID_PROMPT_GEMMA4
        elif not isinstance(_client, GeminiClient):
            base_prompt = self._GRID_PROMPT_STRICT
        else:
            base_prompt = self._GRID_PROMPT
        grid_img, visual_order = self._build_grid_img(img, blk_list)
        prompt = base_prompt.replace('{n}', str(len(blk_list)))
        for attempt in range(2):
            resp = self._call_ocr(grid_img, custom_prompt=prompt, allow_array=True,
                                  max_overload_seconds=self.plan_a_retry_seconds,
                                  max_timeout_seconds=self.plan_a_retry_seconds,
                                  client=client)
            if resp == 'BLOCKED_BY_SAFETY':
                return '安全過濾器擋住', 0
            if resp.startswith('ERR:'):
                return resp[4:], 0
            if not resp:
                return 'API無回應', 0
            matched = self._parse_fullpage_result(resp, blk_list, visual_order, is_gemma4=is_gemma4)
            if matched:
                return 'ok', matched
            if attempt == 0:
                self.logger.warning(f'Plan A JSON解析失敗，重試一次... | 原始回應全文:\n{resp}')
            else:
                self.logger.warning(f'Plan A 第二次解析失敗 | 原始回應全文:\n{resp}')
        return 'JSON解析失敗', 0

    # ── 切片模式 ──────────────────────────────────────────────
    _SLICE_PROMPT = (
        "Extract the Japanese text from this image and translate to Traditional Chinese. "
        "Line breaks: use the original text's line breaks as a loose reference — keep a line break in the translation only if it fits the natural meaning or rhythm. Do NOT add extra line breaks. "
        "Output ONLY valid JSON: {\"original\": \"...\", \"translation\": \"...\", \"direction\": \"v or h\", \"is_handwritten\": false, \"font_size_px\": 24}. "
        "direction: \"v\"=vertical/tategumi, \"h\"=horizontal/yokogumi. "
        "is_handwritten: true if the text appears to be hand-drawn or artistic lettering (not a standard printed font). "
        "font_size_px: estimate the height of a single character in pixels. "
        "If the image contains NO TEXT, output an empty JSON {}."
    )

    def _process_single_blk(self, idx: int, blk: TextBlock,
                             cropped: np.ndarray, log_prefix: str,
                             max_overload_seconds: int = 120,
                             max_timeout_seconds: int = 120,
                             client=None, is_fallback: bool = False,
                             imgname: str = ''):
        used_grok = False
        resp = self._call_ocr(cropped, custom_prompt=self._SLICE_PROMPT,
                              max_overload_seconds=max_overload_seconds,
                              max_timeout_seconds=max_timeout_seconds,
                              client=client)

        if resp == 'BLOCKED_BY_SAFETY':
            if is_fallback:
                # Plan C 本身已是 fallback，不遞迴呼叫
                self._emit(OcrEventType.ERROR, imgname)
                return idx, None
            resp = self._call_ocr_grok(cropped, self._SLICE_PROMPT, log_prefix,
                                       silent=True)
            if not resp or resp.isdigit():
                self.logger.error(f"{log_prefix} 切片 {idx+1} 失敗: {resp or '?'}")
                self._emit(OcrEventType.ERROR, imgname)
                return idx, None
            used_grok = True

        if not resp:
            self._emit(OcrEventType.ERROR, imgname)
            return idx, None

        try:
            clean = re.sub(r'```json\s*|\s*```', '', resp).strip()
            if not clean or clean in ('{}', '[]'):
                self._emit(OcrEventType.ERROR, imgname)
                return idx, None
            # gemma-4 可能把思考文字混在前面，嘗試找第一個 {...} 區塊
            if not clean.startswith('{'):
                m = re.search(r'(\{.*?\})', clean, re.DOTALL)
                if m:
                    clean = m.group(1)
                else:
                    # 從思考文字直接抓 original / translation
                    orig_m  = re.search(r'[Oo]riginal["\s:]+([^\n"]+)', clean)
                    trans_m = re.search(r'[Tt]ranslation["\s:]+([^\n"]+)', clean)
                    dir_m   = re.search(r'[Dd]irection["\s:]+([vh])', clean)
                    if orig_m:
                        original = orig_m.group(1).strip().strip('"')
                        translation = trans_m.group(1).strip().strip('"') if trans_m else ''
                        blk.text = [original]
                        blk.translation = translation
                        blk.obs.ocr_src_text = original
                        if dir_m:
                            blk.obs.ocr_says_vertical = dir_m.group(1).lower() == 'v'
                        self._emit(OcrEventType.GROK_OK if used_grok else OcrEventType.SLICE_OK, imgname)
                        return idx, blk
                    self.logger.warning(f"{log_prefix} 切片 {idx+1} 解析失敗: 找不到 JSON 區塊")
                    self._emit(OcrEventType.ERROR, imgname)
                    return idx, None
            data = json.loads(clean)
            if isinstance(data, list):
                data = data[0] if data else {}
            if isinstance(data, dict) and data.get('original'):
                blk.text = [data['original']]
                blk.translation = data.get('translation', '')
                blk.obs.ocr_src_text = data['original']
                llm_dir = data.get('direction', '').lower()
                if llm_dir == 'v':
                    blk.obs.ocr_says_vertical = True
                elif llm_dir == 'h':
                    blk.obs.ocr_says_vertical = False
                blk.obs.ocr_is_handwritten = bool(data.get('is_handwritten', False))
                blk.obs.ocr_font_size_px = float(data.get('font_size_px', -1))
                self._emit(OcrEventType.GROK_OK if used_grok else OcrEventType.SLICE_OK, imgname)
                return idx, blk
        except Exception as e:
            self.logger.warning(f"{log_prefix} 切片 {idx+1} 解析失敗: {e}")

        self._emit(OcrEventType.ERROR, imgname)
        return idx, None

    def _run_slice_plan(self, img: np.ndarray, blk_list: List[TextBlock],
                         log_prefix: str, probe_retry_seconds: int = 300,
                         client=None, ignore_disable: bool = False,
                         imgname: str = ''):
        if self.disable_plan_b and not ignore_disable:
            return 'SKIP:disabled', 0, len(blk_list), list(range(len(blk_list)))
        snap_imgname = imgname or self.current_imgname
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

        if not tasks:
            return 'ok', 0, 0, []

        # ── 試探框：先送第一個框，確認伺服器可用 ──────────────
        probe_idx, probe_blk, probe_crop = tasks[0]
        self.logger.info(f"{log_prefix} Plan B 試探框 {probe_idx+1}（上限 {probe_retry_seconds}s）...")
        p_idx, p_blk = self._process_single_blk(
            probe_idx, probe_blk, probe_crop, log_prefix,
            max_overload_seconds=probe_retry_seconds,
            max_timeout_seconds=probe_retry_seconds,
            client=client, is_fallback=(client is not None),
            imgname=snap_imgname,
        )
        if p_blk is None:
            self.logger.warning(f"{log_prefix} Plan B 試探框失敗（超過 {probe_retry_seconds}s），跳 Plan C")
            return 'PROBE_FAIL', 0, len(blk_list), list(range(len(blk_list)))

        results_map = {p_idx: p_blk}

        # ── 其餘框並行送出 ────────────────────────────────────
        remaining_tasks = tasks[1:]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {}
            for i, (orig_idx, blk, crop) in enumerate(remaining_tasks):
                futs[ex.submit(self._process_single_blk, orig_idx, blk, crop, log_prefix,
                               120, 120, client, client is not None, snap_imgname)] = orig_idx
                # 在分配任務時加入固定延遲，完美錯開併發流量
                if i < len(remaining_tasks) - 1:
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
        for block in blocks:
            block.sort(key=lambda col: -sum(cx(b) for b in col) / len(col))
            for col in block:
                result.extend(col)

        return result

    # ── 主流程 ────────────────────────────────────────────────
    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock],
                       *args, force_slice=False, imgname: str = '', **kwargs):
        if self.client is None or not blk_list:
            return

        self.page_counter += 1
        # 優先用呼叫方傳入的 imgname，避免多 worker 共用 self.current_imgname 被覆蓋
        snap_imgname = imgname or self.current_imgname
        lp = f"[{self._fmt_imgname(snap_imgname)}]"

        sorted_blks = self._sort_blk_reading_order(blk_list)

        # Plan A：原圖全頁（force_slice=True 時跳過）
        if not force_slice:
            result, matched = self._run_fullpage(img, sorted_blks)
            if result == 'ok':
                self._emit(OcrEventType.PLAN_A_OK, snap_imgname)
                self.logger.success(f"{lp} Plan A 成功（{matched}/{len(blk_list)} 框）")
                for blk in blk_list:
                    resolve_blk_style(blk)
                return
            self.logger.warning(f"{lp} Plan A：{result}")

        # Plan B：切片（含切片層級的 Grok 備援）
        result, ok, fail, failed_indices = self._run_slice_plan(
            img, blk_list, lp,
            probe_retry_seconds=self.plan_b_retry_seconds,
            imgname=snap_imgname,
        )
        if result == 'PROBE_FAIL':
            self.logger.warning(f"{lp} Plan B 試探框失敗（上限 {self.plan_b_retry_seconds}s），跳 Plan C")
            failed_indices = list(range(len(blk_list)))
        elif result == 'ok':
            total = ok + fail
            if fail == 0:
                self.logger.success(f"{lp} Plan B 成功（{ok}/{total} 框）")
                for blk in blk_list:
                    resolve_blk_style(blk)
                return
            else:
                self.logger.warning(f"{lp} Plan B 完成（{ok}/{total} 框，{fail} 框失敗）")
        else:
            self.logger.warning(f"{lp} Plan B：{result}")

        # Plan C：換 fallback model，跑相同路線
        fallback_client = self._build_fallback_client()
        if not fallback_client:
            for i in failed_indices:
                blk_list[i].text = ['●●●']
                blk_list[i].translation = ''
            for blk in blk_list:
                resolve_blk_style(blk)
            self.logger.error(f"{lp} 無備援API，此頁放棄")
            self._emit(OcrEventType.ERROR, snap_imgname)
            return

        all_failed = (result == 'PROBE_FAIL') or (result != 'ok')

        if all_failed:
            # Plan C 全頁模式（和 Plan A 相同邏輯，換 fallback client）
            self.logger.info(f"{lp} Plan C 全頁模式（fallback model）...")
            result_c, matched_c = self._run_fullpage_impl(img, sorted_blks, client=fallback_client)
            if result_c == 'ok':
                self._emit(OcrEventType.GROK_OK, snap_imgname)
                self.logger.success(f"{lp} Plan C 全頁成功（{matched_c}/{len(blk_list)} 框）")
                for blk in blk_list:
                    resolve_blk_style(blk)
                return
            self.logger.warning(f"{lp} Plan C 全頁失敗：{result_c}，降級切片...")
            failed_indices = list(range(len(blk_list)))

        # Plan C 切片模式（部分失敗 或 全頁降級後）
        # 只傳失敗框的子集進去
        failed_blks = [blk_list[i] for i in failed_indices]
        result_c, ok_c, fail_c, local_failed = self._run_slice_plan(
            img, failed_blks, lp,
            probe_retry_seconds=self.plan_c_retry_seconds,
            client=fallback_client,
            ignore_disable=True,
            imgname=snap_imgname,
        )
        # local_failed 是子集內的 index，映射回原始 blk_list index
        still_failed = [failed_indices[li] for li in local_failed]
        for i in still_failed:
            blk_list[i].text = ['●●●']
            blk_list[i].translation = ''
        for blk in blk_list:
            resolve_blk_style(blk)
        if fail_c == 0:
            self.logger.success(f"{lp} Plan C 切片成功（{ok_c}/{len(failed_indices)} 框）")
        else:
            self.logger.warning(f"{lp} Plan C 切片完成（{ok_c}/{len(failed_indices)} 框，{fail_c} 框失敗）")

    def ocr_img(self, img: np.ndarray) -> str:
        return self._call_ocr(img)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ('provider', 'api_key', 'model', 'base_url'):
            self._build_client()
            self.page_counter = 0