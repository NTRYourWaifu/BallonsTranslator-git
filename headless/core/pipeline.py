"""
無頭翻譯 pipeline 核心：吃一個圖片資料夾 → 逐頁翻譯 → 輸出翻譯後圖 + translate.json。

一頁的直線流程（對齊 GUI _imgtrans_pipeline 但單執行緒、零中間檔）：
    img = imread(page)
    mask, blk_list = detector.detect(img)          # 無框 → 原圖直接當輸出，跳過
    ocr.run_ocr(img, blk_list)                     # ocr_llm：Gemini 掃圖，一步填 blk.text + blk.translation
    inpainted = inpainter.inpaint(img, mask, blk)  # 記憶體 numpy，不落地
    apply_style_and_fontcalc(blk_list, ...)        # 顏色/stroke/字體（複用 GUI 配方）
    result = render_page(inpainted, blk_list)      # offscreen Qt 渲染
    imwrite(result)

零中間檔：mask/inpainted/ocr_debug 都不寫；只多一份 translate.json（每頁框+原文+譯文+字體+字數）。
"""
import os
import time
import json
import threading
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from utils.io_utils import imread, imwrite, find_all_imgs
from .renderer import render_page, apply_style_and_fontcalc


@dataclass
class PageResult:
    name: str
    status: str            # 'translated' | 'skipped_no_text' | 'failed'
    n_blocks: int = 0
    char_src: int = 0      # 原文字數（去空白）
    char_trans: int = 0    # 譯文字數（去空白）＝字數供應口徑
    error: str = ''
    seconds: float = 0.0


@dataclass
class BookResult:
    src_dir: str
    out_dir: str
    pages: List[PageResult] = field(default_factory=list)
    total_seconds: float = 0.0

    @property
    def n_translated(self):
        return sum(1 for p in self.pages if p.status == 'translated')

    @property
    def n_failed(self):
        return sum(1 for p in self.pages if p.status == 'failed')


def _count_chars(s: str) -> int:
    if not s:
        return 0
    return len(''.join(s.split()))


def translate_book(src_dir: str, mods, cfg, *,
                   out_dir: Optional[str] = None,
                   st_manager=None,
                   save_ext: str = None,
                   save_quality: int = 92,
                   progress_cb=None,
                   stage_progress_cb=None,
                   workers: int = 5,
                   max_pages: int = None) -> BookResult:
    """
    src_dir：圖片資料夾。out_dir：None=原地覆蓋（寫回 src_dir）。
    mods：HeadlessModules（bootstrap.load_modules 產）。cfg：pcfg。
    save_ext：輸出副檔名（None=沿用原副檔名；.avif 走 ffmpeg 見 P1）。
    workers：階段一並行 worker 數（只並行偵測+Gemini；塗白渲染仍單執行緒）。
    stage_progress_cb：各模組進度回報 fn({'detect':[done,total],'translate':[...],'render':[...]})。
    max_pages：只翻前 N 頁（測試用，省 API）。

    兩階段架構（不走舊 lifecycle 老路）：
      階段一 ThreadPoolExecutor(workers) 並行「偵測(序列)+Gemini(並行等網路)」，只填 blk。
      階段二 單執行緒按頁序 塗白→排版→渲染→輸出（無共享狀態、無 signal/counter/FSM）。
    """
    if out_dir is None:
        out_dir = src_dir
    os.makedirs(out_dir, exist_ok=True)

    imgs = find_all_imgs(src_dir, abs_path=False)
    imgs = sorted(imgs)
    if max_pages is not None:
        imgs = imgs[:max_pages]

    book = BookResult(src_dir=src_dir, out_dir=out_dir)
    translate_json = {
        'version': 1,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'modules': {
            'detector': type(mods.detector).__name__,
            'ocr': type(mods.ocr).__name__,
            'ocr_model': mods.ocr.params['model']['value'],
            'inpainter': type(mods.inpainter).__name__,
        },
        'pages': {},
    }

    total = len(imgs)
    t_book = time.time()

    # ── 進度回報（各模組各自 done/total，對齊氣球翻譯器進度窗形態）──
    prog = {'detect': [0, total], 'translate': [0, total], 'render': [0, total]}

    def _report():
        if stage_progress_cb:
            stage_progress_cb({k: list(v) for k, v in prog.items()})

    # 每頁的中間資料（階段一填、階段二用）
    class _PageCtx:
        __slots__ = ('name', 'img', 'mask', 'blk_list', 'status', 'error', 'char_src', 'char_trans')
        def __init__(self, name):
            self.name = name; self.img = None; self.mask = None; self.blk_list = None
            self.status = 'failed'; self.error = ''; self.char_src = 0; self.char_trans = 0

    ctxs = {name: _PageCtx(name) for name in imgs}

    # ══ 階段一：並行「偵測 + Gemini 翻譯」（只填 blk，不碰塗白渲染）══
    # 偵測(YOLO)本機序列跑（很快、避免多執行緒搶 GPU）；Gemini(run_ocr)是網路等待，交 worker 並行。
    _detect_lock = threading.Lock()

    def _stage1_page(name):
        ctx = ctxs[name]
        try:
            src_path = os.path.join(src_dir, name)
            img = imread(src_path)
            if img is None:
                raise RuntimeError('imread 回 None（解碼失敗）')
            ctx.img = img
            # 偵測：上鎖序列跑（YOLO 快，避免多執行緒同時打 GPU）
            with _detect_lock:
                if hasattr(mods.detector, 'current_imgname'):
                    mods.detector.current_imgname = name
                mask, blk_list = mods.detector.detect(img)
            ctx.mask = mask
            ctx.blk_list = blk_list
            with _lock:
                prog['detect'][0] += 1
                _report()
            if not blk_list:
                ctx.status = 'skipped_no_text'
                with _lock:
                    prog['translate'][0] += 1
                    _report()
                return
            # Gemini 翻譯（網路等待，這是並行的意義）—— run_ocr 執行緒安全（client 唯讀、thread-local 已備）
            mods.ocr.run_ocr(img, blk_list, imgname=name)
            ctx.char_src = sum(_count_chars(b.get_text() or '') for b in blk_list)
            ctx.char_trans = sum(_count_chars(b.translation or '') for b in blk_list)
            ctx.status = 'translated'
        except Exception as e:
            ctx.error = f'{type(e).__name__}: {e}'
            ctx.status = 'failed'
            traceback.print_exc()
        finally:
            with _lock:
                if ctx.status != 'skipped_no_text':
                    prog['translate'][0] += 1
                    _report()

    _lock = threading.Lock()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(_stage1_page, imgs))

    # ══ 階段二：單執行緒按頁序 塗白 → 排版 → 渲染 → 輸出 ══
    # 絕不並行（舊 lifecycle 出雷處）。無共享狀態、無 signal/counter/FSM。
    for i, name in enumerate(imgs):
        ctx = ctxs[name]
        t0 = time.time()
        pr = PageResult(name=name, status=ctx.status,
                        n_blocks=len(ctx.blk_list) if ctx.blk_list else 0,
                        char_src=ctx.char_src, char_trans=ctx.char_trans, error=ctx.error)
        try:
            if ctx.status == 'skipped_no_text':
                _write_out(ctx.img, out_dir, name, save_ext, save_quality)
                translate_json['pages'][name] = {'char_src': 0, 'char_trans': 0, 'blocks': []}
            elif ctx.status == 'translated':
                img = ctx.img
                img_h, img_w = img.shape[:2]
                # 塗白（記憶體 numpy，不落地）
                if ctx.mask is not None and ctx.mask.sum() > 0:
                    inpainted = mods.inpainter.inpaint(img, ctx.mask.copy(), ctx.blk_list)
                else:
                    inpainted = img
                # 顏色/stroke/字體（複用 GUI 配方）
                apply_style_and_fontcalc(ctx.blk_list, img_h, img_w, cfg, mods.ocr)
                # 渲染 + 輸出
                result = render_page(inpainted, ctx.blk_list, cfg,
                                     st_manager=st_manager, proj_img_array=img)
                _write_out(result, out_dir, name, save_ext, save_quality)
                translate_json['pages'][name] = {
                    'char_src': ctx.char_src, 'char_trans': ctx.char_trans,
                    'blocks': [b.to_dict(deep_copy=True) for b in ctx.blk_list],
                }
            else:  # failed：原圖搬去輸出（不留黑洞）
                if ctx.img is not None:
                    _write_out(ctx.img, out_dir, name, save_ext, save_quality)
        except Exception as e:
            pr.error = (pr.error + ' | ' if pr.error else '') + f'render: {type(e).__name__}: {e}'
            pr.status = 'failed'
            traceback.print_exc()
            try:
                if ctx.img is not None:
                    _write_out(ctx.img, out_dir, name, save_ext, save_quality)
            except Exception:
                pass
        prog['render'][0] += 1
        _report()
        pr.seconds = time.time() - t0
        book.pages.append(pr)
        if progress_cb:
            progress_cb(i + 1, total, pr)
        # 釋放大物件
        ctx.img = None; ctx.mask = None

    book.total_seconds = time.time() - t_book

    # 寫 translate.json（存 out_dir）
    with open(os.path.join(out_dir, 'translate.json'), 'w', encoding='utf-8') as f:
        from ui.config_proj import TextBlkEncoder
        json.dump(translate_json, f, ensure_ascii=False, cls=TextBlkEncoder, indent=1)

    return book


def rerender_book(src_dir: str, translate_json_path: str, mods, cfg, *,
                  out_dir: Optional[str] = None,
                  st_manager=None, save_ext: str = None, save_quality: int = 92,
                  stage_progress_cb=None, progress_cb=None) -> BookResult:
    """
    重新排版：讀 translate.json（含每頁框+原文+譯文+字體），對原文圖重跑 inpaint+渲染。
    **不跑 Gemini API**（譯文已在 json 裡）——用於字型修好/調排版後免費重繪。

    src_dir：原文圖資料夾（inpaint 塗白要用原圖，不是譯文圖）。
    translate_json_path：譯文版的 translate.json。
    mods 只需要 inpainter（detector/ocr 不呼叫）。
    """
    if out_dir is None:
        out_dir = src_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(translate_json_path, encoding='utf-8') as f:
        tj = json.load(f)
    pages = tj.get('pages', {})

    from utils.textblock import TextBlock
    book = BookResult(src_dir=src_dir, out_dir=out_dir)
    total = len(pages)
    prog = {'detect': [total, total], 'translate': [total, total], 'render': [0, total]}

    def _report():
        if stage_progress_cb:
            stage_progress_cb({k: list(v) for k, v in prog.items()})

    t_book = time.time()
    for i, (name, pg) in enumerate(pages.items()):
        t0 = time.time()
        src_path = os.path.join(src_dir, name)
        pr = PageResult(name=name, status='failed')
        try:
            img = imread(src_path)
            if img is None:
                raise RuntimeError('原文圖 imread 回 None')
            img_h, img_w = img.shape[:2]
            blocks = pg.get('blocks', [])
            pr.char_src = pg.get('char_src', 0)
            pr.char_trans = pg.get('char_trans', 0)

            if not blocks:
                # 無字頁：原圖直接輸出
                _write_out(img, out_dir, name, save_ext, save_quality)
                pr.status = 'skipped_no_text'
            else:
                # 從 dict 重建 blk_list（TextBlock(**dict) 反向重建，config_proj 同法）
                blk_list = []
                for bd in blocks:
                    try:
                        blk_list.append(TextBlock(**bd))
                    except Exception:
                        # 欄位不合就過濾成 dataclass 認得的
                        valid = {k: v for k, v in bd.items() if k in TextBlock.__dataclass_fields__}
                        blk_list.append(TextBlock(**valid))
                pr.n_blocks = len(blk_list)
                # 重建 mask（從框重算）→ inpaint 塗白
                mask = _mask_from_blocks(img, blk_list)
                inpainted = mods.inpainter.inpaint(img, mask, blk_list) if mask is not None and mask.sum() > 0 else img
                # 字體/顏色/stroke（複用 GUI 配方）+ 渲染
                apply_style_and_fontcalc(blk_list, img_h, img_w, cfg, mods.ocr)
                result = render_page(inpainted, blk_list, cfg, st_manager=st_manager, proj_img_array=img)
                _write_out(result, out_dir, name, save_ext, save_quality)
                pr.status = 'translated'
        except Exception as e:
            pr.error = f'{type(e).__name__}: {e}'
            pr.status = 'failed'
            traceback.print_exc()
            try:
                _img = imread(src_path)
                if _img is not None:
                    _write_out(_img, out_dir, name, save_ext, save_quality)
            except Exception:
                pass
        prog['render'][0] += 1
        _report()
        pr.seconds = time.time() - t0
        book.pages.append(pr)
        if progress_cb:
            progress_cb(i + 1, total, pr)

    book.total_seconds = time.time() - t_book
    # 重寫 translate.json（內容不變，維持一致）
    import shutil
    if os.path.abspath(translate_json_path) != os.path.abspath(os.path.join(out_dir, 'translate.json')):
        shutil.copy2(translate_json_path, os.path.join(out_dir, 'translate.json'))
    return book


def _mask_from_blocks(img, blk_list):
    """從 blk 的框重建 inpaint mask（框內填 255）。重排版無原始 mask，從框幾何重算。"""
    import numpy as np
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for blk in blk_list:
        try:
            x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
        except Exception:
            continue
    return mask


def _write_out(img_bgr: np.ndarray, out_dir: str, name: str, save_ext: str, quality: int = 92):
    """寫成品圖。save_ext 指定副檔名（None=沿用原副檔名）。AVIF 走 ffmpeg SVT-AV1、WebP/JPG 走 cv2。"""
    from .imgio import save_image
    base, ext = os.path.splitext(name)
    ext = (save_ext or ext).lower()
    out_path = os.path.join(out_dir, base + ext)
    save_image(img_bgr, out_path, quality=quality)
