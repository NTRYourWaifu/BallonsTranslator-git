"""
無頭渲染器：把翻譯完的 blk_list（已填 translation + fontformat）畫回 inpainted 底圖，輸出成品 numpy。

策略（見 docs 計畫 D5 + bt_rewrite_module_map）：
  - 留 Qt、offscreen 跑，複用 GUI 久經考驗的字體/顏色/stroke 計算（_apply_color_stroke_and_fontcalc 的邏輯）
    與 TextBlkItem 排版，零視覺回歸。
  - 不搬 canvas.py（綁 GUI scene layers/scale/scrollbar）；新建最小 QGraphicsScene：底圖 pixmap + 每 blk 一個 TextBlkItem。
"""
import numpy as np
from qtpy.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from qtpy.QtGui import QImage, QPainter, QColor, QPixmap
from qtpy.QtCore import QRectF, Qt


def _ndarray_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """BGR numpy → QPixmap（底圖用）。"""
    img = np.ascontiguousarray(img_bgr[:, :, ::-1])  # BGR→RGB
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _qimage_to_ndarray(qimg: QImage) -> np.ndarray:
    """QImage(ARGB32) → BGR numpy。"""
    qimg = qimg.convertToFormat(QImage.Format.Format_RGB32)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    ptr.setsize(qimg.sizeInBytes() if hasattr(qimg, 'sizeInBytes') else qimg.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape(h, qimg.bytesPerLine() // 4, 4)[:, :w, :]
    # Format_RGB32 在 little-endian 記憶體序是 BGRA → 取 BGR
    return np.ascontiguousarray(arr[:, :, :3])


def apply_style_and_fontcalc(blk_list, img_h, img_w, cfg, ocr_module):
    """
    複刻 mainwindow._apply_color_stroke_and_fontcalc 的純計算部分（顏色還原→stroke→E5→字體 render）。
    這段是 D3/E3/E5 歷史 latent bug 的兜底配方，原樣搬、不自創。
    cfg = pcfg（全域 config），讀 let_*_flag 與 module.enable_*。
    """
    from ui.textitem import calc_font_size_by_render
    gf = _global_format()

    override_fnt_color  = cfg.let_fntcolor_flag == 1
    override_fnt_stroke = cfg.let_fntstroke_flag == 1
    override_fnt_scolor = cfg.let_fnt_scolor_flag == 1

    for blk in blk_list:
        # 顏色還原（必須在 stroke 之前）
        if override_fnt_color:
            blk.set_font_colors(fg_colors=gf.frgb)
        elif blk.fontformat.frgb == [0, 0, 0]:
            blk.set_font_colors(fg_colors=gf.frgb)
        if override_fnt_scolor:
            blk.set_font_colors(bg_colors=gf.srgb)
        elif blk.fontformat.srgb == [0, 0, 0]:
            blk.set_font_colors(bg_colors=gf.srgb)
        # stroke（顏色正確後無條件算）
        if override_fnt_stroke:
            blk.stroke_width = gf.stroke_width
        elif cfg.module.enable_ocr:
            blk.recalulate_stroke_width()
        # E5 補正
        sw = blk.stroke_width
        if sw > 0 and cfg.module.enable_ocr and cfg.module.enable_detect and cfg.let_fntsize_flag != 1:
            blk.font_size = blk.font_size / (1 + sw)

    # 字體 render 最終值
    _scale = getattr(ocr_module, 'font_size_scale', 1.0) if ocr_module is not None else 1.0
    _char_scale_table = getattr(ocr_module, 'char_scale_table', None) if ocr_module is not None else None
    if cfg.let_fntsize_flag != 1:
        for blk in blk_list:
            if blk.translation and blk.translation.strip() not in ('', '●●●'):
                blk.font_size = calc_font_size_by_render(
                    blk, scale=_scale, img_h=float(img_h), char_scale_table=_char_scale_table)


def _global_format():
    """取 global_fontformat（GUI 是 formatpanel.global_format，headless 從 config 直接建）。"""
    import utils.config as _cfgmod
    from utils.fontformat import FontFormat
    gf_dict = getattr(_cfgmod.pcfg, 'global_fontformat', None)
    if isinstance(gf_dict, FontFormat):
        return gf_dict
    # pcfg.global_fontformat 可能是 dict，轉成 FontFormat
    if isinstance(gf_dict, dict):
        try:
            return FontFormat(**{k: v for k, v in gf_dict.items() if k in FontFormat.__dataclass_fields__})
        except Exception:
            pass
    return FontFormat()


def render_page(inpainted_bgr: np.ndarray, blk_list, cfg,
                st_manager=None, proj_img_array: np.ndarray = None) -> np.ndarray:
    """
    inpainted 底圖 + 翻譯完的 blk_list → 成品 BGR numpy。
    st_manager：若提供（HeadlessSceneTextManager 替身），用 GUI 原碼 layout_textblk 做自動排版（零複刻風險）；
                None 則只走 TextBlkItem 建構的基本排版（blk 帶 xyxy+fontformat，建構即擺位）。
    """
    from ui.textitem import TextBlkItem

    H, W = inpainted_bgr.shape[:2]
    scene = QGraphicsScene(0, 0, W, H)
    base_item = QGraphicsPixmapItem(_ndarray_to_qpixmap(inpainted_bgr))
    scene.addItem(base_item)

    auto_layout = st_manager is not None and getattr(st_manager, 'auto_textlayout_flag', False)

    for idx, blk in enumerate(blk_list):
        trans = (blk.translation or '').strip()
        if not trans or trans in ('●●●',):
            continue
        # 建 TextBlkItem：建構會 initTextBlock 從 blk 套 fontformat + 位置（基本排版）
        item = TextBlkItem(blk, idx, set_format=True, show_rect=False)
        item.setPlainText(trans)
        # 自動排版（用 GUI 原碼，對齊 GUI addTextBlock 分支）
        if auto_layout and not blk.vertical:
            try:
                st_manager.layout_textblk(item, text=trans)
            except Exception:
                item.setPlainText(trans)
        scene.addItem(item)

    out = QImage(W, H, QImage.Format.Format_ARGB32)
    out.fill(QColor(0, 0, 0, 0))
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    scene.render(p, QRectF(0, 0, W, H), QRectF(0, 0, W, H))
    p.end()

    result = _qimage_to_ndarray(out)
    scene.clear()
    return result
