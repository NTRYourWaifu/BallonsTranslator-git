import base64
import requests
import numpy as np
import cv2
from typing import Union, List, Tuple
from collections import OrderedDict

from utils.textblock import TextBlock

from utils.registry import Registry
TEXTDETECTORS = Registry('textdetectors')
register_textdetectors = TEXTDETECTORS.register_module

from ..base import BaseModule, DEFAULT_DEVICE, DEVICE_SELECTOR

class TextDetectorBase(BaseModule):

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in TEXTDETECTORS.module_dict:
            if TEXTDETECTORS.module_dict[key] == self.__class__:
                self.name = key
                break

    def _detect(self, *args, **kwargs) -> Tuple[np.ndarray, List[TextBlock]]:
        raise NotImplementedError

    def setup_detector(self):
        raise NotImplementedError

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        if not self.all_model_loaded():
            self.load_model()
        mask, blk_list = self._detect(img)
        for blk in blk_list:
            blk.det_model = self.name
        try:
            measure_src_glyph(mask, blk_list)
        except Exception as e:
            self.logger.warning(f'量原文字級失敗：{e}')
        return mask, blk_list


def measure_src_glyph(mask: np.ndarray, blk_list: List[TextBlock]) -> None:
    """量每個框裡原文的字有多大，寫進 blk.obs.src_glyph_px／src_glyph_n（給排版「字級跟原文」用）。

    照 MangaTranslator v1.25.0 的 estimate_glyph_size（commit 5439b80）：框內的文字遮罩切成連通塊，
    丟掉小於 6 像素的雜點，每塊取高寬較大的那邊（中日文是方塊字；細長的塊取短邊），取第 90 百分位——
    注音小字和斷開的筆畫都比一個完整的字小，所以取偏大的那端。
    它用泡泡內的墨水，這裡改用偵測器的文字遮罩，沒泡泡的字、彩色字也量得到。
    """
    if mask is None:
        return
    im_h, im_w = mask.shape[:2]
    for blk in blk_list:
        x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(im_w, x2), min(im_h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        roi = (mask[y1:y2, x1:x2] > 127).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        sizes = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 6:
                continue
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # 描邊字整欄黏成一塊時又細又長：取短邊（＝欄寬≈字大小），不然會量成整欄高
            sizes.append(min(w, h) if max(w, h) > 1.6 * min(w, h) else max(w, h))
        if sizes:
            blk.obs.src_glyph_px = float(np.percentile(sizes, 90))
            blk.obs.src_glyph_n = len(sizes)
