import os
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

from .base import register_textdetectors, TextDetectorBase, TextBlock
from utils.imgproc_utils import xywh2xyxypoly
from utils.textblock import examine_textblk

@register_textdetectors('yolov8')
class DetectorYOLOv8(TextDetectorBase):
    params = {
        'model_path': {
            'value': 'data/models/manga-text-detector.pt',
            'description': 'YOLOv8 模型檔 (.pt) 的路徑'
        },
        'conf_threshold': {
            'value': '0.25',
            'description': '信心度閾值 (0.0~1.0)'
        },
        'x_gap_ratio': {
            'value': '1.5',
            'description': (
                '欄間距容許倍數。'
                '兩框空白距離 ≤ 此值 × min(兩框原始寬度) 才可合併。'
                '建議範圍 0.5 ~ 3.0，預設 1.5。'
            )
        },
        'y_overlap_ratio': {
            'value': '0.3',
            'description': (
                'Y 軸重疊比門檻。'
                '兩框 Y 重疊長度 / 較短框高度 ≥ 此值才視為同欄。'
                '建議範圍 0.1 ~ 0.6，預設 0.3。'
            )
        },
        'force_vertical': {
            'type': 'checkbox',
            'value': True,
            'description': '強制所有框視為直排（日文漫畫建議勾選，預設開啟）'
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self._last_raw_boxes: List[List[int]] = []
        self._load_model()

    @property
    def model_path(self) -> str:
        return self.params['model_path']['value']

    @property
    def conf_threshold(self) -> float:
        try:
            return float(self.params['conf_threshold']['value'])
        except (ValueError, KeyError):
            return 0.25

    @property
    def x_gap_ratio(self) -> float:
        try:
            return float(self.params['x_gap_ratio']['value'])
        except (ValueError, KeyError):
            return 1.5

    @property
    def y_overlap_ratio(self) -> float:
        try:
            return float(self.params['y_overlap_ratio']['value'])
        except (ValueError, KeyError):
            return 0.3

    @property
    def force_vertical(self) -> bool:
        val = self.params['force_vertical']['value']
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            return val != 0
        if isinstance(val, str):
            return val.strip().lower() == 'true'
        return bool(val)

    def setup_detector(self):
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.logger.error(f"找不到 YOLOv8 模型檔: {os.path.abspath(self.model_path)}")
            self.model = None
            return
        try:
            self.logger.info(f"正在載入 YOLOv8 模型: {self.model_path} ...")
            self.model = YOLO(self.model_path)
            self.logger.info("YOLOv8 引擎啟動成功！")
        except Exception as e:
            self.logger.error(f"YOLOv8 載入失敗: {e}")
            self.model = None

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'model_path':
            self._load_model()

    def _should_merge(
        self,
        a: List[int],
        b: List[int],
        is_vertical: bool,
        x_gap_ratio: float,
        y_overlap_ratio: float,
    ) -> bool:
        """
        a/b 格式：[x1, y1, x2, y2, orig_w]
        orig_w 合併時取 min 繼承，防止大框放寬門檻。
        """
        ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
        aw = a[4] if len(a) > 4 else (ax2 - ax1)
        bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
        bw = b[4] if len(b) > 4 else (bx2 - bx1)

        if is_vertical:
            x_gap = max(bx1 - ax2, ax1 - bx2, 0)
            ref_w = min(aw, bw)
            if ref_w <= 0 or x_gap > x_gap_ratio * ref_w:
                return False

            y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
            ref_h = min(ay2 - ay1, by2 - by1)
            if ref_h <= 0 or y_overlap / ref_h < y_overlap_ratio:
                return False
        else:
            ah = a[4] if len(a) > 4 else (ay2 - ay1)
            bh = b[4] if len(b) > 4 else (by2 - by1)
            y_gap = max(by1 - ay2, ay1 - by2, 0)
            ref_h = min(ah, bh)
            if ref_h <= 0 or y_gap > x_gap_ratio * ref_h:
                return False
            x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
            ref_w = min(ax2 - ax1, bx2 - bx1)
            if ref_w <= 0 or x_overlap / ref_w < y_overlap_ratio:
                return False

        return True

    def _merge_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        is_vert = self.force_vertical
        x_gap   = self.x_gap_ratio
        y_ov    = self.y_overlap_ratio

        boxes5 = [[x1, y1, x2, y2, x2 - x1]
                  for x1, y1, x2, y2 in (b[:4] for b in boxes)]

        changed = True
        while changed:
            changed = False
            used = [False] * len(boxes5)
            new_boxes = []

            for i, a in enumerate(boxes5):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2, aw = a

                for j in range(i + 1, len(boxes5)):
                    if used[j]:
                        continue
                    if self._should_merge(
                        [ax1, ay1, ax2, ay2, aw], boxes5[j],
                        is_vert, x_gap, y_ov,
                    ):
                        bx1, by1, bx2, by2, bw = boxes5[j]
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        aw  = min(aw, bw)
                        used[j] = True
                        changed = True

                new_boxes.append([ax1, ay1, ax2, ay2, aw])
                used[i] = True

            boxes5 = new_boxes

        return [[b[0], b[1], b[2], b[3]] for b in boxes5]

    def _detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        if self.model is None:
            return np.zeros(img.shape[:2], dtype=np.uint8), []

        im_h, im_w = img.shape[:2]
        results = self.model(img, conf=self.conf_threshold, verbose=False)

        raw_boxes: List[List[int]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                raw_boxes.append([x1, y1, x2, y2])

        merged_boxes = self._merge_boxes(raw_boxes)
        self._last_raw_boxes = raw_boxes

        blk_list = []
        for (x1, y1, x2, y2) in merged_boxes:
            xywh  = np.array([[x1, y1, x2 - x1, y2 - y1]])
            lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
            blk   = TextBlock(xyxy=[x1, y1, x2, y2], lines=lines)

            examine_textblk(blk, im_w, im_h)

            if self.force_vertical:
                blk.vertical        = True
                blk.src_is_vertical = True
                blk.angle           = 0
            else:
                is_vertical = (y2 - y1) >= (x2 - x1)
                blk.vertical        = is_vertical
                blk.src_is_vertical = is_vertical
                if not is_vertical:
                    blk.angle = 0

            blk._detected_font_size = 0
            blk_list.append(blk)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            mask[y1:y2, x1:x2] = 255

        return mask, blk_list