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
            'description': '信心度閾值 (0.0~1.0)，越低抓越多框，越高越嚴格'
        },
        'merge_margin': {
            'value': '10',
            'description': '碎框合併距離（px），調大會把更遠的框合併在一起'
        },
        'force_vertical': {
            'type': 'checkbox',
            'value': True,
            'description': '強制所有框視為直排（日文漫畫建議勾選）'
        }
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self._load_model()

    @property
    def model_path(self) -> str:
        return self.params['model_path']['value']

    @property
    def conf_threshold(self) -> float:
        try:
            return float(self.params['conf_threshold']['value'])
        except ValueError:
            return 0.25

    @property
    def merge_margin(self) -> int:
        try:
            return int(self.params['merge_margin']['value'])
        except ValueError:
            return 10

    @property
    def force_vertical(self) -> bool:
        return bool(self.params['force_vertical']['value'])

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

    def _merge_boxes(self, boxes: List[List[int]], margin: int) -> List[List[int]]:
        """反覆合併有重疊或相鄰（含 margin）的框，直到沒有可合併的為止"""
        merged = True
        while merged:
            merged = False
            new_boxes = []
            used = [False] * len(boxes)
            for i, a in enumerate(boxes):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = a
                for j, b in enumerate(boxes[i+1:], start=i+1):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2 = b
                    if (ax1 - margin <= bx2 and ax2 + margin >= bx1 and
                            ay1 - margin <= by2 and ay2 + margin >= by1):
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        used[j] = True
                        merged = True
                new_boxes.append([ax1, ay1, ax2, ay2])
                used[i] = True
            boxes = new_boxes
        return boxes

    def _detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        if self.model is None:
            return np.zeros(img.shape[:2], dtype=np.uint8), []

        im_h, im_w = img.shape[:2]

        results = self.model(img, conf=self.conf_threshold, verbose=False)

        raw_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                raw_boxes.append([x1, y1, x2, y2])

        merged_boxes = self._merge_boxes(raw_boxes, margin=self.merge_margin)

        blk_list = []
        for (x1, y1, x2, y2) in merged_boxes:
            xywh = np.array([[x1, y1, x2 - x1, y2 - y1]])
            lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
            blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=lines)

            # 用 examine_textblk 取得角度等資訊
            examine_textblk(blk, im_w, im_h)

            if self.force_vertical:
                # 強制直排：忽略 examine_textblk 對寬框的誤判
                blk.vertical = True
                blk.src_is_vertical = True
                blk.angle = 0
            else:
                # 用框的長寬比再做一次判斷，比 examine_textblk 的向量法更可靠
                # （因為我們只有一個矩形，向量法本質上就是比長寬）
                box_h = y2 - y1
                box_w = x2 - x1
                is_vertical = box_h >= box_w
                blk.vertical = is_vertical
                blk.src_is_vertical = is_vertical
                if not is_vertical:
                    blk.angle = 0

            # 字體大小由 OCR 完成後根據字數/框面積自動計算，不採用這裡的值
            blk._detected_font_size = 0
            blk_list.append(blk)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            mask[y1:y2, x1:x2] = 255
        return mask, blk_list