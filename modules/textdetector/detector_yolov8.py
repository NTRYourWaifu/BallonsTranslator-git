import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
from qtpy.QtCore import QObject, Signal

from .base import register_textdetectors, TextDetectorBase, TextBlock
from utils.imgproc_utils import xywh2xyxypoly
from utils.textblock import examine_textblk


class DetectorSignals(QObject):
    event = Signal(str)


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
        # ── CTD 補充偵測 ─────────────────────────────────────
        'enable_ctd_supplement': {
            'type': 'checkbox',
            'value': True,
            'description': '啟用 CTD 補充偵測（補抓橫排標題/旁白等 YOLOv8 漏框的區域）'
        },
        'ctd_overlap_threshold': {
            'value': '0.3',
            'description': (
                'CTD 框被 YOLO 框覆蓋的面積比例門檻。'
                '超過此值視為重複，以 YOLO 結果為準捨棄 CTD 框。'
                '建議範圍 0.2 ~ 0.5，預設 0.3。'
            )
        },
        # ─────────────────────────────────────────────────────
        'det_debug_log': {
            'type': 'checkbox',
            'value': False,
            'description': '輸出偵測流程 log 並將最終框列表存成 det_debug_*.jpg'
        },
        'det_debug_verbose': {
            'type': 'checkbox',
            'value': False,
            'description': 'CTD×YOLO 每對比對的詳細 overlap 數值（需同時開 det_debug_log）'
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self._ctd: Optional['ComicTextDetector'] = None   # 懶載入
        self._last_raw_boxes: List[List[int]] = []
        self.det_signals = DetectorSignals()
        self.current_imgname: str = ''
        self._load_model()

    # ── properties ───────────────────────────────────────────

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

    @property
    def enable_ctd_supplement(self) -> bool:
        val = self.params['enable_ctd_supplement']['value']
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            return val != 0
        if isinstance(val, str):
            return val.strip().lower() == 'true'
        return bool(val)

    @property
    def ctd_overlap_threshold(self) -> float:
        try:
            return float(self.params['ctd_overlap_threshold']['value'])
        except (ValueError, KeyError):
            return 0.3

    @property
    def det_debug_log(self) -> bool:
        val = self.params['det_debug_log']['value']
        if isinstance(val, bool):
            return val
        return bool(val)

    @property
    def det_debug_verbose(self) -> bool:
        val = self.params.get('det_debug_verbose', {}).get('value', False)
        if isinstance(val, bool):
            return val
        return bool(val)

    # ── 模型載入 ─────────────────────────────────────────────

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

    def _ensure_ctd(self) -> Optional['ComicTextDetector']:
        """懶載入 CTD，載入失敗時回傳 None 並記 log，不中斷主流程。"""
        if self._ctd is not None:
            return self._ctd
        try:
            from .detector_ctd import ComicTextDetector
            self.logger.info("正在懶載入 CTD 補充偵測模型 ...")
            ctd = ComicTextDetector()
            ctd.load_model()
            self._ctd = ctd
            self.logger.info("CTD 補充偵測模型載入完成。")
        except Exception as e:
            self.logger.error(f"CTD 補充偵測模型載入失敗，將跳過補充步驟: {e}")
            self._ctd = None
        return self._ctd

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'model_path':
            self._load_model()
        # 取消勾選時釋放 CTD 模型節省記憶體
        if param_key == 'enable_ctd_supplement' and not self.enable_ctd_supplement:
            if self._ctd is not None:
                self._ctd.unload_model(empty_cache=True)
                self._ctd = None
                self.logger.info("CTD 補充偵測模型已卸載。")

    # ── YOLO 欄位合併邏輯（原有，未動） ─────────────────────

    def _should_merge(
        self,
        a: List[int],
        b: List[int],
        is_vertical: bool,
        x_gap_ratio: float,
        y_overlap_ratio: float,
    ) -> bool:
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

        # 合併後清理：移除被大框包含的殘留小框
        boxes5 = self._remove_contained_boxes(boxes5)
        return boxes5

    # ── 框工具函數 ───────────────────────────────────────────

    @staticmethod
    def _box_area(xyxy: List[int]) -> int:
        return max((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]), 1)

    @staticmethod
    def _intersect_area(a: List[int], b: List[int]) -> int:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        return (ix2 - ix1) * (iy2 - iy1)

    def _overlap_ratio(self, a: List[int], b: List[int]) -> float:
        """交集面積 / min(a面積, b面積)：兩框重疊程度，不受大小比例影響。"""
        inter = self._intersect_area(a, b)
        if inter == 0:
            return 0.0
        return inter / min(self._box_area(a), self._box_area(b))

    def _yolo_covered_by_ctd(self, yolo_xyxy: List[int], ctd_xyxy: List[int]) -> float:
        """YOLO 框有多少比例被 CTD 框覆蓋（交集 / YOLO 框面積）。
        用於「CTD 框包住 YOLO 框大部分但往外延伸」的情況。"""
        inter = self._intersect_area(yolo_xyxy, ctd_xyxy)
        if inter == 0:
            return 0.0
        return inter / self._box_area(yolo_xyxy)

    # ── YOLO 自身包含框清理 ───────────────────────────────────

    def _remove_contained_boxes(self, boxes5: List[List[int]]) -> List[List[int]]:
        """
        移除被其他框大比例包含的小框（YOLO 大框中殘留的小框）。
        判定：小框與大框的交集 / 小框面積 >= 85% → 小框刪除。
        """
        keep = [True] * len(boxes5)
        for i, a in enumerate(boxes5):
            if not keep[i]:
                continue
            a_area = self._box_area(a[:4])
            for j, b in enumerate(boxes5):
                if i == j or not keep[j]:
                    continue
                inter = self._intersect_area(a[:4], b[:4])
                if inter == 0:
                    continue
                if inter / a_area >= 0.85:   # a 被 b 包含 85% 以上 → 刪 a
                    keep[i] = False
                    break
        removed = keep.count(False)
        if self.det_debug_log and removed:
            self.logger.debug(f"YOLO 包含框清理：移除 {removed} 個被包含的小框")
        return [b for b, k in zip(boxes5, keep) if k]

    # ── CTD 補充合併 ─────────────────────────────────────────

    def _merge_ctd_supplement(
        self,
        yolo_blks: List[TextBlock],
        ctd_blks: List[TextBlock],
    ) -> List[TextBlock]:
        """
        將 CTD 結果補充進 YOLO 結果。

        主判斷：CTD blk.vertical
        ─────────────────────────────────────────────────────────
        CTD 框為橫排（vertical=False）：
          - 與它重疊的 YOLO 框必為誤判的直排框 → 全部刪除，用 CTD 框
          - 無重疊 → 直接補入（YOLO 完全漏偵測的橫排）

        CTD 框為直排（vertical=True）：
          - 無重疊 → 補入
          - 有重疊 → 面積比判斷：
              CTD/YOLO 面積比 < area_ratio_thresh → 信 YOLO，捨棄 CTD
              CTD/YOLO 面積比 >= area_ratio_thresh → CTD 框取代 YOLO 框

        輔助驗證（A-2）：
          - 若 CTD.vertical 與面積比推斷的排列方向不一致 → log 警告
        """
        threshold        = self.ctd_overlap_threshold
        area_ratio_thresh = 2.0   # 直排情況下，CTD/YOLO 面積比超過此值才取代

        yolo_replaced: set = set()
        actions = []  # ('add' | 'replace' | 'drop', ctd_blk, yolo_idx_or_None)

        for ctd_blk in ctd_blks:
            ctd_area    = self._box_area(ctd_blk.xyxy)
            ctd_is_hori = not ctd_blk.vertical   # True = 橫排

            # ── 找所有有意義重疊的 YOLO 框 ──────────────────
            overlapping = []
            for i, yb in enumerate(yolo_blks):
                ov  = self._overlap_ratio(ctd_blk.xyxy, yb.xyxy)
                cov = self._yolo_covered_by_ctd(yb.xyxy, ctd_blk.xyxy)
                if ov >= threshold or cov >= threshold:
                    overlapping.append((i, yb))
                if self.det_debug_log and self.det_debug_verbose:
                    self.logger.debug(
                        f"  CTD{ctd_blk.xyxy}(vert={ctd_blk.vertical})"
                        f" vs YOLO{yb.xyxy}"
                        f" overlap={ov:.2f} yolo_cov={cov:.2f}"
                        f" ctd_area={ctd_area} yolo_area={self._box_area(yb.xyxy)}"
                    )

            # ── 無重疊：直接補入 ─────────────────────────────
            if not overlapping:
                actions.append(('add', ctd_blk, None))
                continue

            # ── 有重疊：取最大重疊 YOLO 框做判斷 ────────────
            best_yolo_idx, best_yolo_blk = max(
                overlapping, key=lambda t: self._box_area(t[1].xyxy)
            )
            yolo_area  = self._box_area(best_yolo_blk.xyxy)
            area_ratio = ctd_area / max(yolo_area, 1)

            # 面積比推斷的排列方向（面積比大 → 推斷為橫排信號）
            area_says_hori = area_ratio >= area_ratio_thresh

            # ── A-2 異常警告：vertical 與面積比推斷不一致 ────
            # 永遠 log（不限 det_debug_log），方便排查偵測品質問題

            # ── 主判斷 ───────────────────────────────────────
            if ctd_is_hori:
                # CTD 判斷為橫排 → 刪除所有重疊的 YOLO 框，用 CTD
                for yolo_idx, _ in overlapping:
                    yolo_replaced.add(yolo_idx)
                actions.append(('replace', ctd_blk, best_yolo_idx))
            else:
                # CTD 判斷為直排 → 面積比決定
                if area_ratio >= area_ratio_thresh:
                    # CTD 明顯更大（異常，直排不該差這麼多）→ 取代
                    actions.append(('replace', ctd_blk, best_yolo_idx))
                    yolo_replaced.add(best_yolo_idx)
                else:
                    # 面積差不多 → 信 YOLO，捨棄 CTD
                    actions.append(('drop', ctd_blk, None))

        # ── 組裝結果 ─────────────────────────────────────────
        result: List[TextBlock] = [
            blk for i, blk in enumerate(yolo_blks)
            if i not in yolo_replaced
        ]

        n_add = n_replace = n_drop = 0
        for action, ctd_blk, yolo_idx in actions:
            if action == 'add':
                ctd_blk.det_model = 'ctd_supplement'
                ctd_blk.obs.ctd_says_vertical = ctd_blk.vertical
                result.append(ctd_blk)
                n_add += 1
            elif action == 'replace':
                ctd_blk.det_model = 'ctd_replace'
                ctd_blk.obs.ctd_says_vertical = ctd_blk.vertical
                result.append(ctd_blk)
                n_replace += 1
            else:
                n_drop += 1

        if self.det_debug_log:
            self.logger.debug(
                f"CTD 合併：補入={n_add} 取代={n_replace} 捨棄={n_drop} "
                f"| 最終={len(result)} 框"
            )

        return result

    # ── 主偵測流程 ────────────────────────────────────────────

    def _run_yolo(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        """純 YOLO 偵測，回傳 (mask, blk_list)。"""
        im_h, im_w = img.shape[:2]
        results = self.model(img, conf=self.conf_threshold, verbose=False)

        raw_boxes: List[List[int]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                raw_boxes.append([x1, y1, x2, y2])

        merged_boxes = self._merge_boxes(raw_boxes)
        self._last_raw_boxes = raw_boxes

        blk_list: List[TextBlock] = []
        for (x1, y1, x2, y2, orig_w) in merged_boxes:
            xywh  = np.array([[x1, y1, x2 - x1, y2 - y1]])
            lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
            blk   = TextBlock(xyxy=[x1, y1, x2, y2], lines=lines)

            examine_textblk(blk, im_w, im_h)

            if self.force_vertical:
                blk.vertical        = True
                blk.src_is_vertical = True
                blk.angle           = 0
                blk.obs.yolo_says_vertical = True
            else:
                is_vertical = (y2 - y1) >= (x2 - x1)
                blk.vertical        = is_vertical
                blk.src_is_vertical = is_vertical
                if not is_vertical:
                    blk.angle = 0
                blk.obs.yolo_says_vertical = is_vertical

            blk.obs.yolo_col_width = float(orig_w)
            blk.obs.yolo_box_w     = float(x2 - x1)
            blk.obs.yolo_box_h     = float(y2 - y1)

            if self.det_debug_log:
                self.logger.debug(
                    f"YOLO box=({x1},{y1},{x2},{y2}) "
                    f"merged_w={x2-x1} orig_w={orig_w} "
                    f"vert={blk.vertical} "
                )

            blk_list.append(blk)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for blk in blk_list:
            bx1, by1, bx2, by2 = blk.xyxy
            mask[by1:by2, bx1:bx2] = 255

        return mask, blk_list

    def _detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        if self.model is None:
            return np.zeros(img.shape[:2], dtype=np.uint8), []

        # ── CTD 補充偵測關閉：直接走原有流程 ──────────────────
        if not self.enable_ctd_supplement:
            return self._run_yolo(img)

        # ── CTD 補充偵測開啟：兩者並行 ────────────────────────
        ctd = self._ensure_ctd()
        if ctd is None:
            # CTD 載入失敗，降級為純 YOLO
            return self._run_yolo(img)

        # ThreadPoolExecutor 讓兩個推理在不同執行緒排隊，
        # CUDA 會自動透過 stream 盡量並行；
        # 即使無法真正同步，總時間也不超過較慢的那個 + 少量 overhead。
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_yolo = pool.submit(self._run_yolo, img)
            f_ctd  = pool.submit(ctd.detect, img)      # 使用 base.detect() 以觸發 det_model 標記

            yolo_mask, yolo_blks = f_yolo.result()
            _ctd_mask, ctd_blks  = f_ctd.result()

        # 合併：YOLO 主體 + CTD 補充（去重）
        merged_blks = self._merge_ctd_supplement(yolo_blks, ctd_blks)

        # mask 取聯集（CTD 的精確 mask 比 YOLO 矩形填充更準確）
        merged_mask = np.maximum(yolo_mask, _ctd_mask)

        if self.det_debug_log:
            self.logger.debug(
                f"偵測完成：YOLO={len(yolo_blks)} CTD={len(ctd_blks)} "
                f"合併後={len(merged_blks)}"
            )
            self._save_debug_img(img, yolo_blks, ctd_blks, merged_blks)

        return merged_mask, merged_blks

    # ── Debug 視覺化 ──────────────────────────────────────────

    def _save_debug_img(
        self,
        img: np.ndarray,
        yolo_blks: List[TextBlock],
        ctd_blks: List[TextBlock],
        merged_blks: List[TextBlock],
    ):
        """
        det_debug_log 開啟時，把三層結果畫在原圖上並存到 logs/det_debug_*.jpg。

        顏色規則：
          藍框  = YOLO 原始結果（合併後）
          綠框  = CTD 原始結果
          紅框  = 最終合併結果（帶白底編號）

        log 同步輸出每個最終框的 index / xyxy / det_model，
        讓畫面上的編號與 log 一一對應。
        """
        import time
        from utils import shared as _shared

        canvas = img.copy()
        h, w = canvas.shape[:2]
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(1.0, w / 1200))
        thick      = max(1, int(font_scale * 2))
        pad        = 3

        # ── 藍框：YOLO ────────────────────────────────────────
        for blk in yolo_blks:
            x1, y1, x2, y2 = blk.xyxy
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 100, 0), thick)

        # ── 綠框：CTD ─────────────────────────────────────────
        for blk in ctd_blks:
            x1, y1, x2, y2 = blk.xyxy
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 80), thick)

        # ── 紅框 + 白底編號：最終合併結果 ────────────────────
        self.logger.debug("── 最終框列表 ──────────────────────────────")
        for idx, blk in enumerate(merged_blks):
            x1, y1, x2, y2 = blk.xyxy
            det = getattr(blk, 'det_model', 'yolov8')
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 220), thick + 1)

            # 白底編號標籤
            label = str(idx)
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thick)
            lx = max(0, x1)
            ly = max(th + pad * 2, y1)
            cv2.rectangle(
                canvas,
                (lx, ly - th - pad * 2),
                (lx + tw + pad * 2, ly),
                (255, 255, 255), cv2.FILLED
            )
            cv2.putText(
                canvas, label,
                (lx + pad, ly - pad),
                font, font_scale, (0, 0, 0), thick, cv2.LINE_AA
            )

            self.logger.debug(
                f"  [{idx:02d}] xyxy={blk.xyxy} "
                f"vert={blk.vertical} "
                f"det={det} "
                f"fs={getattr(blk, '_detected_font_size', -1):.1f}"
            )
        self.logger.debug("────────────────────────────────────────────")

        # ── 存檔 ─────────────────────────────────────────────
        log_dir = _shared.LOGGING_PATH
        os.makedirs(log_dir, exist_ok=True)
        ts  = int(time.time() * 1000) % 1_000_000   # 6 位毫秒尾數，避免覆蓋
        out = os.path.join(log_dir, f"det_debug_{ts}.jpg")
        cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.logger.debug(f"Debug 圖已存至：{out}")