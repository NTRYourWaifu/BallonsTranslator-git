"""整句偵測：一句／一顆泡泡一框，取代「YOLO 行框＋CTD 補充」（2026-10-29 調研，報告 Ichaival/研究/reports/漫畫掃框與字級新工具.md）。

模型
  RF-DETR     Koharu Layout RF-DETR Seg 2XL（text／擬聲詞／泡泡／分格）。BT 的 Python 裝不進，
              借 BT-headless/tools/newmask 的獨立環境開常駐子程序（region_worker.py）。
  lordtrilink lordtrilink/manga-text-detector-v0（YOLO11s，Manga109-s 整句標註，text-region／sfx），本程序直接跑。
選項
  用泡泡分組   同一顆泡泡裡的多個文字塊併成一框（只有 RF-DETR 有泡泡）。
  補沒泡泡的字 跑原本的 YOLO 一行一框＋合併（吃 yolov8 的設定，不開 CTD），主模型沒蓋到的框補進來。
  保留擬聲詞   擬聲詞也出框，交給認字判要不要翻；關掉就連遮罩一起拿掉。跟文字框重疊的擬聲詞框一律丟掉。
  補CTD       再跑 CTD，前面都沒蓋到的框補進來（彩色 CG 寫在身上的字主模型會漏；CTD 會框大框，預設關）。
YOLO 的原始行框另外用來決定直排與 lines（落在框裡就算直排），字級初估才不會被整框寬度灌大。
"""
import json
import os
import os.path as osp
import subprocess
import tempfile
import threading
from copy import deepcopy
from typing import List, Tuple

import cv2
import numpy as np

from .base import register_textdetectors, TextDetectorBase, TextBlock
from utils.imgproc_utils import xywh2xyxypoly
from utils.textblock import examine_textblk, sort_regions
from utils.textblock_mask import canny_flood

BT_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
HEADLESS = osp.join(osp.dirname(BT_ROOT), 'BT-headless')
RF_PYTHON = osp.join(HEADLESS, 'tools', 'newmask', '.venv', 'Scripts', 'python.exe')
RF_MODELS = osp.join(HEADLESS, 'data', 'models', 'newmask')
WORKER = osp.join(osp.dirname(osp.abspath(__file__)), 'region_worker.py')
LT_MODEL = osp.join(BT_ROOT, 'data', 'models', 'lordtrilink', 'best.pt')
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


class _RFWorker:
    """常駐的 RF-DETR 子程序；一次只處理一頁。"""

    def __init__(self, logger):
        self.logger = logger
        self.proc = None
        self.lock = threading.Lock()
        self.tmp = tempfile.mkdtemp(prefix='bt_region_')

    def _start(self):
        if not osp.exists(RF_PYTHON):
            raise RuntimeError(f'找不到 RF-DETR 環境：{RF_PYTHON}（在 BT-headless/tools/newmask 跑 uv sync）')
        err = open(osp.join(self.tmp, 'worker_err.log'), 'ab')
        self.proc = subprocess.Popen([RF_PYTHON, '-I', WORKER, RF_MODELS], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=err, creationflags=CREATE_NO_WINDOW, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                                     encoding='utf-8', bufsize=1)
        msg = self._read()
        if msg.get('type') != 'ready':
            raise RuntimeError(f'RF-DETR 子程序沒有起來：{msg}')
        self.logger.info(f'RF-DETR 子程序已啟動（載入 {msg.get("load_s")} 秒）')

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError('RF-DETR 子程序結束了，詳見 ' + osp.join(self.tmp, 'worker_err.log'))
            line = line.strip()
            if line.startswith('{'):
                return json.loads(line)

    def run(self, img_bgr: np.ndarray) -> Tuple[list, np.ndarray]:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            src, out = osp.join(self.tmp, 'page.png'), osp.join(self.tmp, 'mask.png')
            cv2.imencode('.png', img_bgr)[1].tofile(src)
            self.proc.stdin.write(json.dumps({'img': src, 'mask_out': out}) + '\n')
            self.proc.stdin.flush()
            msg = self._read()
            if msg.get('type') != 'result':
                raise RuntimeError(f'RF-DETR 偵測失敗：{msg.get("msg")}')
            mask = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_GRAYSCALE)
            return msg['inst'], mask

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None


@register_textdetectors('整句偵測')
class DetectorRegion(TextDetectorBase):
    params = {
        '模型': {'type': 'selector', 'options': ['RF-DETR', 'lordtrilink'], 'value': 'RF-DETR'},
        '用泡泡分組': {'type': 'checkbox', 'value': True, 'controlled_by': '模型', 'visible_for_models': ['RF-DETR']},
        '補沒泡泡的字': {'type': 'checkbox', 'value': True},
        '保留擬聲詞': {'type': 'checkbox', 'value': True},
        '補CTD': {'type': 'checkbox', 'value': False},
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._rf = None
        self._lt = None
        self._yolo = None
        self._ctd = None

    def _get_ctd(self):
        if self._ctd is None:
            from .detector_ctd import ComicTextDetector
            self._ctd = ComicTextDetector()
            self._ctd.load_model()
        return self._ctd

    @staticmethod
    def _dedupe_sfx(texts: list, sfx: list) -> list:
        """擬聲詞框跟文字框重疊一半以上就丟（同一處被判成兩類）；被另一個擬聲詞框包住八成以上也丟。"""
        sfx = [s for s in sfx if not any(_inter(s, t) >= .5 * (_area(s) or 1) for t in texts)]
        sfx.sort(key=_area, reverse=True)
        out = []
        for s in sfx:
            if not any(_inter(s, o) >= .8 * (_area(s) or 1) for o in out):
                out.append(s)
        return out

    def _flag(self, key) -> bool:
        v = self.get_param_value(key)
        return v if isinstance(v, bool) else str(v).lower() == 'true'

    def _get_yolo(self):
        """原本的 YOLO 行偵測器，參數照使用者 yolov8 的設定，但不開 CTD。"""
        if self._yolo is None:
            from .detector_yolov8 import DetectorYOLOv8
            from utils.config import pcfg
            p = deepcopy(DetectorYOLOv8.params)
            saved = (pcfg.module.textdetector_params or {}).get('yolov8') or {}
            for k, v in saved.items():
                if k in p and isinstance(v, dict) and 'value' in v:
                    p[k]['value'] = v['value']
            p['enable_ctd_supplement']['value'] = False
            p['det_debug_log']['value'] = False
            y = DetectorYOLOv8.__new__(DetectorYOLOv8)
            y.params = p
            DetectorYOLOv8.__init__(y)
            self._yolo = y
        return self._yolo

    def _main_instances(self, img: np.ndarray) -> Tuple[list, np.ndarray]:
        """主模型 → ([{'cls','xyxy'}], 文字遮罩或 None)"""
        if self.get_param_value('模型') == 'lordtrilink':
            if self._lt is None:
                from ultralytics import YOLO
                self._lt = YOLO(LT_MODEL)
            res = self._lt(img, conf=0.05, iou=0.7, imgsz=1024, verbose=False)[0]
            inst = [{'cls': 'text' if int(c) == 0 else 'sfx', 'xyxy': [int(v) for v in b]}
                    for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy())]
            return inst, None
        if self._rf is None:
            self._rf = _RFWorker(self.logger)
        return self._rf.run(img)

    @staticmethod
    def _group_by_bubble(texts: list, bubbles: list) -> list:
        """每個文字塊歸到包住它（≥60%）的最小泡泡；同一顆泡泡的併成一框。"""
        groups, alone = {}, []
        for t in texts:
            a = _area(t) or 1
            cand = [(i, _area(b)) for i, b in enumerate(bubbles) if _inter(t, b) >= .6 * a]
            if cand:
                i = min(cand, key=lambda x: x[1])[0]
                groups.setdefault(i, []).append(t)
            else:
                alone.append(t)
        out = list(alone)
        for ts in groups.values():
            out.append([min(t[0] for t in ts), min(t[1] for t in ts), max(t[2] for t in ts), max(t[3] for t in ts)])
        return out

    def _make_blk(self, box, cols, im_w, im_h) -> TextBlock:
        x1, y1, x2, y2 = box
        inside = [c for c in cols if x1 <= (c[0] + c[2]) / 2 <= x2 and y1 <= (c[1] + c[3]) / 2 <= y2]
        vertical = bool(inside) or (y2 - y1) > 1.5 * (x2 - x1)
        if inside:
            inside.sort(key=lambda c: -c[0])
            xywh = np.array([[c[0], c[1], c[2] - c[0], c[3] - c[1]] for c in inside])
        else:
            xywh = np.array([[x1, y1, x2 - x1, y2 - y1]])
        lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
        blk = TextBlock(xyxy=[int(x1), int(y1), int(x2), int(y2)], lines=lines)
        examine_textblk(blk, im_w, im_h)
        blk.vertical = blk.src_is_vertical = vertical
        blk.angle = 0
        blk.obs.yolo_says_vertical = vertical
        if inside:
            blk.obs.yolo_col_width = float(np.median([c[2] - c[0] for c in inside]))
        blk.obs.yolo_box_w = float(x2 - x1)
        blk.obs.yolo_box_h = float(y2 - y1)
        return blk

    def _detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        im_h, im_w = img.shape[:2]
        inst, rf_mask = self._main_instances(img)
        clip = lambda b: [max(0, int(b[0])), max(0, int(b[1])), min(im_w, int(b[2])), min(im_h, int(b[3]))]
        texts = [clip(i['xyxy']) for i in inst if i['cls'] == 'text']
        sfx = [clip(i['xyxy']) for i in inst if i['cls'] == 'sfx']
        bubbles = [clip(i['xyxy']) for i in inst if i['cls'] == 'bubble']
        if bubbles and self._flag('用泡泡分組'):
            texts = self._group_by_bubble(texts, bubbles)
        sfx = self._dedupe_sfx(texts, sfx)
        boxes = texts + (sfx if self._flag('保留擬聲詞') else [])

        yolo = self._get_yolo()
        yolo_mask, yolo_blks = yolo._run_yolo(img) if yolo.model is not None else (None, [])
        cols = [b[:4] for b in yolo._last_raw_boxes] if yolo.model is not None else []

        blk_list = [self._make_blk(b, cols, im_w, im_h) for b in boxes if _area(b) > 0]
        sfx_ids = {id(b) for b, src in zip(blk_list, [b for b in boxes if _area(b) > 0]) if src in sfx}
        extra, ctd_mask = [], None
        covered = texts + sfx
        if self._flag('補沒泡泡的字'):
            for yb in yolo_blks:
                a = _area(yb.xyxy) or 1
                if sum(_inter(yb.xyxy, c) for c in covered) < .3 * a:
                    extra.append(yb)
                    covered.append(clip(yb.xyxy))
        extra_ids = {id(b) for b in extra}
        ctd_ids = set()
        if self._flag('補CTD'):
            ctd_mask, ctd_blks = self._get_ctd().detect(img)
            for cb in ctd_blks:
                a = _area(cb.xyxy) or 1
                if sum(_inter(cb.xyxy, c) for c in covered) < .3 * a:
                    extra.append(cb)
                    ctd_ids.add(id(cb))
        blk_list = sort_regions(blk_list + extra)

        # 遮罩：RF-DETR 用 comic-text-mask 筆畫；lordtrilink 沒有遮罩就照 YOLO 的做法在框內抽筆畫。
        # 補進來的 YOLO 框用 YOLO 自己的遮罩。最後只留框內的，關掉擬聲詞時它們的筆畫也一起拿掉。
        mask = np.zeros((im_h, im_w), np.uint8)
        keep = np.zeros((im_h, im_w), np.uint8)
        for blk in blk_list:
            x1, y1, x2, y2 = clip(blk.xyxy)
            if x2 <= x1 or y2 <= y1:
                continue
            px, py = max(2, (x2 - x1) // 20), max(2, (y2 - y1) // 20)
            keep[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px] = 255
            if id(blk) in extra_ids and yolo_mask is not None:
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], yolo_mask[y1:y2, x1:x2])
            elif id(blk) in ctd_ids and ctd_mask is not None:
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], ctd_mask[y1:y2, x1:x2])
            elif rf_mask is None and id(blk) not in sfx_ids:   # lordtrilink 的擬聲詞框常很大，框內抽筆畫會整片塗；交給認字判
                try:
                    tm, _, _ = canny_flood(img[y1:y2, x1:x2])
                    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], tm)
                except Exception:
                    mask[y1:y2, x1:x2] = 255
        if rf_mask is not None:
            mask = np.maximum(mask, rf_mask)
        mask[keep == 0] = 0
        return mask, blk_list

    def unload_model(self, empty_cache=False):
        if self._rf is not None:
            self._rf.close()
            self._rf = None
        self._lt = None
        if self._yolo is not None:
            self._yolo.model = None
            self._yolo = None
        if self._ctd is not None:
            self._ctd.unload_model()
            self._ctd = None
        return super().unload_model(empty_cache)
