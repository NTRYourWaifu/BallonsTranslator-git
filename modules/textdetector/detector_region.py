"""整句偵測：一句／一顆泡泡一框，取代「YOLO 行框＋CTD 補充」（2026-10-29 調研，報告 Ichaival/研究/reports/漫畫掃框與字級新工具.md）。

模型
  RF-DETR     Koharu Layout RF-DETR Seg 2XL（text／擬聲詞／泡泡／分格）。BT 的 Python 裝不進，
              借 BT-headless/tools/newmask 的獨立環境開常駐子程序（region_worker.py）。
  lordtrilink lordtrilink/manga-text-detector-v0（YOLO11s，Manga109-s 整句標註，text-region／sfx），本程序直接跑。
選項
  lordtrilink補漏 RF-DETR 漏掉的字用 lordtrilink 補：它的框被 RF-DETR 的框蓋不到三成、框裡描得到筆畫（≥ 框面積 2%）就補進來，
              但把 RF 的台詞框和擬聲詞框併成一塊、或沒帶來 RF 漏掉的字的不收（lt_reject，T-30 規則①），
              筆畫用 comic-text-mask 只描這幾個框（T-30，11-04 實測補進 66 框，彩色 CG 整頁沒框的頁靠它）。
  用泡泡分組   同一顆泡泡裡的多個文字塊併成一框（只有 RF-DETR 有泡泡）。T-30 起預設關：斜對角相連的兩顆泡泡會被併成一框。
  新描筆畫     lordtrilink 的框改用 comic-text-mask 描筆畫（借 RF-DETR 那個子程序，但不跑 RF-DETR）；關著是舊的 canny_flood。
  補沒泡泡的字 跑原本的 YOLO 一行一框＋合併（吃 yolov8 的設定，不開 CTD），主模型沒蓋到的框補進來。
              T-30 起預設關：有 lordtrilink 補漏後它幾乎補不到東西，補進來的多半是誤框。
  保留擬聲詞   擬聲詞也出框，交給認字判要不要翻；關掉就連遮罩一起拿掉。跟文字框重疊一半以上的擬聲詞框丟掉。
重疊框一律照 T-30 規則 H 清（clean_overlaps），見該函式。
  補CTD       再跑 CTD，前面都沒蓋到的框補進來（彩色 CG 寫在身上的字主模型會漏；CTD 會框大框，預設關）。
YOLO 的原始行框另外用來決定直排與 lines（落在框裡就算直排），字級初估才不會被整框寬度灌大。
環境變數
  BT_REGION_IDLE_CLOSE  RF-DETR 子程序閒置幾秒就關（放掉顯存），0＝不關（GUI 預設）；BT-headless 設 60，一本偵測完就放。
  BT_REGION_FALLBACK    1（預設）＝整句偵測出錯（子程序起不來、顯存不夠）那頁退回舊偵測 yolov8＋CTD，0＝照樣丟錯。
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


INSIDE, LOOSE_FILL, LOOSE_COV = .8, .3, .85
LT_COVER, LT_STROKE = .3, .02
# lordtrilink 補漏的擋法（T-30 規則①，11-04，實驗 Ichaival/lab/T30_掃框收尾/rule_n.py）
LT_RESID, LT_PAD, LT_PAD_PAGE, LT_JOIN = .05, .1, .005, .01


def _contains(a, b):
    """b 被 a 包住八成以上（a 不比 b 小）。"""
    return _area(a) >= _area(b) and _inter(a, b) >= INSIDE * (_area(b) or 1)


def _loose(o, inner, mask) -> bool:
    """文字框是不是鬆散的併框：裡面最外層的小框面積加總不到三成，卻蓋到框裡八成五以上的筆畫。"""
    ls = [b for b in inner if not any(c is not b and _contains(c, b) and _area(c) > _area(b) for c in inner)]
    if not ls or mask is None or sum(_area(b) for b in ls) >= LOOSE_FILL * _area(o):
        return False
    sub = mask[o[1]:o[3], o[0]:o[2]] > 0
    k = np.zeros_like(sub)
    for b in ls:
        k[max(0, b[1] - o[1]):max(0, b[3] - o[1]), max(0, b[0] - o[0]):max(0, b[2] - o[0])] = True
    return (sub & k).sum() >= LOOSE_COV * max(1, sub.sum())


def _residual(o, boxes, mask) -> float:
    """o 框裡的筆畫，被 boxes（各外擴一成＋整頁短邊 0.5%）蓋不到的部分，相鄰的字併成塊後最大一塊佔 o 框筆畫的比例。"""
    x1, y1, x2, y2 = o
    sub = mask[y1:y2, x1:x2] > 0
    n = int(sub.sum())
    if n == 0:
        return 0.0
    pad = int(min(mask.shape[:2]) * LT_PAD_PAGE)
    k = np.zeros_like(sub)
    for r in boxes:
        px, py = int((r[2] - r[0]) * LT_PAD) + pad, int((r[3] - r[1]) * LT_PAD) + pad
        r = [r[0] - px, r[1] - py, r[2] + px, r[3] + py]
        if _inter(o, r):
            k[max(0, r[1] - y1):max(0, r[3] - y1), max(0, r[0] - x1):max(0, r[2] - x1)] = True
    un = (sub & ~k).astype(np.uint8)
    if not un.any():
        return 0.0
    ks = max(3, int(min(mask.shape[:2]) * LT_JOIN))
    cnt, lab = cv2.connectedComponents(cv2.dilate(un, np.ones((ks, ks), np.uint8)), connectivity=8)
    sizes = np.bincount(lab[un > 0].ravel(), minlength=cnt)[1:]
    return float(sizes.max()) / n if sizes.size else 0.0


def lt_reject(cand, rf_items, mask) -> bool:
    """T-30 規則①：lordtrilink 補漏候選框不收的兩種情況（rf_items＝RF 的 [{'xyxy','sfx'}]，mask＝候選框的筆畫遮罩）：
      - 同時包著 RF 的文字框和擬聲詞框（各八成在內）＝把台詞和擬聲詞併成一塊
        （11-04 字級測試 a05 第 157、212 頁：大鬆框收進來把擬聲詞一起消掉、排成超大字，台詞只讀到擬聲詞）；
      - RF 沒蓋到的筆畫，相鄰字併成塊後最大一塊不到候選框筆畫的 5%＝沒帶來 RF 漏掉的字。
    RF 只框到一部分（漏了句尾、一欄只框半截）時照收，不然那截字就沒框。"""
    inside = [it for it in rf_items if _contains(cand, it['xyxy'])]
    if any(it['sfx'] for it in inside) and any(not it['sfx'] for it in inside):
        return True
    return _residual(cand, [it['xyxy'] for it in rf_items], mask) < LT_RESID


def clean_overlaps(items: list, mask=None) -> list:
    """T-30 規則 H（11-04，實驗 Ichaival/lab/T30_掃框收尾/dedupe_h.py）：items＝[{'xyxy', 'sfx', ...}]，回留下的（照原順序）。

    框由大到小看：
      - 被已留下的框包住八成以上 → 丟（留外框：一句一框，不拆成一行一框）；
      - 擬聲詞框包著文字框 → 自己丟，裡面的框照同樣規則往下看（擬聲詞大框不吞對白）；
      - 文字框是鬆散的併框（見 _loose；mask 沒給就不看）→ 自己丟、往下看；
      - 其他 → 留。
    70 頁標準答案：有對白的組 82%（刪組合大框 49%、Jev 是非題 84%）。
    """
    order = sorted(range(len(items)), key=lambda i: -_area(items[i]['xyxy']))
    keep = []
    for i in order:
        b = items[i]
        if any(_contains(items[k]['xyxy'], b['xyxy']) for k in keep):
            continue
        inner = [j for j in range(len(items)) if j != i and _contains(b['xyxy'], items[j]['xyxy'])]
        if b['sfx'] and any(not items[j]['sfx'] for j in inner):
            continue
        if not b['sfx'] and _loose(b['xyxy'], [items[j]['xyxy'] for j in inner], mask):
            continue
        keep.append(i)
    return [items[i] for i in sorted(keep)]


class _RFWorker:
    """常駐的 RF-DETR 子程序；一次只處理一頁。閒置 BT_REGION_IDLE_CLOSE 秒就關掉，下一頁再開。"""

    def __init__(self, logger):
        self.logger = logger
        self.proc = None
        self.lock = threading.Lock()
        self.tmp = tempfile.mkdtemp(prefix='bt_region_')
        self.idle = float(os.environ.get('BT_REGION_IDLE_CLOSE', '0') or 0)
        self._timer = None

    def _arm_idle(self):
        if self.idle <= 0:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.idle, self._idle_close)
        self._timer.daemon = True
        self._timer.start()

    def _idle_close(self):
        with self.lock:
            if self.proc is not None:
                self.logger.info(f'RF-DETR 子程序閒置 {self.idle:.0f} 秒，先關掉放顯存')
                self._close()

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

    def run(self, img_bgr: np.ndarray, boxes: list = None) -> Tuple[list, np.ndarray]:
        """boxes 給了就只在這些框裡描筆畫（不跑 RF-DETR）。"""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            src, out = osp.join(self.tmp, 'page.npy'), osp.join(self.tmp, 'mask.png')
            np.save(src, np.ascontiguousarray(img_bgr))   # 直接存陣列：大頁 PNG 壓縮要 0.1 秒以上
            req = {'img': src, 'mask_out': out}
            if boxes is not None:
                req['boxes'] = [[int(v) for v in b] for b in boxes]
            self.proc.stdin.write(json.dumps(req) + '\n')
            self.proc.stdin.flush()
            msg = self._read()
            if msg.get('type') != 'result':
                raise RuntimeError(f'RF-DETR 偵測失敗：{msg.get("msg")}')
            mask = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_GRAYSCALE)
            self._arm_idle()
            return msg['inst'], mask

    def close(self):
        if self._timer is not None:
            self._timer.cancel()
        with self.lock:
            self._close()

    def _close(self):
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
        'lordtrilink補漏': {'type': 'checkbox', 'value': True, 'controlled_by': '模型', 'visible_for_models': ['RF-DETR']},
        '用泡泡分組': {'type': 'checkbox', 'value': False, 'controlled_by': '模型', 'visible_for_models': ['RF-DETR']},
        '新描筆畫': {'type': 'checkbox', 'value': False, 'controlled_by': '模型', 'visible_for_models': ['lordtrilink']},
        '補沒泡泡的字': {'type': 'checkbox', 'value': False},
        '保留擬聲詞': {'type': 'checkbox', 'value': True},
        '補CTD': {'type': 'checkbox', 'value': False},
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._rf = None
        self._lt = None
        self._yolo = None
        self._ctd = None
        self._fallback = None

    def _get_fallback(self):
        """出錯時退回的舊偵測：yolov8 照使用者存的設定（含 CTD 補充）。"""
        if self._fallback is None:
            from .detector_yolov8 import DetectorYOLOv8
            from utils.config import pcfg
            p = deepcopy(DetectorYOLOv8.params)
            saved = (pcfg.module.textdetector_params or {}).get('yolov8') or {}
            for k, v in saved.items():
                if k in p and isinstance(v, dict) and 'value' in v:
                    p[k]['value'] = v['value']
            y = DetectorYOLOv8.__new__(DetectorYOLOv8)
            y.params = p
            DetectorYOLOv8.__init__(y)
            self._fallback = y
        return self._fallback

    def _get_ctd(self):
        if self._ctd is None:
            from .detector_ctd import ComicTextDetector
            self._ctd = ComicTextDetector()
            self._ctd.load_model()
        return self._ctd

    @staticmethod
    def _dedupe_sfx(texts: list, sfx: list) -> list:
        """擬聲詞框跟文字框重疊一半以上就丟（同一處被判成兩類）。包住關係已由 clean_overlaps 清過。"""
        return [s for s in sfx if not any(_inter(s, t) >= .5 * (_area(s) or 1) for t in texts)]

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

    def _lt_instances(self, img: np.ndarray) -> list:
        if self._lt is None:
            from ultralytics import YOLO
            self._lt = YOLO(LT_MODEL)
        res = self._lt(img, conf=0.05, iou=0.7, imgsz=1024, verbose=False)[0]
        return [{'cls': 'text' if int(c) == 0 else 'sfx', 'xyxy': [int(v) for v in b]}
                for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy())]

    def _get_rf(self) -> _RFWorker:
        if self._rf is None:
            self._rf = _RFWorker(self.logger)
        return self._rf

    def _main_instances(self, img: np.ndarray) -> Tuple[list, np.ndarray]:
        """主模型 → ([{'cls','xyxy'}], 文字遮罩或 None)"""
        if self.get_param_value('模型') == 'lordtrilink':
            return self._lt_instances(img), None
        return self._get_rf().run(img)

    def _lt_fill(self, img: np.ndarray, items: list, clip) -> Tuple[list, np.ndarray]:
        """lordtrilink 補漏：被已有框蓋不到三成、框裡描得到筆畫的框補進來 → (補進來的 items, 它們的筆畫遮罩或 None)"""
        cand = []
        for i in self._lt_instances(img):
            b = clip(i['xyxy'])
            if _area(b) and sum(_inter(b, it['xyxy']) for it in items) < LT_COVER * _area(b):
                cand.append({'xyxy': b, 'sfx': i['cls'] == 'sfx', 'src': 'lt'})
        if not cand:
            return [], None
        m = self._get_rf().run(img, boxes=[c['xyxy'] for c in cand])[1]
        add = [c for c in cand if np.count_nonzero(m[c['xyxy'][1]:c['xyxy'][3], c['xyxy'][0]:c['xyxy'][2]])
               >= LT_STROKE * _area(c['xyxy']) and not lt_reject(c['xyxy'], items, m)]
        return add, m

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
        try:
            return self._detect_region(img)
        except Exception as e:
            if os.environ.get('BT_REGION_FALLBACK', '1') != '1':
                raise
            self.logger.warning(f'整句偵測失敗，這頁退回舊偵測（yolov8＋CTD）：{type(e).__name__}: {e}')
            print(f'[整句偵測] 失敗、這頁退回舊偵測：{type(e).__name__}: {e}', flush=True)
            fb = self._get_fallback()
            if not fb.all_model_loaded():
                fb.load_model()
            return fb._detect(img)

    def _detect_region(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        im_h, im_w = img.shape[:2]
        inst, rf_mask = self._main_instances(img)
        clip = lambda b: [max(0, int(b[0])), max(0, int(b[1])), min(im_w, int(b[2])), min(im_h, int(b[3]))]
        texts = [clip(i['xyxy']) for i in inst if i['cls'] == 'text']
        sfx = [clip(i['xyxy']) for i in inst if i['cls'] == 'sfx']
        bubbles = [clip(i['xyxy']) for i in inst if i['cls'] == 'bubble']
        if bubbles and self._flag('用泡泡分組'):
            texts = self._group_by_bubble(texts, bubbles)
        items = [{'xyxy': b, 'sfx': False} for b in texts if _area(b)] + [{'xyxy': b, 'sfx': True} for b in sfx if _area(b)]
        if self.get_param_value('模型') == 'RF-DETR' and self._flag('lordtrilink補漏'):
            add, lt_mask = self._lt_fill(img, items, clip)
            items += add
            if lt_mask is not None:
                rf_mask = lt_mask if rf_mask is None else np.maximum(rf_mask, lt_mask)
        if rf_mask is None and self._flag('新描筆畫') and items:
            rf_mask = self._get_rf().run(img, boxes=[it['xyxy'] for it in items])[1]
        items = clean_overlaps(items, rf_mask)
        texts = [it['xyxy'] for it in items if not it['sfx']]
        sfx = self._dedupe_sfx(texts, [it['xyxy'] for it in items if it['sfx']])
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
        if self._fallback is not None:
            self._fallback.model = None
            self._fallback = None
        return super().unload_model(empty_cache)
