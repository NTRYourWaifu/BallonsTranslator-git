"""整句偵測的 RF-DETR 子程序（detector_region.py 呼叫，不在 BT 的 Python 裡跑）。

Koharu Layout RF-DETR 要 transformers ≥5.1，BT 的嵌入式 Python 裝不進，所以借 BT-headless
重排版新遮罩那個獨立環境（BT-headless/tools/newmask/.venv）常駐跑，模型也用那邊的離線權重。
做法照 BT-headless/tools/newmask/gen_masks.py：頁面縮到長邊 2048 再丟模型；
文字遮罩＝每個 text／擬聲詞實例框跑 comic-text-mask，再跟外擴 1% 的實例遮罩取交集。

stdin 一行一個 JSON：{"img": 圖檔, "mask_out": 遮罩 PNG 輸出路徑}
stdout 一行一個 JSON：ready → 每頁 result／error
"""
import importlib.util, json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
MODELS = Path(sys.argv[1])
THR = {0: 0.25, 1: 0.20, 2: 0.50, 3: 0.50}   # 模型卡的類別門檻：text／擬聲詞／氣泡／分格
CLS = ['text', 'sfx', 'bubble', 'panel']
S = 384


def emit(**kw):
    sys.stdout.write(json.dumps(kw, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def main():
    t0 = time.time()
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    spec = importlib.util.spec_from_file_location('koharu_layout_loader', MODELS / 'rfdetr' / 'load_model.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    model = loader.load_model(MODELS / 'rfdetr' / 'model.safetensors')
    ctm = torch.jit.load(str(MODELS / 'ctm' / 'model.pt'), map_location='cuda').eval()
    emit(type='ready', load_s=round(time.time() - t0, 1))

    def ctm_mask(rgb, box):
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        px, py = int((x2 - x1) * .1), int((y2 - y1) * .1)
        x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
        res = np.zeros((h, w), np.uint8)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return res
        crop = rgb[y1:y2, x1:x2]
        s = S / max(crop.shape[:2])
        nh, nw = max(1, round(crop.shape[0] * s)), max(1, round(crop.shape[1] * s))
        canvas = np.zeros((S, S, 3), np.uint8)
        ox, oy = (S - nw) // 2, (S - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            p = ctm(torch.from_numpy(canvas).permute(2, 0, 1)[None].cuda())[0, 0].float().cpu().numpy()
        p = cv2.resize(p[oy:oy + nh, ox:ox + nw], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        m = (p > 0.5).astype(np.uint8) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) >= 12:
                poly = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2) + (x1, y1)
                if len(poly) >= 3:
                    cv2.fillPoly(res, [poly.astype(np.int32)], 255)
        return res

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            pil = Image.open(req['img']).convert('RGB')
            rgb = np.asarray(pil)
            h, w = rgb.shape[:2]
            sc = min(1., 2048 / max(h, w))
            small = pil.resize((round(w * sc), round(h * sc)), Image.BILINEAR) if sc < 1 else pil
            det = model.predict(small, threshold=0.20, shape=(1152, 1152), include_source_image=False)
            keep = np.asarray([c >= THR[int(k)] for k, c in zip(det.class_id, det.confidence)], bool)
            det = det[keep] if len(det) else det
            if sc < 1 and len(det):
                det.xyxy = det.xyxy / sc
                if det.mask is not None:
                    det.mask = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) > 0
                                         for m in det.mask])
            masks = det.mask if det.mask is not None else np.zeros((0, h, w), bool)
            combo = np.zeros((h, w), np.uint8)
            kd = max(3, int(min(h, w) * .01)) | 1
            inst = []
            for m, k, c, b in zip(masks, det.class_id, det.confidence, det.xyxy):
                k = int(k)
                inst.append({'cls': CLS[k], 'conf': round(float(c), 3), 'xyxy': [round(float(v)) for v in b]})
                if k in (0, 1):
                    region = cv2.dilate(m.astype(np.uint8) * 255, np.ones((kd, kd), np.uint8))
                    combo |= ctm_mask(rgb, b) & region
            cv2.imencode('.png', combo)[1].tofile(req['mask_out'])
            emit(type='result', inst=inst)
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            emit(type='error', msg=f'{type(e).__name__}: {e}'[:300])


if __name__ == '__main__':
    main()
