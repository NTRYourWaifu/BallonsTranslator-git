"""整句偵測的 RF-DETR 子程序（detector_region.py 呼叫，不在 BT 的 Python 裡跑）。

Koharu Layout RF-DETR 要 transformers ≥5.1，BT 的嵌入式 Python 裝不進，所以借 BT-headless
重排版新遮罩那個獨立環境（BT-headless/tools/newmask/.venv）常駐跑，模型也用那邊的離線權重。
做法照 BT-headless/tools/newmask/gen_masks.py：頁面縮到長邊 2048 再丟模型；
文字遮罩＝每個 text／擬聲詞實例框跑 comic-text-mask，再跟外擴 1% 的實例遮罩取交集。

stdin 一行一個 JSON：{"img": 圖檔（.npy＝BGR 陣列，其他照圖片讀）, "mask_out": 遮罩 PNG 輸出路徑[, "boxes": [[x1,y1,x2,y2], ...]]}
  帶 boxes＝只描筆畫：不跑 RF-DETR，每個框跑 comic-text-mask，跟外擴 1% 短邊的框取交集（給 lordtrilink 用）。
stdout 一行一個 JSON：ready → 每頁 result／error

T-29（11-04）改快、改省顯存，遮罩跟舊寫法逐像素相同（40 頁實測，工具 Ichaival/lab/T29_量測/）：
  - 不走 model.predict：它把 160 個候選的遮罩全放大到 2048 才篩門檻（顯存峰值都在這），這裡先篩，
    只放大留下的 text／擬聲詞，一次最多 CHUNK 個；泡泡、分格只取框。
  - 每個實例只在 comic-text-mask 的裁切範圍裡外擴、取交集、合併，不開整頁大小的圖。
  - 放大回原圖時，彼此讀不到對方像素的實例併成一張一起放大（結果跟一個一個放大一樣）。
  - 呼叫端傳 .npy，省掉大圖 PNG 壓縮解壓。
  一頁的 comic-text-mask 一次送顯卡試過：結果會變、框多的頁吃到 11 GB，不要。
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
CHUNK = 8   # 遮罩放大一次幾個：8 個時顯存峰值約 2.1 GB，速度跟全部一起放大一樣


def emit(**kw):
    sys.stdout.write(json.dumps(kw, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def main():
    t0 = time.time()
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from rfdetr.utilities import box_ops
    spec = importlib.util.spec_from_file_location('koharu_layout_loader', MODELS / 'rfdetr' / 'load_model.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    model = loader.load_model(MODELS / 'rfdetr' / 'model.safetensors')
    ctx = model.model
    net, dev = ctx.model.to(ctx.device).eval(), ctx.device
    num_select = ctx.postprocess.num_select
    ctm = torch.jit.load(str(MODELS / 'ctm' / 'model.pt'), map_location='cuda').eval()
    emit(type='ready', load_s=round(time.time() - t0, 1))

    def detect(small):
        """等於 model.predict(small, threshold=0.20, shape=(1152, 1152))＋類別門檻，但只放大 text／擬聲詞的遮罩。
        回傳 xyxy（縮圖座標）、conf、cls、text 遮罩（縮圖大小 bool，跟 cls 是 0／1 的實例同順序）。"""
        img = TF.to_tensor(small)
        h, w = img.shape[1:]
        x = TF.normalize(TF.resize(img.to(dev), [1152, 1152]), model.means, model.stds)
        with torch.no_grad():
            pred = net(torch.stack([x]))
            if isinstance(pred, tuple):
                pred = {'pred_logits': pred[1], 'pred_boxes': pred[0], 'pred_masks': pred[2]}
            logits, out_masks = pred['pred_logits'], pred['pred_masks']
            topv, topi = torch.topk(logits.sigmoid().view(1, -1), num_select, dim=1)
            scores, qidx, labels = topv[0], topi[0] // logits.shape[2], topi[0] % logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(pred['pred_boxes'])[0][qidx]
            img_h, img_w = torch.tensor([(h, w)], device=dev).unbind(1)
            boxes = boxes * torch.stack([img_w, img_h, img_w, img_h], dim=1)[0]
            keep = scores > 0.20
            xyxy = boxes[keep].float().cpu().numpy()
            conf = scores[keep].float().cpu().numpy()
            cls = labels[keep].cpu().numpy()
            sel = np.asarray([c >= THR[int(k)] for k, c in zip(cls, conf)], bool)
            xyxy, conf, cls = xyxy[sel], conf[sel], cls[sel]
            q = qidx[keep][torch.from_numpy(sel).to(dev)]
            tq = q[torch.from_numpy(np.isin(cls, (0, 1))).to(dev)]
            tmasks = np.empty((len(tq), int(h), int(w)), bool)
            for i in range(0, len(tq), CHUNK):
                m = torch.nn.functional.interpolate(out_masks[0][tq[i:i + CHUNK]].unsqueeze(1), size=(int(h), int(w)),
                                                    mode='bilinear', align_corners=False)
                tmasks[i:i + CHUNK] = (m > 0.0).squeeze(1).cpu().numpy()
        return xyxy, conf, cls, tmasks

    def ctm_window(box, h, w):
        """comic-text-mask 的裁切範圍：框四邊各放大 10%。"""
        x1, y1, x2, y2 = (int(v) for v in box)
        px, py = int((x2 - x1) * .1), int((y2 - y1) * .1)
        return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)

    def ctm_one(rgb, win):
        """comic-text-mask 一個框（置中補黑縮進 384、門檻 0.5、外輪廓填滿），回傳裁切範圍大小的遮罩。"""
        x1, y1, x2, y2 = win
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
        res = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) >= 12:
                poly = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
                if len(poly) >= 3:
                    cv2.fillPoly(res, [poly.astype(np.int32)], 255)
        return res

    def ctm_masks(rgb, boxes):
        """每個框的 (裁切範圍, 筆畫遮罩)；裁切範圍小於 4 像素的遮罩是 None。"""
        h, w = rgb.shape[:2]
        wins = [ctm_window(b, h, w) for b in boxes]
        return wins, [ctm_one(rgb, wn) if wn[2] - wn[0] >= 4 and wn[3] - wn[1] >= 4 else None for wn in wins]

    def src_range(a, b, n_src, n_dst):
        """放大後的第 a～b-1 列（或欄）用 INTER_LINEAR 會讀到縮圖的哪幾列，多留 2 格餘裕。"""
        sc = n_src / n_dst
        return max(0, int(np.floor((a + .5) * sc - .5)) - 2), min(n_src, int(np.floor((b - .5) * sc - .5)) + 3)

    def full_res_windows(tmasks, need, h, w):
        """每個實例 need[i] 範圍內、放大回原圖大小的遮罩（等於整張放大再裁切）。
        彼此讀不到對方像素的實例併成一張一起放大，一頁通常只放大一兩次。"""
        hs, ws = tmasks.shape[1:]
        rng = [(*src_range(Y1, Y2, hs, h), *src_range(X1, X2, ws, w)) for X1, Y1, X2, Y2 in need]
        out = [None] * len(tmasks)
        todo = list(range(len(tmasks)))
        while todo:
            grp = [todo[0]]
            for j in todo[1:]:
                c1, c2, d1, d2 = rng[j]
                if not any(tmasks[j][rng[i][0]:rng[i][1], rng[i][2]:rng[i][3]].any() or tmasks[i][c1:c2, d1:d2].any()
                           for i in grp):
                    grp.append(j)
            src = tmasks[grp[0]] if len(grp) == 1 else np.logical_or.reduce(tmasks[grp])
            big = cv2.resize(src.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
            for i in grp:
                X1, Y1, X2, Y2 = need[i]
                out[i] = big[Y1:Y2, X1:X2] > 0
            todo = [i for i in todo if i not in grp]
        return out

    def load(path):
        if path.endswith('.npy'):
            rgb = np.ascontiguousarray(np.load(path)[..., ::-1])
            return Image.fromarray(rgb), rgb
        pil = Image.open(path).convert('RGB')
        return pil, np.asarray(pil)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            pil, rgb = load(req['img'])
            h, w = rgb.shape[:2]
            combo = np.zeros((h, w), np.uint8)
            if req.get('boxes') is not None:
                kd = max(3, int(min(h, w) * .01))
                wins, cms = ctm_masks(rgb, req['boxes'])
                for b, (x1, y1, x2, y2), cm in zip(req['boxes'], wins, cms):
                    if cm is None:
                        continue
                    bx1, by1, bx2, by2 = (int(v) for v in b)
                    ry1, ry2 = max(0, by1 - kd, y1), min(by2 + kd, y2)   # 外擴的框跟裁切範圍的交集
                    rx1, rx2 = max(0, bx1 - kd, x1), min(bx2 + kd, x2)
                    if ry2 > ry1 and rx2 > rx1:
                        sub = combo[ry1:ry2, rx1:rx2]
                        sub |= cm[ry1 - y1:ry2 - y1, rx1 - x1:rx2 - x1]
                cv2.imencode('.png', combo)[1].tofile(req['mask_out'])
                emit(type='result', inst=[])
                continue
            sc = min(1., 2048 / max(h, w))
            small = pil.resize((round(w * sc), round(h * sc)), Image.BILINEAR) if sc < 1 else pil
            xyxy, conf, cls, tmasks = detect(small)
            if sc < 1 and len(xyxy):
                xyxy = xyxy / sc
            inst = [{'cls': CLS[int(k)], 'conf': round(float(c), 3), 'xyxy': [round(float(v)) for v in b]}
                    for b, k, c in zip(xyxy, cls, conf)]
            kd = max(3, int(min(h, w) * .01)) | 1
            r = kd // 2
            wins, cms = ctm_masks(rgb, [b for b, k in zip(xyxy, cls) if int(k) in (0, 1)])
            # 外擴只會讀到半徑 r 內的像素，所以實例遮罩只要裁切範圍再往外 r 那一塊
            need = [(max(0, x1 - r), max(0, y1 - r), min(w, x2 + r), min(h, y2 + r)) for x1, y1, x2, y2 in wins]
            if sc < 1:
                fulls = full_res_windows(tmasks, need, h, w)
            else:
                fulls = [tmasks[i][Y1:Y2, X1:X2] for i, (X1, Y1, X2, Y2) in enumerate(need)]
            ker = np.ones((kd, kd), np.uint8)
            for (x1, y1, x2, y2), (X1, Y1, X2, Y2), cm, fm in zip(wins, need, cms, fulls):
                if cm is None:
                    continue
                region = cv2.dilate(fm.astype(np.uint8) * 255, ker)[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1]
                sub = combo[y1:y2, x1:x2]
                sub |= cm & region
            cv2.imencode('.png', combo)[1].tofile(req['mask_out'])
            emit(type='result', inst=inst)
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            emit(type='error', msg=f'{type(e).__name__}: {e}'[:300])


if __name__ == '__main__':
    main()
