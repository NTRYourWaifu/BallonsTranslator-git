"""顯存實測（零 API）：量 YOLO+lama 載入/閒置/卸載的顯存佔用，定閒置卸載策略。"""
import os, sys, time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_HERE = os.path.dirname(os.path.abspath(__file__))
BT_ROOT = os.path.dirname(_HERE)
os.chdir(BT_ROOT); sys.path.insert(0, BT_ROOT)


def vram():
    """回 (已配置 MB, 保留 MB)。"""
    import torch
    if not torch.cuda.is_available():
        return None
    return (torch.cuda.memory_allocated() / 1024**2,
            torch.cuda.memory_reserved() / 1024**2)


def gpu_used_mb():
    """整卡實際用量（nvidia-smi，含其他程式）。"""
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).decode()
        return int(out.strip().split('\n')[0])
    except Exception as e:
        return f'err:{e}'


def main():
    import torch
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"}')
    print(f'[基線] torch={vram()}  整卡nvidia-smi={gpu_used_mb()}MB')
    print()

    from headless.core.bootstrap import load_modules
    t0 = time.time()
    mods = load_modules(only_gemini=True, gemini_model='gemini-3.1-flash-lite', verbose=False)
    load_t = time.time() - t0
    print(f'[載入後] 載入耗時 {load_t:.1f}s')
    print(f'  torch={vram()}  整卡={gpu_used_mb()}MB')
    print()

    # 跑一次偵測+inpaint（不打 API），看推理峰值
    import numpy as np, cv2
    from utils.io_utils import imread
    scratch = r'C:\Users\yee\AppData\Local\Temp\claude\f--Vs-Ichaival\ee4775ee-cf44-45a4-bd0c-814faf0115fb\scratchpad'
    testimg = os.path.join(scratch, 'tr_in', '003.webp')
    if os.path.exists(testimg):
        img = imread(testimg)
        mask, blk_list = mods.detector.detect(img)
        print(f'[偵測後] 框={len(blk_list)}  torch={vram()}  整卡={gpu_used_mb()}MB')
        if mask is not None and mask.sum() > 0:
            _ = mods.inpainter.inpaint(img, mask.copy(), blk_list)
            print(f'[inpaint後] torch={vram()}  整卡={gpu_used_mb()}MB')
    print()

    # 閒置 5 秒觀察（模型不動）
    print('[閒置 5s 觀察，模型保持載入]')
    for i in range(5):
        time.sleep(1)
    print(f'  閒置後 torch={vram()}  整卡={gpu_used_mb()}MB')
    print()

    # 卸載 + empty_cache
    print('[卸載模型 + empty_cache]')
    try:
        mods.detector.unload_model(empty_cache=True)
    except Exception as e:
        print(f'  detector unload: {e}')
    try:
        mods.inpainter.unload_model(empty_cache=True)
    except Exception as e:
        print(f'  inpainter unload: {e}')
    del mods
    import gc; gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    print(f'  卸載後 torch={vram()}  整卡={gpu_used_mb()}MB')


if __name__ == '__main__':
    main()
