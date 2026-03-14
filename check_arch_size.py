"""
check_arch_size.py
放到 BallonsTranslator 根目錄執行：
    python check_arch_size.py
顯示 export_arch.py 清單內每個檔案的大小，幫助決定哪些可以刪減。
"""

from pathlib import Path

FILES = [
    "modules/base.py",
    "utils/textblock.py",
    "utils/registry.py",
    "utils/config.py",
    "utils/shared.py",
    "utils/imgproc_utils.py",
    "utils/io_utils.py",
    "utils/fontformat.py",
    "utils/text_layout.py",
    "modules/ocr/__init__.py",
    "modules/ocr/base.py",
    "modules/ocr/ocr_llm.py",
    "modules/textdetector/__init__.py",
    "modules/textdetector/base.py",
    "modules/textdetector/detector_yolov8.py",
    "modules/translators/__init__.py",
    "modules/translators/base.py",
    "modules/translators/trans_chatgpt.py",
    "modules/inpaint/__init__.py",
    "modules/inpaint/base.py",
    "modules/__init__.py",
    "ui/module_manager.py",
    "ui/mainwindow.py",
    "ui/canvas.py",
    "ui/configpanel.py",
    "ui/config_proj.py",
    "ui/funcmaps.py",
    "ui/scenetext_manager.py",
    "ui/textitem.py",
    "ui/scene_textlayout.py",
    "launch.py",
]

def main():
    root = Path(__file__).parent
    total = 0
    rows = []

    for f in FILES:
        p = root / f
        if p.exists():
            size = p.stat().st_size
            total += size
            rows.append((size, f))
        else:
            rows.append((0, f + "  ← 找不到"))

    rows.sort(reverse=True)

    print(f"\n{'大小':>8}  {'約token':>8}  檔案")
    print("-" * 55)
    for size, f in rows:
        kb = size / 1024
        tok = size // 4
        print(f"{kb:7.1f}KB  {tok//1000:6d}K   {f}")

    print("-" * 55)
    print(f"{total/1024:7.1f}KB  {total//4//1000:6d}K   總計")
    print(f"\n建議目標：總計壓到 400KB / 100K token 以下")

if __name__ == '__main__':
    main()
