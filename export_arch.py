"""
export_arch.py
放到 BallonsTranslator 根目錄執行：
    python export_arch.py
把核心程式碼複製到 /架構/ 資料夾，檔名用路徑拼接避免重名。
"""

from pathlib import Path
import shutil

# 要匯出的檔案，格式：相對於根目錄的路徑
FILES = [
    # 核心基底
    "modules/base.py",
    "utils/textblock.py",
    "utils/registry.py",
    "utils/config.py",
    "utils/shared.py",
    "utils/imgproc_utils.py",
    "utils/io_utils.py",
    "utils/fontformat.py",
    # "utils/text_layout.py",  <-- 已移除：排版渲染與 OCR 無關

    # OCR
    "modules/ocr/__init__.py",
    "modules/ocr/base.py",
    "modules/ocr/ocr_llm.py",

    # 文字偵測
    "modules/textdetector/__init__.py",
    "modules/textdetector/base.py",
    "modules/textdetector/detector_yolov8.py",

    # 翻譯器基底 (已移除：專注 OCR 開發，暫不需要翻譯模組)

    # modules 入口
    "modules/__init__.py",

    # UI 核心 (保留 Pipeline 相關)
    "ui/module_manager.py",
    "ui/config_proj.py",
    "ui/funcmaps.py",

    # 入口
    "launch.py",
]


def path_to_name(rel_path: str) -> str:
    """把路徑轉成扁平檔名，例如 modules/ocr/base.py → modules_ocr_base.py"""
    p = Path(rel_path)
    parts = list(p.parts)
    return '_'.join(parts)


def main():
    root = Path(__file__).parent
    output_dir = root / '架構'
    output_dir.mkdir(exist_ok=True)

    ok, missing = [], []

    for rel in FILES:
        src = root / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst_name = path_to_name(rel)
        dst = output_dir / dst_name
        shutil.copy2(src, dst)
        ok.append(f"  {rel} → 架構/{dst_name}")

    print(f"\n✓ 匯出成功（{len(ok)} 個）：")
    for line in ok:
        print(line)

    if missing:
        print(f"\n✗ 找不到（{len(missing)} 個）：")
        for f in missing:
            print(f"  {f}")

    total_kb = sum((output_dir / path_to_name(r)).stat().st_size
                   for r in FILES if (root / r).exists()) / 1024
    print(f"\n架構/ 資料夾總大小：{total_kb:.1f} KB")
    print(f"輸出目錄：{output_dir}")


if __name__ == '__main__':
    main()