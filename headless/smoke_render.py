"""
P0-a 煙霧測試：驗證 offscreen Qt 渲染可行性（全案唯一技術風險）。
在 BT portable python 下跑：
    ballontrans_pylibs_win\python.exe headless\smoke_render.py

驗三件事：
  1. offscreen 平台下 QApplication 能起（TextBlkItem 繼承 QGraphicsTextItem 需 QtWidgets）
  2. QTextDocument 能 offscreen 渲染出非空像素（字真的畫出來了）
  3. QGraphicsScene.render 能把底圖+文字合成成 QImage → numpy
成功印 [OK]，任何一步失敗印 [FAIL] + 原因。
"""
import os
import sys

# 必須在 import 任何 qt 前設定 offscreen 平台
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

BT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BT_ROOT)
sys.path.insert(0, BT_ROOT)

import numpy as np


def step1_app():
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    print(f'[OK] step1 QApplication 起成功，platform={app.platformName()!r}')
    return app


def step2_qtextdocument_render():
    """QTextDocument 獨立 offscreen 渲染（calc_font_size_by_render 已用此法）→ 驗字畫得出來"""
    from qtpy.QtGui import QTextDocument, QImage, QPainter, QColor, QFont
    from qtpy.QtCore import Qt

    doc = QTextDocument()
    doc.setDocumentMargin(0)
    f = QFont()
    f.setPixelSize(48)
    doc.setDefaultFont(f)
    doc.setPlainText('翻譯測試あ')
    sz = doc.size()
    w, h = max(1, int(sz.width())), max(1, int(sz.height()))

    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    doc.drawContents(p)
    p.end()

    # 轉 numpy 檢查有沒有非透明像素（＝字真的畫出來了）
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes() if hasattr(img, 'sizeInBytes') else img.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :]
    nonzero_alpha = int((arr[:, :, 3] > 0).sum())
    if nonzero_alpha < 10:
        raise RuntimeError(f'QTextDocument 渲染出來幾乎全透明（nonzero_alpha={nonzero_alpha}）＝字沒畫出來')
    print(f'[OK] step2 QTextDocument 渲染 {w}x{h}，非透明像素={nonzero_alpha}（字有畫出來）')
    return True


def step3_scene_composite():
    """QGraphicsScene 合成底圖+文字 → QImage（計畫 §四.3 的路徑，備用方案驗證）"""
    from qtpy.QtWidgets import QGraphicsScene, QGraphicsTextItem, QGraphicsPixmapItem
    from qtpy.QtGui import QImage, QPainter, QColor, QPixmap, QFont
    from qtpy.QtCore import QRectF

    W, H = 400, 200
    scene = QGraphicsScene(0, 0, W, H)
    # 底圖：灰底 pixmap（模擬 inpainted）
    base = QPixmap(W, H)
    base.fill(QColor(200, 200, 200))
    scene.addItem(QGraphicsPixmapItem(base))
    # 文字 item
    ti = QGraphicsTextItem('合成測試 abc')
    f = QFont(); f.setPixelSize(40); ti.setFont(f)
    ti.setPos(20, 60)
    scene.addItem(ti)

    out = QImage(W, H, QImage.Format.Format_ARGB32)
    out.fill(QColor(0, 0, 0, 0))
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    scene.render(p, QRectF(0, 0, W, H), QRectF(0, 0, W, H))
    p.end()

    ptr = out.constBits()
    ptr.setsize(out.sizeInBytes() if hasattr(out, 'sizeInBytes') else out.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape(H, out.bytesPerLine() // 4, 4)[:, :W, :]
    # 底圖是灰的所以整片非透明，檢查有沒有「非灰」像素＝文字畫上去了
    gray_mask = (arr[:, :, 0] == 200) & (arr[:, :, 1] == 200) & (arr[:, :, 2] == 200)
    non_gray = int((~gray_mask).sum())
    if non_gray < 10:
        raise RuntimeError(f'scene 合成後沒有非灰像素（non_gray={non_gray}）＝文字沒疊上去')
    print(f'[OK] step3 QGraphicsScene 合成 {W}x{H}，非灰像素={non_gray}（文字疊上底圖成功）')
    return True


def main():
    print('=== offscreen Qt 渲染煙霧測試 ===')
    print(f'python: {sys.version.split()[0]}')
    print(f'QT_QPA_PLATFORM={os.environ.get("QT_QPA_PLATFORM")!r}')
    print()
    try:
        app = step1_app()
        step2_qtextdocument_render()
        step3_scene_composite()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print()
        print(f'[FAIL] {type(e).__name__}: {e}')
        sys.exit(1)
    print()
    print('[ALL OK] offscreen 渲染兩條路徑都通（QTextDocument 直繪 + QGraphicsScene 合成）')


if __name__ == '__main__':
    main()
