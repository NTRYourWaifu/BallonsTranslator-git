"""
headless 圖片輸出：AVIF 走 ffmpeg SVT-AV1（複用 GUI io_thread 的參數），其餘走 cv2。
獨立於 Qt io_thread（不綁 QThread），供 pipeline 直接呼叫。
"""
import os
import subprocess
import tempfile
import numpy as np
import cv2

# 複用 ui/io_thread.py 的常數（見 bt_project_avif_export）
_FFMPEG = r'F:\Vs\Toolpackage\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe'
_AVIF_PRESET = 8
_SCALE_FILTER = (
    "scale='min(4096,iw)':'min(4096,ih)':force_original_aspect_ratio=decrease,"
    "scale=trunc(iw/2)*2:trunc(ih/2)*2"
)


def save_avif(img_bgr: np.ndarray, dst: str, quality: int = 92):
    """BGR numpy → AVIF（ffmpeg SVT-AV1）。crf 公式與 GUI 一致。"""
    crf = int((100 - quality) * 0.6 + 10)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tmp_path = tf.name
    try:
        cv2.imencode('.png', img_bgr)[1].tofile(tmp_path)
        cmd = [
            _FFMPEG, '-y', '-i', tmp_path,
            '-vf', _SCALE_FILTER,
            '-c:v', 'libsvtav1',
            '-pix_fmt', 'yuv420p',
            '-crf', str(crf),
            '-preset', str(_AVIF_PRESET),
            '-svtav1-params', 'lad=0:scd=0',
            dst,
        ]
        r = subprocess.run(cmd, capture_output=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if r.returncode != 0:
            raise RuntimeError(f'ffmpeg svtav1 failed: {r.stderr.decode(errors="replace")[-300:]}')
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def save_image(img_bgr: np.ndarray, dst: str, quality: int = 92):
    """依 dst 副檔名存圖。.avif→ffmpeg；.webp/.jpg/.png→cv2。"""
    ext = os.path.splitext(dst)[1].lower()
    if ext == '.avif':
        save_avif(img_bgr, dst, quality)
        return
    if ext in ('.jpg', '.jpeg'):
        param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.webp':
        param = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:
        param = None
    if param is not None:
        cv2.imencode(ext, img_bgr, param)[1].tofile(dst)
    else:
        cv2.imencode(ext, img_bgr)[1].tofile(dst)
