"""
單本翻譯成「譯文版」CLI —— 給 MangaServ 背景 worker 用 subprocess 呼叫（跨 venv 隔離）。

與 bookzip.translate_zip 不同：**不覆蓋原檔**，而是輸出獨立的譯文 zip 到指定路徑。
原文版留著、譯文版帶原 uuid（供 MangaServ 對應同一本）。

用法：
    ballontrans_pylibs_win/python.exe headless/translate_to_version.py
        --src  原文 zip 路徑
        --out  譯文 zip 輸出路徑
        [--workers 5] [--max N]

進度與結果用 JSON 逐行印到 stdout（每行一個 JSON 物件），MangaServ 解析：
    {"type":"progress","detect":[d,t],"translate":[d,t],"render":[d,t]}
    {"type":"done","id":"...","n_translated":N,"n_failed":N,"n_pages":N,"reading_time":{...},"out":"..."}
    {"type":"error","msg":"..."}
"""
import os
import sys
import json
import shutil
import zipfile
import tempfile
import argparse

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_HERE = os.path.dirname(os.path.abspath(__file__))
BT_ROOT = os.path.dirname(_HERE)
os.chdir(BT_ROOT)
sys.path.insert(0, BT_ROOT)


def _emit(obj):
    """一行一個 JSON 印到 stdout（供父進程逐行解析）。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='原文 zip 路徑')
    ap.add_argument('--out', required=True, help='譯文 zip 輸出路徑（會建父目錄）')
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--max', type=int, default=None)
    ap.add_argument('--model', default='gemini-3.1-flash-lite')
    args = ap.parse_args()

    try:
        from headless.core.bootstrap import load_modules
        import utils.config as _cfgmod
        mods = load_modules(only_gemini=True, gemini_model=args.model, verbose=False)
        cfg = _cfgmod.pcfg
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        _emit({'type': 'error', 'msg': f'載入模組失敗: {type(e).__name__}: {e}'})
        sys.exit(1)

    # 讀原 uuid
    sys.path.insert(0, r'F:\Vs\VideoServ')
    from mangaserv import zipid
    orig_id = zipid.read_id(args.src)

    work = tempfile.mkdtemp(prefix='bt_ver_')
    tmp_in = os.path.join(work, 'in')
    tmp_out = os.path.join(work, 'out')
    os.makedirs(tmp_in); os.makedirs(tmp_out)

    try:
        # 解壓
        with zipfile.ZipFile(args.src) as z:
            z.extractall(tmp_in)

        # 進度回報
        def stage_cb(prog):
            _emit({'type': 'progress', **prog})

        from headless.core.pipeline import translate_book
        # 譯文版一律輸出 WebP q90（使用者決定：譯文是「看的版本」，WebP 相容性好、cv2 直編免 ffmpeg）
        book = translate_book(tmp_in, mods, cfg, out_dir=tmp_out,
                              stage_progress_cb=stage_cb,
                              save_ext='.webp', save_quality=90,
                              workers=args.workers, max_pages=args.max)

        # 補非圖檔（metadata.json 等）
        from headless.core.bookzip import _copy_nonimage_files, _repack
        _copy_nonimage_files(tmp_in, tmp_out)

        # 打包譯文 zip 到 out
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        new_zip = os.path.join(work, 'new.zip')
        _repack(tmp_out, new_zip)

        # 寫回原 uuid（譯文版跟原文版共用同一 id → MangaServ 對應同一本）
        written_id = zipid.write_id(new_zip, orig_id)

        # 字數直寫（譯文字數口徑）
        rt = None
        try:
            tj_path = os.path.join(tmp_out, 'translate.json')
            if os.path.exists(tj_path) and written_id:
                with open(tj_path, encoding='utf-8') as f:
                    tj = json.load(f)
                from headless.core.reading_time import write_reading_time
                rt = write_reading_time(written_id, tj)
        except Exception as e:
            rt = {'written': False, 'error': str(e)}

        # 移到最終譯文路徑
        shutil.move(new_zip, args.out)

        _emit({'type': 'done', 'id': written_id, 'orig_id': orig_id,
               'n_translated': book.n_translated, 'n_failed': book.n_failed,
               'n_pages': len(book.pages), 'reading_time': rt,
               'total_seconds': round(book.total_seconds, 1), 'out': args.out})
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        _emit({'type': 'error', 'msg': f'{type(e).__name__}: {e}'})
        sys.exit(1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
