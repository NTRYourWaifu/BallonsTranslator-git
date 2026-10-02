"""
重新排版 CLI —— 給 MangaServ worker 用 subprocess 呼叫。**不跑 Gemini API**。

用譯文版 zip 裡的 translate.json（含框+原文+譯文+字體）對原文圖重跑 inpaint+渲染，
輸出新的譯文版 zip（覆蓋舊譯文版）。用於：字型修好/調排版後免費重繪，譯文不重譯。

用法：
    ballontrans_pylibs_win/python.exe headless/rerender_version.py
        --orig 原文 zip 路徑（讀圖）
        --trans 舊譯文 zip 路徑（讀 translate.json）
        --out 新譯文 zip 輸出路徑
        [--save-ext .webp] [--save-quality 90]

進度/結果逐行 JSON 印 stdout（同 translate_to_version）。
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
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', required=True, help='原文 zip（讀圖 inpaint 用）')
    ap.add_argument('--trans', required=True, help='舊譯文 zip（讀 translate.json）')
    ap.add_argument('--out', required=True, help='新譯文 zip 輸出路徑')
    ap.add_argument('--save-ext', default='.webp')
    ap.add_argument('--save-quality', type=int, default=90)
    args = ap.parse_args()

    try:
        from headless.core.bootstrap import load_modules
        import utils.config as _cfgmod
        # rerender 只用 inpainter，但 load_modules 一次載齊；detector/ocr 不呼叫、不吃 API
        mods = load_modules(only_gemini=True, gemini_model='gemini-3.1-flash-lite', verbose=False)
        cfg = _cfgmod.pcfg
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        _emit({'type': 'error', 'msg': f'載入模組失敗: {type(e).__name__}: {e}'})
        sys.exit(1)

    sys.path.insert(0, r'F:\Vs\VideoServ')
    from mangaserv import zipid
    orig_id = zipid.read_id(args.trans) or zipid.read_id(args.orig)

    work = tempfile.mkdtemp(prefix='bt_rerender_')
    src_in = os.path.join(work, 'orig')   # 原文圖
    tj_in = os.path.join(work, 'trans')   # 舊譯文（取 translate.json）
    out_dir = os.path.join(work, 'out')
    os.makedirs(src_in); os.makedirs(tj_in); os.makedirs(out_dir)

    try:
        with zipfile.ZipFile(args.orig) as z:
            z.extractall(src_in)
        with zipfile.ZipFile(args.trans) as z:
            z.extractall(tj_in)

        tj_path = os.path.join(tj_in, 'translate.json')
        if not os.path.exists(tj_path):
            _emit({'type': 'error', 'msg': '譯文 zip 內無 translate.json，無法重排版（需先翻譯過）'})
            sys.exit(1)

        def stage_cb(prog):
            _emit({'type': 'progress', **prog})

        from headless.core.pipeline import rerender_book
        book = rerender_book(src_in, tj_path, mods, cfg, out_dir=out_dir,
                             save_ext=args.save_ext, save_quality=args.save_quality,
                             stage_progress_cb=stage_cb)

        # 補非圖檔（metadata.json 等，從舊譯文帶）
        from headless.core.bookzip import _copy_nonimage_files, _repack
        _copy_nonimage_files(tj_in, out_dir)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        new_zip = os.path.join(work, 'new.zip')
        _repack(out_dir, new_zip)
        written_id = zipid.write_id(new_zip, orig_id)
        shutil.move(new_zip, args.out)

        _emit({'type': 'done', 'id': written_id, 'orig_id': orig_id,
               'n_translated': book.n_translated, 'n_failed': book.n_failed,
               'n_pages': len(book.pages), 'total_seconds': round(book.total_seconds, 1),
               'out': args.out, 'rerender': True})
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        _emit({'type': 'error', 'msg': f'{type(e).__name__}: {e}'})
        sys.exit(1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
