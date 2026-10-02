"""
zip 進出：把 MangaServ 的漫畫 zip 翻譯後重打包，寫回原永久 id、原檔送回收桶、新檔覆蓋。

流程（見計畫 P1）：
  1. read_id(zip) 記住原 uuid（MANGASERV:v1:...）
  2. 解壓 zip → tmp_in（保留 metadata.json）
  3. translate_book(tmp_in → tmp_out)：翻譯，產翻譯後圖 + translate.json
  4. 重打包 tmp_out（圖同格式）+ metadata.json + translate.json → new.zip
  5. zipid.write_id(new.zip, 原uuid) 寫回原 id
  6. 原 zip send2trash → new.zip 覆蓋原路徑

零中間檔：tmp 工作區用完刪；no-text 頁在 translate_book 內已 raw copy 進 tmp_out。
"""
import os
import sys
import shutil
import zipfile
import tempfile
import json

# 引入 MangaServ 的 zipid（寫回永久 id）
_MANGASERV = r'F:\Vs\VideoServ'
if _MANGASERV not in sys.path:
    sys.path.insert(0, _MANGASERV)


def _load_zipid():
    from mangaserv import zipid
    return zipid


def translate_zip(zip_path: str, mods, cfg, *,
                  st_manager=None, progress_cb=None, max_pages=None,
                  keep_original=True, dry_run=False) -> dict:
    """
    翻譯單一 zip 並就地覆蓋（寫回原 uuid）。
    keep_original=True：原檔 send2trash（可救）；False：直接覆蓋。
    dry_run=True：翻譯+打包到 tmp，但不覆蓋原檔（回報 new_zip 路徑供檢查）。
    回 dict：{id, n_translated, n_failed, n_pages, new_zip, translate_json}
    """
    zipid = _load_zipid()
    zip_path = os.path.abspath(zip_path)
    orig_id = zipid.read_id(zip_path)   # 可能 None（沒被寫過 id 的新檔）

    work = tempfile.mkdtemp(prefix='bt_headless_')
    tmp_in = os.path.join(work, 'in')
    tmp_out = os.path.join(work, 'out')
    os.makedirs(tmp_in); os.makedirs(tmp_out)

    try:
        # 1) 解壓（保留所有非圖檔如 metadata.json）
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp_in)

        # 2) 翻譯（translate_book 只處理圖片、no-text 頁 raw copy 到 tmp_out）
        from .pipeline import translate_book
        book = translate_book(tmp_in, mods, cfg,
                              out_dir=tmp_out, st_manager=st_manager,
                              progress_cb=progress_cb, max_pages=max_pages)

        # 3) 把非圖檔（metadata.json 等）從 tmp_in 補進 tmp_out
        _copy_nonimage_files(tmp_in, tmp_out)

        # 4) 重打包 → new.zip
        new_zip = os.path.join(work, 'new.zip')
        _repack(tmp_out, new_zip)

        # 5) 寫回原 uuid（有原 id 就沿用，沒有就發新的）
        written_id = zipid.write_id(new_zip, orig_id)

        # 5.5) 字數直寫 reading_time.sqlite（順手供應秒數，省 VideoServ GPU 重算）
        rt_result = None
        try:
            tj_path = os.path.join(tmp_out, 'translate.json')
            if os.path.exists(tj_path):
                with open(tj_path, encoding='utf-8') as f:
                    tj = json.load(f)
                from .reading_time import write_reading_time
                rt_result = write_reading_time(written_id, tj)
        except Exception as e:
            rt_result = {'written': False, 'error': f'{type(e).__name__}: {e}'}

        result = {
            'id': written_id,
            'orig_id': orig_id,
            'n_translated': book.n_translated,
            'n_failed': book.n_failed,
            'n_pages': len(book.pages),
            'new_zip': new_zip,
            'total_seconds': book.total_seconds,
            'reading_time': rt_result,
            'pages': [(p.name, p.status, p.char_src, p.char_trans) for p in book.pages],
        }

        if dry_run:
            # 保留 work 供檢查（呼叫方負責清）
            result['work_dir'] = work
            result['_no_cleanup'] = True
            return result

        # 6) 覆蓋：原檔 send2trash → new.zip 移到原路徑
        if keep_original:
            from send2trash import send2trash
            send2trash(zip_path)
        else:
            os.remove(zip_path)
        shutil.move(new_zip, zip_path)
        result['new_zip'] = zip_path
        return result

    finally:
        # dry_run 時保留 work
        if not (dry_run):
            shutil.rmtree(work, ignore_errors=True)


def _copy_nonimage_files(src_dir, dst_dir):
    """把 src 內的非圖片檔（metadata.json、translate.json 除外——後者已在 dst）補進 dst。"""
    from utils.io_utils import IMG_EXT
    for name in os.listdir(src_dir):
        sp = os.path.join(src_dir, name)
        if not os.path.isfile(sp):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXT:
            continue  # 圖片已由 translate_book 輸出到 dst
        dp = os.path.join(dst_dir, name)
        if not os.path.exists(dp):
            shutil.copy2(sp, dp)


def _repack(src_dir, zip_path):
    """把 src_dir 內容打包成 zip（頂層檔案，不含子目錄結構；對齊 MangaServ 平鋪頁）。"""
    files = sorted(os.listdir(src_dir))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
        for name in files:
            fp = os.path.join(src_dir, name)
            if os.path.isfile(fp):
                z.write(fp, arcname=name)
