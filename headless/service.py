"""
常駐翻譯服務（P3）：poll MangaServ 待翻譯清單 → 逐本翻譯覆蓋 → rescan → 移出清單。

流程（見計畫 P3）：
  每 POLL_INTERVAL 秒 GET /api/untranslated
  → 有單 → FIFO 佇列一次一本（GPU 序列化）
  → GET /api/admin/archive_path/{id} 拿 zip path
  → bookzip.translate_zip 就地翻譯覆蓋（寫回原 uuid、原檔 send2trash、字數直寫）
  → POST /api/admin/rescan/{id}（content_sig 變 → 縮圖重生、tag/進度保留）
  → DELETE /api/archives/{id}/untranslated 移出清單
  失敗：記 failed 名單、原檔不動、不自動重試

狀態寫 headless/status.json 供儀表板讀。
啟動：ballontrans_pylibs_win\python.exe headless\service.py [--once] [--max N]
"""
import os
import sys
import json
import time
import argparse
import traceback

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_HERE = os.path.dirname(os.path.abspath(__file__))
BT_ROOT = os.path.dirname(_HERE)
os.chdir(BT_ROOT)
sys.path.insert(0, BT_ROOT)

import urllib.request
import urllib.error

MANGASERV = 'http://127.0.0.1:48381'
POLL_INTERVAL = 60
STATUS_FILE = os.path.join(_HERE, 'status.json')

# 失敗名單（避免對同一本燒錢重試）
_failed_ids = set()


def _http_json(method, url, timeout=30):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode('utf-8')
        return json.loads(body) if body else None


def _write_status(state: dict):
    state['updated_at'] = time.time()
    try:
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _fetch_untranslated():
    try:
        rows = _http_json('GET', f'{MANGASERV}/api/untranslated', timeout=15)
        return rows or []
    except Exception as e:
        print(f'[service] 取待翻譯清單失敗：{e}')
        return []


def _fetch_path(archive_id):
    try:
        r = _http_json('GET', f'{MANGASERV}/api/admin/archive_path/{archive_id}', timeout=15)
        return r.get('path') if r else None
    except Exception as e:
        print(f'[service] 取 path 失敗 {archive_id}：{e}')
        return None


def process_one(archive_id, title, mods, cfg, *, max_pages=None):
    """翻譯覆蓋單本 → rescan → 移出清單。回 result dict。"""
    path = _fetch_path(archive_id)
    if not path or not os.path.exists(path):
        print(f'[service] 找不到檔案：{archive_id} path={path}')
        _failed_ids.add(archive_id)
        return {'ok': False, 'reason': 'path not found'}

    print(f'[service] ▶ 開始翻譯 {title or archive_id}  ({os.path.basename(path)})')
    from headless.core.bookzip import translate_zip

    def prog(done, total, pr):
        tag = {'translated': 'v', 'skipped_no_text': '.', 'failed': 'x'}.get(pr.status, '?')
        _write_status({'phase': 'translating', 'current_id': archive_id,
                       'current_title': title, 'page': f'{done}/{total}',
                       'last_page': f'{tag} {pr.name}'})

    try:
        r = translate_zip(path, mods, cfg, progress_cb=prog,
                          max_pages=max_pages, keep_original=True)
    except Exception as e:
        print(f'[service] ✗ 翻譯失敗 {archive_id}：{e}')
        traceback.print_exc()
        _failed_ids.add(archive_id)
        return {'ok': False, 'reason': str(e)}

    print(f'[service] ✓ 翻譯完成 {r["n_translated"]}/{r["n_pages"]} 頁，id={r["id"]}，'
          f'字數={r.get("reading_time")}')

    # rescan（觸發縮圖重生、content_sig 更新）
    try:
        rescan = _http_json('POST', f'{MANGASERV}/api/admin/rescan/{r["id"]}', timeout=30)
        print(f'[service] rescan：{rescan}')
    except Exception as e:
        print(f'[service] rescan 失敗（翻譯已覆蓋，可手動 rescan）：{e}')

    # 移出待翻譯清單
    try:
        _http_json('DELETE', f'{MANGASERV}/api/archives/{r["id"]}/untranslated', timeout=15)
    except Exception as e:
        print(f'[service] 移出清單失敗：{e}')

    return {'ok': True, 'result': r}


def run_loop(once=False, max_pages=None):
    from headless.core.bootstrap import load_modules
    import utils.config as _cfgmod

    print('[service] 載入模組...')
    mods = load_modules(only_gemini=True, gemini_model='gemini-3.1-flash-lite')
    cfg = _cfgmod.pcfg
    print('[service] 就緒，開始輪詢')

    done_count = 0
    while True:
        rows = _fetch_untranslated()
        # 過濾失敗名單
        queue = [r for r in rows if r['id'] not in _failed_ids]
        _write_status({'phase': 'idle', 'queue_len': len(queue),
                       'done_count': done_count, 'failed_count': len(_failed_ids)})

        if queue:
            item = queue[0]  # FIFO：清單依 title 排序，取第一個
            res = process_one(item['id'], item.get('title', ''), mods, cfg, max_pages=max_pages)
            if res.get('ok'):
                done_count += 1
            # 立即 continue 抓下一本（不等 poll interval）
            continue

        if once:
            print('[service] --once 模式，佇列空，結束')
            _write_status({'phase': 'done', 'done_count': done_count,
                           'failed_count': len(_failed_ids)})
            break

        time.sleep(POLL_INTERVAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true', help='佇列清空就結束（不常駐）')
    ap.add_argument('--max', type=int, default=None, help='每本只翻前 N 頁（測試省 API）')
    args = ap.parse_args()
    run_loop(once=args.once, max_pages=args.max)


if __name__ == '__main__':
    main()
