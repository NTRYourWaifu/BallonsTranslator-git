"""
字數直寫 reading_time.sqlite：翻譯完順手把每頁譯文字數寫進 VideoServ 的秒數表，
省掉 VideoServ 那條 GPU 重算（見計畫 D6 / project_auto_slideshow_text_pacing）。

口徑：譯文字數（覆蓋後頁面上是中文譯文，全庫口徑＝OCR「頁面上的字」）。
    有譯文的 blk 用 translation 字數、沒譯文的 blk（安全過濾失敗頁等）用原文 text 字數。
key：MangaServ uuid。page_names：每頁檔名（與 counts 同序，供刪頁後對齊）。
"""
import os
import json
import time
import sqlite3

# VideoServ 的 reading_time.sqlite（與 video_server.py 同目錄）
_RT_DB = r'F:\Vs\VideoServ\reading_time.sqlite'


def _count_chars(s: str) -> int:
    if not s:
        return 0
    return len(''.join(s.split()))


def per_page_counts_from_translate_json(tj: dict):
    """從 translate.json 算每頁字數 + 檔名（與頁順序同序）。

    回 (counts, page_names)。counts[i] = 該頁所有 blk 的字數加總（優先譯文、退原文）。
    """
    counts = []
    page_names = []
    for name, pg in tj.get('pages', {}).items():
        page_names.append(name)
        total = 0
        for b in pg.get('blocks', []):
            tr = b.get('translation') or ''
            if tr.strip():
                total += _count_chars(tr)
            else:
                src = b.get('text') or ''
                if isinstance(src, list):
                    src = ''.join(src)
                total += _count_chars(src)
        counts.append(total)
    return counts, page_names


def write_reading_time(archive_id: str, tj: dict, db_path: str = None):
    """把 translate.json 的每頁字數寫進 reading_time.sqlite（key=archive_id）。"""
    db_path = db_path or _RT_DB
    counts, page_names = per_page_counts_from_translate_json(tj)
    if not counts:
        return {'written': False, 'reason': 'no pages'}

    conn = sqlite3.connect(db_path)
    try:
        # 確保表存在（對齊 compute_reading_time.init_db）
        conn.execute('''
            CREATE TABLE IF NOT EXISTS reading_time (
                arcid TEXT PRIMARY KEY,
                pagecount INTEGER,
                counts TEXT,
                total INTEGER,
                updated_at REAL,
                page_names TEXT
            )
        ''')
        cols = [r[1] for r in conn.execute('PRAGMA table_info(reading_time)')]
        if 'page_names' not in cols:
            conn.execute('ALTER TABLE reading_time ADD COLUMN page_names TEXT')
        conn.execute(
            'INSERT OR REPLACE INTO reading_time(arcid, pagecount, counts, total, updated_at, page_names) '
            'VALUES (?,?,?,?,?,?)',
            (archive_id, len(counts), json.dumps(counts), int(sum(counts)), time.time(),
             json.dumps(page_names, ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()
    return {'written': True, 'arcid': archive_id, 'pages': len(counts), 'total': int(sum(counts))}
