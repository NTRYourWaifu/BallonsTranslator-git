"""P1 測試：zip 進出（dry-run，只翻前 3 頁，不覆蓋原檔）。"""
import os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_HERE = os.path.dirname(os.path.abspath(__file__))
BT_ROOT = os.path.dirname(_HERE)
os.chdir(BT_ROOT); sys.path.insert(0, BT_ROOT)

import argparse, zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip_path')
    ap.add_argument('--max', type=int, default=3)
    args = ap.parse_args()

    from headless.core.bootstrap import load_modules
    from headless.core.bookzip import translate_zip
    import utils.config as _cfgmod

    print(f'=== P1 zip 進出測試（dry-run，翻 {args.max} 頁）===')
    print(f'zip: {args.zip_path}')
    mods = load_modules(only_gemini=True, gemini_model='gemini-3.1-flash-lite')
    cfg = _cfgmod.pcfg

    def prog(done, total, pr):
        tag = {'translated': 'v', 'skipped_no_text': '.', 'failed': 'x'}.get(pr.status, '?')
        print(f'  [{done}/{total}] {tag} {pr.name} blk={pr.n_blocks} trans={pr.char_trans} {pr.seconds:.1f}s'
              + (f'  ERR={pr.error}' if pr.error else ''))

    r = translate_zip(args.zip_path, mods, cfg, progress_cb=prog,
                      max_pages=args.max, dry_run=True)

    print()
    print('=== 結果 ===')
    print(f'  原 id     : {r["orig_id"]}')
    print(f'  寫回 id   : {r["id"]}')
    print(f'  id 一致   : {r["orig_id"] == r["id"]}')
    print(f'  翻譯/失敗 : {r["n_translated"]}/{r["n_failed"]}  共 {r["n_pages"]} 頁')
    print(f'  new_zip   : {r["new_zip"]}')

    # 驗證 new.zip 內部
    nz = r['new_zip']
    z = zipfile.ZipFile(nz)
    names = z.namelist()
    print()
    print('=== new.zip 驗證 ===')
    print(f'  項目數        : {len(names)}')
    print(f'  有 metadata   : {any("metadata.json" in n.lower() for n in names)}')
    print(f'  有 translate  : {any("translate.json" in n.lower() for n in names)}')
    print(f'  副檔名        : {set(n.rsplit(".",1)[-1].lower() for n in names if "." in n)}')
    from mangaserv import zipid
    print(f'  zip 內永久 id : {zipid.read_id(nz)}')
    print(f'  前 5 項       : {names[:5]}')

    # 清理 dry-run work
    import shutil
    if r.get('work_dir'):
        shutil.rmtree(r['work_dir'], ignore_errors=True)
        print(f'\n  已清 work: {r["work_dir"]}')


if __name__ == '__main__':
    main()
