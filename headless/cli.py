"""
headless 翻譯 CLI（P0 驗證用）。
    ballontrans_pylibs_win\python.exe headless\cli.py <圖片資料夾> [--out 輸出夾] [--max N] [--model gemini-3.1-flash-lite]

預設原地覆蓋（--out 未給 = 寫回來源夾）。P0 測試務必給 --out 到別的夾、別動原檔。
"""
import os
import sys
import argparse

# offscreen 必須最先設定
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

_HERE = os.path.dirname(os.path.abspath(__file__))
BT_ROOT = os.path.dirname(_HERE)
os.chdir(BT_ROOT)
sys.path.insert(0, BT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_dir', help='圖片資料夾')
    ap.add_argument('--out', default=None, help='輸出夾（不給=原地覆蓋）')
    ap.add_argument('--max', type=int, default=None, help='只翻前 N 頁（省 API）')
    ap.add_argument('--model', default='gemini-3.1-flash-lite', help='Gemini 主模型')
    ap.add_argument('--auto-layout', action='store_true', help='開啟 GUI 原碼自動排版（否則只用基本排版）')
    args = ap.parse_args()

    from headless.core.bootstrap import load_modules
    import utils.config as _cfgmod

    print(f'=== headless 翻譯 CLI ===')
    print(f'src={args.src_dir}')
    print(f'out={args.out or "(原地覆蓋)"}')
    print(f'max_pages={args.max}  model={args.model}')
    print()

    mods = load_modules(only_gemini=True, gemini_model=args.model)
    cfg = _cfgmod.pcfg

    st_manager = None
    if args.auto_layout:
        from headless.core.scene_stub import make_stub_manager
        st_manager = make_stub_manager(cfg)

    from headless.core.pipeline import translate_book

    def progress(done, total, pr):
        tag = {'translated': '✓', 'skipped_no_text': '·', 'failed': '✗'}.get(pr.status, '?')
        extra = f'blk={pr.n_blocks} src={pr.char_src} trans={pr.char_trans}'
        if pr.error:
            extra += f'  ERR={pr.error}'
        print(f'  [{done}/{total}] {tag} {pr.name}  {extra}  {pr.seconds:.1f}s')

    print('--- 開始翻譯 ---')
    book = translate_book(args.src_dir, mods, cfg,
                          out_dir=args.out, st_manager=st_manager,
                          progress_cb=progress, max_pages=args.max)
    print()
    print(f'=== 完成：{book.n_translated} 頁翻譯 / {book.n_failed} 頁失敗 / '
          f'共 {len(book.pages)} 頁 / {book.total_seconds:.1f}s ===')
    print(f'輸出：{book.out_dir}')


if __name__ == '__main__':
    main()
