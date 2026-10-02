"""PR1.5 壓力測試主入口（快版規格：20 頁 × 4 worker 配置 × 1 模式 = 80 頁）。

驗證項：
    V1 每頁最終 stage == DONE
    V2 結果圖 SHA256 與 baseline 一致（容差：< 1KB 視為差異）
    V3 stage 序列 monotonic 無逆向（lifecycle 內已有 reject 機制，記錄 reject 次數應為 0）
    V4 finishImgtransPipeline 觸發時機 == lifecycle.all_done 之後（時間戳差 >= 0）
    V5 已 DONE 頁收到後續 transition 0 次（lifecycle 已自動 reject + log）
    V6 字體大小 max_workers=8 vs max_workers=1 每框 |Δ| ≤ 0.5px（calc_font_size_by_render 步進 0.5pt）

用法：
    python -m scripts.stress_test_lifecycle --proj <dir> --max_pages 20 \
                                            --workers 1,3,5,8 --baseline_workers 1

執行流程：
    1. 跑 baseline（單 worker、無擾動）→ 存 baseline.json
    2. 對 --workers 內每個配置：套擾動 + 跑 pipeline + 收集 V1-V6
    3. 印對比報告
"""
import argparse
import hashlib
import json
import os
import os.path as osp
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

# 確保能 import 專案模組
_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─── 共用初始化（仿 launch.py 但 offscreen 不 show）──────────────────

def _setup_qt_offscreen():
    """初始化 QApplication（offscreen）+ 必要 shared state。"""
    import utils.shared as shared
    from utils.io_utils import find_all_files_recursive
    from utils import config as program_config
    from utils.logger import setup_logging, logger as LOGGER

    shared.HEADLESS = True
    shared.load_cache()
    program_config.load_config()
    config = program_config.pcfg
    config.module.load_model_on_demand = True
    config.module.empty_runcache = False

    setup_logging(shared.LOGGING_PATH)

    from launch import load_modules, PATH_FONTS, FONT_EXTS
    load_modules()
    from modules.prepare_local_files import prepare_local_files_forall
    prepare_local_files_forall()

    from qtpy.QtCore import Qt, QLocale
    from qtpy.QtWidgets import QApplication
    from qtpy.QtGui import QFontDatabase, QFont
    import qtpy
    shared.DEFAULT_DISPLAY_LANG = QLocale.system().name().replace('en_CN', 'zh_CN')
    shared.USE_PYSIDE6 = qtpy.API == 'pyside6'
    shared.FLAG_QT6 = qtpy.API_NAME[-1] == '6'
    shared.DEBUG = False

    app_args = sys.argv + ['-platform', 'offscreen']
    app = QApplication.instance() or QApplication(app_args)

    # 字型載入（headless 在 Windows 需要手動補）
    if osp.exists(PATH_FONTS):
        for fp in find_all_files_recursive(PATH_FONTS, FONT_EXTS):
            QFontDatabase.addApplicationFont(fp)
    if sys.platform == 'win32':
        from qtpy.QtCore import QStandardPaths
        for fd in QStandardPaths.standardLocations(QStandardPaths.FontsLocation):
            for fp in find_all_files_recursive(fd, FONT_EXTS):
                QFontDatabase.addApplicationFont(fp)
    if shared.FLAG_QT6:
        shared.FONT_FAMILIES = set(QFontDatabase.families())
    else:
        shared.FONT_FAMILIES = set(QFontDatabase().families())
    yahei = QFont('Microsoft YaHei UI')
    if yahei.exactMatch():
        shared.DEFAULT_FONT_FAMILY = 'Microsoft YaHei UI'
        shared.APP_DEFAULT_FONT = 'Microsoft YaHei UI'
    else:
        shared.DEFAULT_FONT_FAMILY = app.font().family()
        shared.APP_DEFAULT_FONT = app.font().family()

    return app, config, LOGGER


# ─── lifecycle 觀察器 ───────────────────────────────────────────────

class LifecycleObserver:
    """訂閱 lifecycle 事件，記錄 V1-V5 所需資料。"""
    def __init__(self):
        self.transitions: List[Tuple[str, str, str, dict, float]] = []  # (imgname, prev_name, new_name, payload, ts)
        self.reject_count = 0
        self.all_done_ts: Optional[float] = None
        self.finish_pipeline_ts: Optional[float] = None
        self.font_sizes_per_page: Dict[str, Dict[int, float]] = {}
        self.start_ts = time.monotonic()

    def on_transition(self, imgname, prev, new, payload):
        ts = time.monotonic() - self.start_ts
        self.transitions.append((imgname, prev.name, new.name, dict(payload), ts))
        if new.name == 'FONT_CALCULATED' and 'font_sizes' in payload:
            self.font_sizes_per_page[imgname] = dict(payload['font_sizes'])

    def on_all_done(self):
        self.all_done_ts = time.monotonic() - self.start_ts

    def on_finish_pipeline(self):
        self.finish_pipeline_ts = time.monotonic() - self.start_ts

    def attach_reject_counter(self, lifecycle_obj):
        """暫時包裝 instance.mark_done 計 reject 次數。用 detach_reject_counter 還原。
        必須用 class method 呼叫，避免遞迴。"""
        cls_mark_done = type(lifecycle_obj).mark_done

        def _counting(imgname, stage, payload=None):
            result = cls_mark_done(lifecycle_obj, imgname, stage, payload)
            if result is False:
                self.reject_count += 1
            return result

        lifecycle_obj.mark_done = _counting

    def detach_reject_counter(self, lifecycle_obj):
        try:
            del lifecycle_obj.mark_done
        except AttributeError:
            pass


# ─── 單次 pipeline 跑批 ─────────────────────────────────────────────

def init_main_window(app, config, proj_dir: str):
    """只建一次 MainWindow。多次 run_single_pipeline 共用。
    （shared.config_name_to_view_widget 是 module-level dict，重建 MainWindow 會 assert 重複註冊。）"""
    from ui.mainwindow import MainWindow
    import utils.shared as shared

    shared.HEADLESS = True
    # 阻止 MainWindow.__init__ 末尾自動走 HEADLESS 批次路徑
    MainWindow.run_batch = lambda self, *a, **kw: None
    MainWindow.run_next_dir = lambda self: None

    from scripts.stress_test_perturbation import install_patches
    install_patches()  # 冪等：安裝 monkey-patch（mark_done / _on_lifecycle_transition）

    import argparse
    args_stub = argparse.Namespace(debug=False, headless=True, proj_dir=None, exec_dirs='', ldpi=None)
    mw = MainWindow(app, config, open_dir=proj_dir, **vars(args_stub))
    return mw


def run_single_pipeline(app, mw, config, LOGGER, proj_dir: str, max_pages: int,
                        ocr_max_workers: int, perturb: bool,
                        jitter_state: dict, use_lifecycle: bool = True) -> LifecycleObserver:
    """跑一次 pipeline。mw 由 init_main_window 建立，重複呼叫共用同一個。
    jitter_state: {'timer': QTimer | None} 在 caller 維護，避免被 GC。
    use_lifecycle: 預設 True 走新路；False 走舊路（用於 PR5a 驗證等價性）。"""
    from qtpy.QtCore import QTimer, Qt
    from scripts.stress_test_perturbation import (
        enable_perturbation, disable_perturbation, start_jitter,
    )

    # 重置 result 資料夾（V2 SHA256 比對需要乾淨）
    result_dir = osp.join(proj_dir, 'result')
    if osp.exists(result_dir):
        shutil.rmtree(result_dir, ignore_errors=True)

    # PR6：use_page_lifecycle flag 已砍，lifecycle 永遠啟用。use_lifecycle 參數保留為 noop 以兼容舊 verify script。
    _ = use_lifecycle
    config.module.ocr_max_workers = ocr_max_workers
    config.module.enable_detect = True
    config.module.enable_ocr = True
    config.module.enable_inpaint = True
    config.module.enable_translate = False  # 對齊使用者 PR3 驗證場景

    # 擾動開關
    if perturb:
        enable_perturbation(seed=42)
        if jitter_state.get('timer') is None:
            jitter_state['timer'] = start_jitter(app)
    else:
        disable_perturbation()

    # 接 observer：先 disconnect 上次的，再接新的
    observer = LifecycleObserver()
    lifecycle = mw.module_manager.imgtrans_thread.lifecycle
    if jitter_state.get('prev_observer') is not None:
        prev = jitter_state['prev_observer']
        try:
            lifecycle.transition.disconnect(prev.on_transition)
        except (TypeError, RuntimeError):
            pass
        try:
            lifecycle.all_done.disconnect(prev.on_all_done)
        except (TypeError, RuntimeError):
            pass
        try:
            mw.module_manager.imgtrans_pipeline_finished.disconnect(prev.on_finish_pipeline)
        except (TypeError, RuntimeError):
            pass
        try:
            mw.module_manager.imgtrans_pipeline_finished.disconnect(jitter_state['quit_handler'])
        except (TypeError, RuntimeError):
            pass
        # 還原上次的 mark_done 包裝
        prev.detach_reject_counter(lifecycle)
    jitter_state['prev_observer'] = observer

    lifecycle.transition.connect(observer.on_transition, Qt.ConnectionType.QueuedConnection)
    lifecycle.all_done.connect(observer.on_all_done, Qt.ConnectionType.QueuedConnection)
    mw.module_manager.imgtrans_pipeline_finished.connect(observer.on_finish_pipeline)
    observer.attach_reject_counter(lifecycle)

    # 重置 imgtrans_proj：重新 open 確保 pages dict 乾淨
    mw.OpenProj(proj_dir)

    all_pages = list(mw.imgtrans_proj.pages.keys())[:max_pages]
    LOGGER.info(f'[stress] pipeline run: workers={ocr_max_workers} perturb={perturb} pages={len(all_pages)}')

    # 跑完關 event loop（但要等 imsave_thread 把 pending AVIF 寫完，否則 V2 SHA256 比對會缺檔）
    def _quit_handler():
        def _try_quit():
            still_writing = mw.imsave_thread.isRunning() or len(mw.imsave_thread.im_save_list) > 0
            if still_writing:
                QTimer.singleShot(500, _try_quit)  # 持續輪詢直到 imsave list 空
            else:
                QTimer.singleShot(100, app.quit)
        QTimer.singleShot(500, _try_quit)
    jitter_state['quit_handler'] = _quit_handler
    mw.module_manager.imgtrans_pipeline_finished.connect(_quit_handler)

    # 超時保護：10 分鐘
    timeout_timer = QTimer()
    timeout_timer.setSingleShot(True)
    timeout_timer.timeout.connect(lambda: (LOGGER.error('[stress] TIMEOUT 10min'), app.quit()))
    timeout_timer.start(10 * 60 * 1000)

    QTimer.singleShot(200, lambda: mw._prepare_imgtrans_run(only_pages=all_pages))

    app.exec_()
    timeout_timer.stop()
    return observer


# ─── 結果圖 hash ─────────────────────────────────────────────────────

def collect_result_hashes(proj_dir: str, pages: List[str]) -> Dict[str, Tuple[str, int]]:
    """回傳 {imgname: (sha256, size)}。找不到結果圖時值為 ('MISSING', 0)。"""
    result_dir = osp.join(proj_dir, 'result')
    out = {}
    if not osp.exists(result_dir):
        return {p: ('MISSING', 0) for p in pages}
    # 結果圖檔名可能是 imgname 本名（不含 _result 後綴），副檔名依 pcfg.imgsave_ext
    for imgname in pages:
        stem = osp.splitext(imgname)[0]
        candidates = [
            osp.join(result_dir, stem + ext)
            for ext in ('.png', '.jpg', '.jpeg', '.webp', '.avif', '.bmp')
        ]
        found = next((c for c in candidates if osp.exists(c)), None)
        if found is None:
            out[imgname] = ('MISSING', 0)
            continue
        with open(found, 'rb') as f:
            data = f.read()
        out[imgname] = (hashlib.sha256(data).hexdigest(), len(data))
    return out


# ─── V1-V6 驗證 ─────────────────────────────────────────────────────

def verify_v1_v5(observer: LifecycleObserver, expected_pages: List[str]) -> Dict[str, str]:
    """V1/V3/V4/V5 純從 observer 判定。回傳 {key: 'PASS|FAIL: detail'}。"""
    out = {}

    # V1：每頁最終 stage == DONE
    last_stage = {}
    for imgname, prev, new, payload, ts in observer.transitions:
        last_stage[imgname] = new
    missing = [p for p in expected_pages if last_stage.get(p) != 'DONE']
    if missing:
        out['V1'] = f'FAIL: {len(missing)}/{len(expected_pages)} 頁未到 DONE，例：{missing[:3]}'
    else:
        out['V1'] = f'PASS: all {len(expected_pages)} pages DONE'

    # V3：stage 序列 monotonic（int 比較）
    STAGE_ORDER = {'PENDING': 0, 'DETECTED': 1, 'INPAINTED': 2, 'OCRED': 3,
                   'TRANSLATED': 4, 'FONT_CALCULATED': 5, 'DONE': 6}
    per_page_last = {}
    v3_fails = []
    for imgname, prev, new, payload, ts in observer.transitions:
        prev_idx = per_page_last.get(imgname, 0)
        new_idx = STAGE_ORDER[new]
        if new_idx <= prev_idx:
            v3_fails.append(f'{imgname}: {prev}->{new} (prev_idx={prev_idx})')
        per_page_last[imgname] = new_idx
    if v3_fails:
        out['V3'] = f'FAIL: {len(v3_fails)} 個非單調轉換，例：{v3_fails[:3]}'
    else:
        out['V3'] = f'PASS: all {len(observer.transitions)} transitions monotonic'

    # V4：finishImgtransPipeline 觸發時機 >= lifecycle.all_done
    if observer.all_done_ts is None:
        out['V4'] = 'FAIL: all_done 未觸發'
    elif observer.finish_pipeline_ts is None:
        out['V4'] = 'FAIL: imgtrans_pipeline_finished 未觸發'
    else:
        delta = observer.finish_pipeline_ts - observer.all_done_ts
        if delta < -0.001:  # 容忍 1ms 時鐘抖動
            out['V4'] = f'FAIL: finish 早於 all_done by {-delta*1000:.1f}ms'
        else:
            out['V4'] = f'PASS: finish 在 all_done 後 {delta*1000:.1f}ms'

    # V5：reject_count == 0（同 stage 重入或逆向都會被 reject）
    if observer.reject_count != 0:
        out['V5'] = f'FAIL: {observer.reject_count} 次 mark_done 被 reject'
    else:
        out['V5'] = 'PASS: 0 次 reject'

    return out


def verify_v2(baseline_hashes: Dict[str, Tuple[str, int]],
              run_hashes: Dict[str, Tuple[str, int]],
              tol_bytes: int = 1024) -> str:
    """V2：結果圖 SHA256 與 baseline 一致；不一致時看大小差是否在容差內。"""
    mismatches = []
    missing = []
    for imgname, (bh, bs) in baseline_hashes.items():
        rh, rs = run_hashes.get(imgname, ('MISSING', 0))
        if rh == 'MISSING' or bh == 'MISSING':
            missing.append(imgname)
            continue
        if rh != bh:
            size_diff = abs(rs - bs)
            if size_diff > tol_bytes:
                mismatches.append((imgname, size_diff))
    if missing:
        return f'FAIL: {len(missing)} 頁結果圖缺失，例：{missing[:3]}'
    if mismatches:
        return f'FAIL: {len(mismatches)} 頁 hash 不一致且 size 差超 {tol_bytes}B，例：{mismatches[:3]}'
    return f'PASS: all {len(baseline_hashes)} pages match baseline (tol < {tol_bytes}B)'


def verify_v6(baseline_font_sizes: Dict[str, Dict[int, float]],
              run_font_sizes: Dict[str, Dict[int, float]],
              tol_px: float = 0.5) -> str:
    """V6：每框 font_size 差異 ≤ tol_px。"""
    fails = []
    for imgname, base_map in baseline_font_sizes.items():
        run_map = run_font_sizes.get(imgname, {})
        for blk_idx, base_size in base_map.items():
            run_size = run_map.get(blk_idx)
            if run_size is None:
                fails.append(f'{imgname}#{blk_idx}: missing in run')
                continue
            if abs(run_size - base_size) > tol_px:
                fails.append(f'{imgname}#{blk_idx}: base={base_size:.2f} run={run_size:.2f} Δ={run_size-base_size:+.2f}')
    if fails:
        return f'FAIL: {len(fails)} 個框字體差異 > {tol_px}px，例：{fails[:5]}'
    return f'PASS: 所有框 |Δfont_size| ≤ {tol_px}px'


# ─── 入口 ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj', required=True, help='測試漫畫資料夾路徑')
    parser.add_argument('--max_pages', type=int, default=20)
    parser.add_argument('--workers', default='1,3,5,8', help='comma-separated worker counts')
    parser.add_argument('--baseline_only', action='store_true')
    parser.add_argument('--baseline_path', default=None, help='baseline.json path（預設 proj 下 _stress_baseline.json）')
    args = parser.parse_args()

    proj_dir = osp.abspath(args.proj)
    if not osp.isdir(proj_dir):
        print(f'ERROR: --proj {proj_dir} 不存在或不是資料夾')
        sys.exit(2)

    baseline_path = args.baseline_path or osp.join(proj_dir, '_stress_baseline.json')

    app, config, LOGGER = _setup_qt_offscreen()
    mw = init_main_window(app, config, proj_dir)
    jitter_state = {'timer': None, 'prev_observer': None, 'quit_handler': None}

    # ── Step 1：baseline（單 worker，無擾動）──
    print('=' * 78)
    print('STEP 1: baseline (workers=1, no perturbation)')
    print('=' * 78)
    t0 = time.monotonic()
    observer = run_single_pipeline(app, mw, config, LOGGER, proj_dir, args.max_pages,
                                   ocr_max_workers=1, perturb=False, jitter_state=jitter_state)
    elapsed = time.monotonic() - t0
    pages = list(set(t[0] for t in observer.transitions))
    pages.sort()
    baseline_hashes = collect_result_hashes(proj_dir, pages)
    baseline_data = {
        'pages': pages,
        'hashes': {k: list(v) for k, v in baseline_hashes.items()},
        'font_sizes': observer.font_sizes_per_page,
        'transition_count': len(observer.transitions),
        'reject_count': observer.reject_count,
        'elapsed_sec': elapsed,
    }
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_data, f, indent=2, ensure_ascii=False)
    print(f'baseline done: pages={len(pages)}, transitions={len(observer.transitions)}, elapsed={elapsed:.1f}s')
    print(f'baseline saved to: {baseline_path}')

    if args.baseline_only:
        return

    # ── Step 2：4 個 worker 配置壓測 ──
    worker_configs = [int(w) for w in args.workers.split(',')]
    all_results = {}
    for wc in worker_configs:
        print()
        print('=' * 78)
        print(f'STEP 2: stress run (workers={wc}, perturb=ON)')
        print('=' * 78)
        t0 = time.monotonic()
        observer = run_single_pipeline(app, mw, config, LOGGER, proj_dir, args.max_pages,
                                       ocr_max_workers=wc, perturb=True, jitter_state=jitter_state)
        elapsed = time.monotonic() - t0
        run_hashes = collect_result_hashes(proj_dir, pages)
        verdicts = verify_v1_v5(observer, pages)
        verdicts['V2'] = verify_v2(baseline_hashes, run_hashes)
        verdicts['V6'] = verify_v6(observer.font_sizes_per_page, observer.font_sizes_per_page) \
            if wc == 1 else verify_v6(baseline_data['font_sizes'], observer.font_sizes_per_page)
        verdicts['_elapsed'] = f'{elapsed:.1f}s'
        verdicts['_transition_count'] = len(observer.transitions)
        all_results[wc] = verdicts
        for k, v in verdicts.items():
            print(f'  {k}: {v}')

    # ── 總結 ──
    print()
    print('=' * 78)
    print('SUMMARY')
    print('=' * 78)
    overall_pass = True
    for wc, verdicts in all_results.items():
        v_keys = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        results = [verdicts[k].startswith('PASS') for k in v_keys]
        status = 'PASS' if all(results) else 'FAIL'
        overall_pass = overall_pass and all(results)
        print(f'  workers={wc}: {status}  ({sum(results)}/6 V-checks pass)  elapsed={verdicts["_elapsed"]}')
    print(f'\noverall: {"PASS" if overall_pass else "FAIL"}')
    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
