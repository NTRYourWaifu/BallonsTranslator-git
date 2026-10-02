"""PR1.5 壓力測試：擾動點 monkey-patch 工具。

四個擾動點（盤點 2）：
  1. Worker emit 前         → 涵蓋於擾動點 2（新路下 emit page_ocr_trans_done 已 noop，mark TRANSLATED 前 sleep 等價）
  2. Worker mark_done 前    → 包裝 PageLifecycleManager.mark_done，入口前 sleep(0-50ms)
  3. UI handler 入口        → 包裝 MainWindow._on_lifecycle_transition，入口前 msleep(0-30ms)
  4. 主執行緒抖動           → 背景 QTimer.start(10)，每次跑 5ms 佔用計算

用法：
    from scripts.stress_test_perturbation import enable_perturbation, install_patches, start_jitter
    enable_perturbation(seed=42)
    install_patches()                 # 安裝 monkey-patch
    jitter_timer = start_jitter(app)  # 主執行緒抖動
"""
import random
import time
import threading

_perturb_enabled = False
_perturb_rng: random.Random = None
_patches_installed = False


def enable_perturbation(seed: int = 42):
    """開啟擾動。固定 seed 確保可重現。"""
    global _perturb_enabled, _perturb_rng
    _perturb_enabled = True
    _perturb_rng = random.Random(seed)


def disable_perturbation():
    global _perturb_enabled
    _perturb_enabled = False


def _maybe_sleep_ms(max_ms: float):
    if _perturb_enabled and _perturb_rng is not None:
        time.sleep(_perturb_rng.uniform(0, max_ms / 1000.0))


def install_patches():
    """安裝 monkey-patch（擾動點 2 + 3）。冪等：重複呼叫不會重複包裝。"""
    global _patches_installed
    if _patches_installed:
        return
    _patches_installed = True

    # 擾動點 2：mark_done 入口 sleep
    from ui.page_lifecycle import PageLifecycleManager
    _orig_mark_done = PageLifecycleManager.mark_done

    def _perturbed_mark_done(self, imgname, stage, payload=None):
        _maybe_sleep_ms(50)
        return _orig_mark_done(self, imgname, stage, payload)

    PageLifecycleManager.mark_done = _perturbed_mark_done

    # 擾動點 3：UI handler 入口 sleep
    # 注意：_on_lifecycle_transition 在主執行緒跑，sleep 會阻塞 event loop；
    # 用較小的 max_ms（30ms）避免整個 UI 凍住太久
    from ui.mainwindow import MainWindow
    _orig_handler = MainWindow._on_lifecycle_transition

    def _perturbed_handler(self, imgname, prev, new, payload):
        _maybe_sleep_ms(30)
        return _orig_handler(self, imgname, prev, new, payload)

    MainWindow._on_lifecycle_transition = _perturbed_handler


def start_jitter(app):
    """啟動擾動點 4：背景 QTimer 抖動主執行緒。回傳 timer 物件（呼叫者持有避免被 GC）。"""
    from qtpy.QtCore import QTimer
    timer = QTimer()
    timer.setInterval(10)

    def _jitter():
        if not _perturb_enabled:
            return
        # 佔用 5ms 主執行緒，模擬其他 UI 工作搶 event loop
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.005:
            pass

    timer.timeout.connect(_jitter)
    timer.start()
    return timer
