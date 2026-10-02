"""每頁生命週期狀態機（PageLifecycle FSM）。

設計依據：docs/page_lifecycle_design.md

執行緒契約（核心）：
    - mark_done 可由任何 worker thread 呼叫，內部用 RLock 保護 _state 寫入
    - transition signal 必須以 Qt.QueuedConnection 連接（在 connect 端設定），
      跨執行緒一律 queued，接收方在自己 thread 的 event loop 處理
    - worker 只 mark 到 TRANSLATED 就回去抓下一頁，不等 font_calc；
      FONT_CALCULATED 由主執行緒 runner（訂閱 TRANSLATED transition）跑完渲染後自己 mark
    - DONE 由主執行緒 runner（訂閱 FONT_CALCULATED transition）跑完 UI 刷新 + save 後 mark

Stage owner：
    PENDING            initialize() 即設定
    DETECTED           detect worker thread
    INPAINTED          inpaint worker thread
    OCRED              OCR worker thread
    TRANSLATED         OCR worker thread（與 OCR 同 worker）
    FONT_CALCULATED    主執行緒 runner
    DONE               主執行緒 runner
"""
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Set, Tuple

from qtpy.QtCore import QObject, Signal

from utils.logger import logger as LOGGER


class PageStage(IntEnum):
    """單調遞增；mark_done 用 int 比較拒絕逆向轉移。"""
    PENDING         = 0
    DETECTED        = 1
    INPAINTED       = 2
    OCRED           = 3
    TRANSLATED      = 4
    FONT_CALCULATED = 5
    DONE            = 6


@dataclass
class PageState:
    imgname: str
    expected_stages: Set[PageStage]
    current_stage: PageStage = PageStage.PENDING
    history: List[Tuple[PageStage, float, dict]] = field(default_factory=list)


class PageLifecycleManager(QObject):
    """每頁生命週期狀態機。一次跑（runImgtransPipeline）建一個新實例，跑完丟棄。

    使用模式：
        lifecycle = PageLifecycleManager(parent)
        lifecycle.initialize(['p1.jpg', 'p2.jpg'], expected_stages={DETECTED, OCRED, ...})
        lifecycle.transition.connect(handler, Qt.QueuedConnection)  # 跨執行緒必須 Queued
        lifecycle.all_done.connect(on_finished, Qt.QueuedConnection)

        # worker thread：
        lifecycle.mark_done('p1.jpg', PageStage.OCRED, payload={'blk_count_with_text': 3})
    """

    # (imgname, prev_stage, new_stage, payload)
    transition = Signal(str, object, object, dict)
    # 所有頁都到 DONE 時 emit 一次
    all_done = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.RLock()
        self._state: Dict[str, PageState] = {}
        self._global_expected_stages: Set[PageStage] = set()
        self._all_done_emitted = False

    # ── 主執行緒呼叫 ────────────────────────────────────────────

    def initialize(self, pages: List[str], expected_stages: Set[PageStage]) -> None:
        """跑前由主執行緒呼叫一次。pages 應為本批要處理的頁清單；
        殘留 signal 對應的 imgname 不在這個 list 內，mark_done 自然 reject。

        expected_stages 是「全局」預期：依 cfg_module.enable_* 旗組合決定，
        所有頁共用。空頁的 noop 由各 runner 自行判斷（見 design d 節）。
        """
        with self._lock:
            self._state.clear()
            self._global_expected_stages = set(expected_stages)
            self._all_done_emitted = False
            for imgname in pages:
                self._state[imgname] = PageState(
                    imgname=imgname,
                    expected_stages=set(expected_stages),
                )

    # ── 任何執行緒呼叫（thread-safe）─────────────────────────────

    def mark_done(self, imgname: str, stage: PageStage, payload: dict = None) -> bool:
        """記錄某頁完成某 stage。回傳 True 表示成功推進、有 emit transition。

        會被 reject 的情況（回傳 False）：
            - imgname 不在 _state 中（殘留 signal / 不屬於本批）
            - stage 不在該頁的 expected_stages 中
            - 同 stage 重入（new == current）
            - 逆向（new < current）

        全部 reject 都是「安靜丟棄」（log 等級從 debug 到 warning 不等），
        不拋例外，避免 worker 端要做額外守衛。
        """
        payload = payload or {}
        emit_args = None  # (imgname, prev, stage, payload)
        emit_all_done = False

        with self._lock:
            state = self._state.get(imgname)
            if state is None:
                LOGGER.debug(f'[lifecycle] reject mark_done: imgname={imgname!r} not in batch')
                return False
            if stage not in state.expected_stages:
                LOGGER.debug(f'[lifecycle] reject mark_done: {imgname!r} stage={stage.name} not expected')
                return False
            prev = state.current_stage
            if int(stage) == int(prev):
                LOGGER.warning(f'[lifecycle] reject mark_done: {imgname!r} reentrant stage={stage.name}')
                return False
            if int(stage) < int(prev):
                LOGGER.error(f'[lifecycle] reject mark_done: {imgname!r} reverse {prev.name} -> {stage.name}')
                return False

            state.current_stage = stage
            state.history.append((stage, time.monotonic(), dict(payload)))
            emit_args = (imgname, prev, stage, dict(payload))

            if not self._all_done_emitted and self._check_all_done_locked():
                self._all_done_emitted = True
                emit_all_done = True

        # 鎖外 emit，避免接收方在 DirectConnection 下倒灌進來造成 deadlock
        # （正常情況應全用 QueuedConnection，但保險起見仍釋鎖後 emit）
        self.transition.emit(*emit_args)
        if emit_all_done:
            self.all_done.emit()
        return True

    def expects_stage(self, stage: PageStage) -> bool:
        """全局查詢：本次 lifecycle 是否預期跑此 stage。
        worker 端用來判斷要不要起對應的處理（取代散落的 if cfg_module.enable_*）。
        """
        with self._lock:
            return stage in self._global_expected_stages

    def is_all_done(self) -> bool:
        with self._lock:
            return self._check_all_done_locked()

    def current_stage_of(self, imgname: str) -> PageStage:
        """查詢某頁目前 stage；imgname 不存在時回 PENDING（視為未處理）。"""
        with self._lock:
            state = self._state.get(imgname)
            return state.current_stage if state else PageStage.PENDING

    def __contains__(self, imgname: str) -> bool:
        with self._lock:
            return imgname in self._state

    def __len__(self) -> int:
        with self._lock:
            return len(self._state)

    def snapshot(self) -> Dict[str, PageStage]:
        """偵錯用：拿到當前所有頁的 stage 快照（不含 history）。"""
        with self._lock:
            return {name: s.current_stage for name, s in self._state.items()}

    # ── 內部 helper ────────────────────────────────────────────

    def _check_all_done_locked(self) -> bool:
        """呼叫前必須已持 _lock。"""
        if not self._state:
            return False
        return all(s.current_stage == PageStage.DONE for s in self._state.values())
