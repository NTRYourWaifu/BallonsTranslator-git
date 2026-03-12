"""
ui/custom_widget/ocr_stats_bar.py

輕量狀態列，顯示每次 Run 的 OCR Plan 命中次數。
放在 BottomBar 的 spacer 位置（左側區域）。

使用方式（在 mainwindow.py 或 BottomBar 初始化後）：
    self.bottomBar.ocr_stats_bar.connect_ocr(ocr_module.stats_signals)
"""

from qtpy.QtWidgets import QHBoxLayout, QLabel
from qtpy.QtCore import Qt, Slot
from qtpy.QtGui import QFont

from .widget import Widget


# 對應 OcrEventType 的顯示設定
# (event_key, 圖示, 顏色, tooltip)
_PLAN_CONFIG = [
    ('plan_a_ok',  '✓', '#4caf50', 'Plan A（原圖）成功'),   # 綠
    ('plan_a2_ok', '▲', '#ffc107', 'Plan B（黑圖）成功'),   # 黃
    ('slice_ok',   '✂', '#ff9800', 'Plan C（切片）成功'),   # 橘
    ('grok_ok',    '♦', '#e91e96', 'Plan D（Grok）成功'),   # 粉
    ('error',      '✕', '#f44336', '最終失敗 / 放棄'),       # 紅
]


class _StatLabel(QLabel):
    """單一 Plan 的圖示 + 計數 label"""

    def __init__(self, icon: str, color: str, tooltip: str, parent=None):
        super().__init__(parent)
        self._icon = icon
        self._color = color
        self._count = 0
        self.setToolTip(tooltip)
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        font = QFont()
        font.setPointSizeF(8.5)
        self.setFont(font)
        self._refresh()

    def _refresh(self):
        self.setText(f'<span style="color:{self._color}">{self._icon} {self._count}</span>')

    def increment(self):
        self._count += 1
        self._refresh()

    def reset(self):
        self._count = 0
        self._refresh()


class OcrStatsBar(Widget):
    """
    插入 BottomBar 的 OCR 統計條。

    外部呼叫：
        connect_ocr(stats_signals)   # 綁定 OcrStatsSignals
        reset()                      # Run 開始前清零
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: dict[str, _StatLabel] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(10)

        for key, icon, color, tip in _PLAN_CONFIG:
            lbl = _StatLabel(icon, color, tip, self)
            self._labels[key] = lbl
            layout.addWidget(lbl)

        self.setFixedHeight(24)

    # ── 公開 API ──────────────────────────────────────────────

    def connect_ocr(self, stats_signals):
        """綁定 OcrStatsSignals.event signal"""
        stats_signals.event.connect(self._on_event)

    def reset(self):
        """Run 開始時清零所有計數"""
        for lbl in self._labels.values():
            lbl.reset()

    # ── slot ─────────────────────────────────────────────────

    @Slot(str)
    def _on_event(self, event_type: str):
        if event_type in self._labels:
            self._labels[event_type].increment()