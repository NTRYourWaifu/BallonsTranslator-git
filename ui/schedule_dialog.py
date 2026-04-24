"""
排程/延遲執行對話框

提供兩種模式：
  - 延遲模式：N 分鐘後自動 RUN
  - 定時模式：指定 HH:MM 後自動 RUN
"""

from __future__ import annotations
from datetime import datetime, timedelta

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QButtonGroup,
    QSpinBox, QTimeEdit, QFrame, QWidget,
)
from qtpy.QtCore import Qt, QTimer, QTime, Signal
from qtpy.QtGui import QFont


class ScheduleDialog(QDialog):
    """排程設定對話框，確認後開始倒數，時間到 emit schedule_triggered。"""

    schedule_triggered = Signal()   # 時間到，呼叫方執行 RUN
    schedule_cancelled = Signal()   # 使用者取消排程

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('排程執行'))
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(300)

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_tick)
        self._remaining_secs: int = 0

        self._build_ui()

    # ── UI 建構 ─────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── 模式選擇 ────────────────────────────────────────────
        mode_row = QHBoxLayout()
        self._rb_delay = QRadioButton(self.tr('延遲模式'))
        self._rb_clock = QRadioButton(self.tr('定時模式'))
        self._rb_delay.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self._rb_delay)
        mode_group.addButton(self._rb_clock)
        mode_row.addWidget(self._rb_delay)
        mode_row.addWidget(self._rb_clock)
        mode_row.addStretch()
        root.addLayout(mode_row)

        # ── 延遲輸入（分鐘） ─────────────────────────────────────
        self._delay_widget = QWidget()
        delay_row = QHBoxLayout(self._delay_widget)
        delay_row.setContentsMargins(0, 0, 0, 0)
        delay_row.addWidget(QLabel(self.tr('延遲')))
        self._spin_minutes = QSpinBox()
        self._spin_minutes.setRange(1, 1440)
        self._spin_minutes.setValue(30)
        self._spin_minutes.setSuffix(self.tr(' 分鐘'))
        self._spin_minutes.setFixedWidth(110)
        delay_row.addWidget(self._spin_minutes)
        delay_row.addStretch()
        root.addWidget(self._delay_widget)

        # ── 定時輸入（HH:MM） ────────────────────────────────────
        self._clock_widget = QWidget()
        clock_row = QHBoxLayout(self._clock_widget)
        clock_row.setContentsMargins(0, 0, 0, 0)
        clock_row.addWidget(QLabel(self.tr('開始時間')))
        self._time_edit = QTimeEdit()
        self._time_edit.setDisplayFormat('HH:mm')
        # 預設加 1 小時
        t = QTime.currentTime().addSecs(3600)
        self._time_edit.setTime(t)
        self._time_edit.setFixedWidth(80)
        clock_row.addWidget(self._time_edit)
        clock_row.addStretch()
        self._clock_widget.setVisible(False)
        root.addWidget(self._clock_widget)

        # ── 模式切換顯示 ─────────────────────────────────────────
        self._rb_delay.toggled.connect(self._on_mode_changed)

        # ── 分隔線 ───────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        # ── 倒數顯示（排程進行中才顯示） ─────────────────────────
        self._countdown_label = QLabel('')
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self._countdown_label.setFont(font)
        self._countdown_label.setAlignment(Qt.AlignCenter)
        self._countdown_label.setVisible(False)
        root.addWidget(self._countdown_label)

        # ── 按鈕列 ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton(self.tr('開始排程'))
        self._cancel_btn = QPushButton(self.tr('取消'))
        self._cancel_btn.setVisible(False)
        self._close_btn = QPushButton(self.tr('關閉'))
        self._start_btn.clicked.connect(self._on_start)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

    # ── 槽 ───────────────────────────────────────────────────────
    def _on_mode_changed(self, delay_checked: bool):
        self._delay_widget.setVisible(delay_checked)
        self._clock_widget.setVisible(not delay_checked)

    def _on_start(self):
        """計算剩餘秒數，啟動倒數。"""
        if self._rb_delay.isChecked():
            secs = self._spin_minutes.value() * 60
        else:
            now = datetime.now()
            target = now.replace(
                hour=self._time_edit.time().hour(),
                minute=self._time_edit.time().minute(),
                second=0, microsecond=0
            )
            if target <= now:
                target += timedelta(days=1)
            secs = int((target - now).total_seconds())

        self._remaining_secs = secs
        self._update_countdown_label()

        # 切換 UI 到「倒數中」狀態
        self._rb_delay.setEnabled(False)
        self._rb_clock.setEnabled(False)
        self._delay_widget.setEnabled(False)
        self._clock_widget.setEnabled(False)
        self._start_btn.setVisible(False)
        self._cancel_btn.setVisible(True)
        self._countdown_label.setVisible(True)

        self._countdown_timer.start()

    def _on_cancel(self):
        """使用者取消排程，還原 UI。"""
        self._countdown_timer.stop()
        self._remaining_secs = 0
        self._countdown_label.setVisible(False)
        self._countdown_label.setText('')
        self._start_btn.setVisible(True)
        self._cancel_btn.setVisible(False)
        self._rb_delay.setEnabled(True)
        self._rb_clock.setEnabled(True)
        self._delay_widget.setEnabled(True)
        self._clock_widget.setEnabled(True)
        self.schedule_cancelled.emit()

    def _on_tick(self):
        self._remaining_secs -= 1
        if self._remaining_secs <= 0:
            self._countdown_timer.stop()
            self._countdown_label.setText(self.tr('啟動中…'))
            self.accept()                   # 關閉對話框
            self.schedule_triggered.emit()  # 觸發 RUN
        else:
            self._update_countdown_label()

    def _update_countdown_label(self):
        h = self._remaining_secs // 3600
        m = (self._remaining_secs % 3600) // 60
        s = self._remaining_secs % 60
        if h > 0:
            text = f'{h:02d}:{m:02d}:{s:02d}'
        else:
            text = f'{m:02d}:{s:02d}'
        self._countdown_label.setText(text)

    # ── 公開方法 ─────────────────────────────────────────────────
    def is_running(self) -> bool:
        """是否正在倒數中。"""
        return self._countdown_timer.isActive()

    def closeEvent(self, event):
        """關閉視窗時若倒數中，詢問是否取消。"""
        if self.is_running():
            self._on_cancel()
        super().closeEvent(event)
