from qtpy.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QFontComboBox, QComboBox,
    QLabel, QSizePolicy, QApplication
)
from qtpy.QtCore import Qt, QPoint, QObject, QEvent
from qtpy.QtGui import QDoubleValidator, QFont, QColor, QMouseEvent, QWheelEvent

from utils import shared
from .custom_widget import QFontChecker
from .color_wheel import ColorWheelPopup

_DRAG_THRESHOLD = 4  # px，超過才進入拖曳模式


def _global_pos(event) -> QPoint:
    try:
        return event.globalPosition().toPoint()
    except AttributeError:
        return event.globalPos()


class _LineEditDragFilter(QObject):
    """安裝在 QComboBox.lineEdit() 上，截取上下拖曳來調整 combo 數值"""
    def __init__(self, combo: '_SizeCombo'):
        super().__init__(combo)
        self._combo = combo
        self._press_pos: QPoint = None
        self._start_val: float = None
        self._dragging: bool = False

    def eventFilter(self, obj, event: QEvent) -> bool:
        t = event.type()
        if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = _global_pos(event)
            self._start_val = self._combo._value
            self._dragging = False
            return False  # 不吃掉，讓 lineEdit 正常收到

        if t == QEvent.Type.MouseMove and self._press_pos is not None:
            dy = self._press_pos.y() - _global_pos(event).y()  # 往上 → 正
            if not self._dragging and abs(dy) >= _DRAG_THRESHOLD:
                self._dragging = True
                QApplication.setOverrideCursor(Qt.CursorShape.SizeVerCursor)
            if self._dragging:
                new_val = self._start_val + dy * self._combo.step
                self._combo.setValue(new_val)
                self._combo._emit_value()
                return True  # 拖曳中吃掉事件

        if t == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self._dragging:
                QApplication.restoreOverrideCursor()
                self._dragging = False
                self._press_pos = None
                return True
            self._press_pos = None

        return False


class _SizeCombo(QComboBox):
    """數值 ComboBox，支援滾輪調整與在 lineEdit 上按住上下拖曳快速調整"""
    def __init__(self, items, min_val, max_val, step=1.0, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self._value = 0.0
        self.setEditable(True)
        validator = QDoubleValidator(min_val, max_val, 2, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)
        self.addItems(items)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._drag_filter = _LineEditDragFilter(self)
        self.lineEdit().installEventFilter(self._drag_filter)

    def value(self) -> float:
        try:
            v = float(self.currentText())
            self._value = min(self.max_val, max(self.min_val, v))
            return self._value
        except Exception:
            return self._value

    def setValue(self, v: float):
        v = min(self.max_val, max(self.min_val, v))
        self._value = v
        txt = str(int(v)) if v == int(v) else f'{v:.1f}'
        self.setCurrentText(txt)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.setValue(self._value + self.step)
        elif delta < 0:
            self.setValue(self._value - self.step)
        event.accept()
        self._emit_value()

    def _emit_value(self):
        self.activated.emit(self.currentIndex())


class MiniFontBar(QFrame):
    """
    浮動於 gv 上方的輕量字體格式列（兩行布局）。
    起始點對齊游標左側。所有修改透過 font_format_panel.on_param_changed 走現有 undo 流程。
    """

    def __init__(self, gv, font_format_panel):
        super().__init__(gv)
        self.ffp = font_format_panel
        self._gv = gv

        self.setObjectName("MiniFontBar")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(6, 4, 6, 4)
        vl.setSpacing(3)

        # ── 第一行：字體族、大小、B/I/U、字體顏色色塊 ──────────────
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.setContentsMargins(0, 0, 0, 0)

        self.familyBox = QFontComboBox(self)
        self.familyBox.setObjectName("FontFamilyBox")
        self.familyBox.setEditable(True)
        self.familyBox.setMaximumWidth(160)
        self.familyBox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.familyBox.currentFontChanged.connect(self._onFamilyChanged)
        row1.addWidget(self.familyBox)

        self.sizeBox = _SizeCombo(
            ["10", "12", "14", "16", "18", "20", "24", "28", "32",
             "36", "42", "48", "56", "64", "72", "96"],
            1, 1000, step=1.0, parent=self
        )
        self.sizeBox.setObjectName("FontSizeBox")
        self.sizeBox.setMinimumWidth(52)
        self.sizeBox.setMaximumWidth(64)
        self.sizeBox.activated.connect(self._onSizeActivated)
        self.sizeBox.lineEdit().returnPressed.connect(self._onSizeReturn)
        row1.addWidget(self.sizeBox)

        # B/I/U 使用 QFontChecker，繼承全局樣式表的 SVG 圖示
        self.boldBtn = QFontChecker(self)
        self.boldBtn.setObjectName("FontBoldChecker")
        self.italicBtn = QFontChecker(self)
        self.italicBtn.setObjectName("FontItalicChecker")
        self.underlineBtn = QFontChecker(self)
        self.underlineBtn.setObjectName("FontUnderlineChecker")
        self.boldBtn.clicked.connect(lambda checked: self._emit('bold', checked))
        self.italicBtn.clicked.connect(lambda checked: self._emit('italic', checked))
        self.underlineBtn.clicked.connect(lambda checked: self._emit('underline', checked))
        row1.addWidget(self.boldBtn)
        row1.addWidget(self.italicBtn)
        row1.addWidget(self.underlineBtn)

        row1.addSpacing(4)

        # 顏色色塊（開啟調色盤）
        self._colorSwatch = QLabel(self)
        self._colorSwatch.setFixedSize(20, 20)
        self._colorSwatch.setToolTip("開啟調色盤\n文字色/輪廓色")
        self._colorSwatch.setStyleSheet("background:black; border:1px solid #888;")
        self._colorSwatch.mousePressEvent = lambda e: self._showColorWheel(e)
        row1.addWidget(self._colorSwatch)

        vl.addLayout(row1)

        # ── 第二行：輪廓寬度、行高、橫/豎 ─────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.setContentsMargins(0, 0, 0, 0)

        stroke_label = QLabel("輪廓", self)
        stroke_label.setObjectName("fontStrokeLabel")
        row2.addWidget(stroke_label)

        self.strokeBox = _SizeCombo(
            ["0", "0.1", "0.2", "0.3", "0.5", "0.8", "1.0"],
            0, 10, step=0.1, parent=self
        )
        self.strokeBox.setObjectName("StrokeWidthBox")
        self.strokeBox.setMinimumWidth(48)
        self.strokeBox.setMaximumWidth(60)
        self.strokeBox.activated.connect(self._onStrokeActivated)
        self.strokeBox.lineEdit().returnPressed.connect(self._onStrokeReturn)
        row2.addWidget(self.strokeBox)

        row2.addSpacing(6)

        lsp_label = QLabel("≡", self)
        row2.addWidget(lsp_label)

        self.lineSpBox = _SizeCombo(
            ["1.0", "1.1", "1.2", "1.3", "1.5", "2.0"],
            0, 100, step=0.1, parent=self
        )
        self.lineSpBox.setObjectName("LineSpacingBox")
        self.lineSpBox.setMinimumWidth(48)
        self.lineSpBox.setMaximumWidth(60)
        self.lineSpBox.activated.connect(self._onLineSpActivated)
        self.lineSpBox.lineEdit().returnPressed.connect(self._onLineSpReturn)
        row2.addWidget(self.lineSpBox)

        row2.addSpacing(6)

        self.verticalBtn = QFontChecker(self)
        self.verticalBtn.setObjectName("FontVerticalChecker")
        self.verticalBtn.setToolTip("橫/豎排")
        self.verticalBtn.clicked.connect(lambda checked: self._emit('vertical', checked))
        row2.addWidget(self.verticalBtn)

        vl.addLayout(row2)

        # ── 浮動調色盤（延遲建立，避免 gv 尚未完整初始化）──
        self._colorWheel: ColorWheelPopup = None

        self.adjustSize()
        self.hide()
        self._loading = False

    # ── 公開 API ──────────────────────────────────────────────

    def show_at(self, global_pos: QPoint, blkitem):
        """在 global_pos 上方顯示，起始點對齊游標左側"""
        self._load_format(blkitem)
        gv = self.parent()
        local_pos = gv.mapFromGlobal(global_pos)
        self.adjustSize()
        x = max(0, min(local_pos.x(), gv.width() - self.width()))
        y = max(0, local_pos.y() - self.height() - 4)
        self.move(x, y)
        self.show()
        self.raise_()

    def hideEvent(self, event):
        if self._colorWheel is not None and self._colorWheel.isVisible():
            self._colorWheel.hide()
        super().hideEvent(event)

    def _load_format(self, blkitem):
        self._loading = True
        try:
            self.ffp.set_textblk_item(blkitem)

            if blkitem is None:
                ffmt = self.ffp.global_format
            else:
                ffmt = blkitem.get_fontformat()

            if ffmt.font_family:
                self.familyBox.blockSignals(True)
                self.familyBox.setCurrentText(ffmt.font_family)
                self.familyBox.blockSignals(False)

            self.sizeBox.setValue(round(ffmt.font_size, 1))
            self.boldBtn.setChecked(ffmt.bold)
            self.italicBtn.setChecked(ffmt.italic)
            self.underlineBtn.setChecked(ffmt.underline)
            self.strokeBox.setValue(ffmt.stroke_width)
            self.lineSpBox.setValue(ffmt.line_spacing)
            self.verticalBtn.setChecked(ffmt.vertical)

            # 更新色塊顯示
            fc = QColor(*ffmt.foreground_color())
            self._updateColorSwatch(fc)

            # 若調色盤已開啟，同步顏色
            if self._colorWheel is not None and self._colorWheel.isVisible():
                sc = QColor(*ffmt.stroke_color())
                self._colorWheel.setPrimary(fc)
                self._colorWheel.setSecondary(sc)
        finally:
            self._loading = False

    # ── 調色盤 ────────────────────────────────────────────────

    def _ensureColorWheel(self):
        if self._colorWheel is None:
            self._colorWheel = ColorWheelPopup(self._gv, self._gv)
            self._colorWheel.primaryChanged.connect(self._onPrimaryChanged)
            self._colorWheel.secondaryChanged.connect(self._onSecondaryChanged)

    def _showColorWheel(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._ensureColorWheel()

        # 取得當前格式的兩色
        if self.ffp.textblk_item is not None:
            ffmt = self.ffp.textblk_item.get_fontformat()
        else:
            ffmt = self.ffp.global_format
        fc = QColor(*ffmt.foreground_color())
        sc = QColor(*ffmt.stroke_color())

        # 色塊在 gv 座標系的位置
        gv = self._gv
        swatch_global = self._colorSwatch.mapToGlobal(QPoint(0, 0))
        self._colorWheel.show_at(swatch_global, fc, sc)
        # 載入最近使用顏色（若有）
        # 調色盤已保留自身狀態，不需重載

    def _updateColorSwatch(self, c: QColor):
        r, g, b = c.red(), c.green(), c.blue()
        self._colorSwatch.setStyleSheet(
            f"background: rgb({r},{g},{b}); border:1px solid #888;"
        )

    def _onPrimaryChanged(self, c: QColor):
        self._updateColorSwatch(c)
        self._emit('frgb', (c.red(), c.green(), c.blue()))

    def _onSecondaryChanged(self, c: QColor):
        self._emit('srgb', (c.red(), c.green(), c.blue()))

    # ── 內部槽 ────────────────────────────────────────────────

    def _emit(self, param: str, value):
        if self._loading:
            return
        self.ffp.on_param_changed(param, value)

    def _onFamilyChanged(self, font: QFont):
        if self._loading:
            return
        family = font.family()
        if shared.FONT_FAMILIES is None or family in shared.FONT_FAMILIES:
            self._emit('font_family', family)

    def _onSizeActivated(self):
        self._emit('font_size', self.sizeBox.value())

    def _onSizeReturn(self):
        self._emit('font_size', self.sizeBox.value())

    def _onStrokeActivated(self):
        self._emit('stroke_width', self.strokeBox.value())

    def _onStrokeReturn(self):
        self._emit('stroke_width', self.strokeBox.value())

    def _onLineSpActivated(self):
        self._emit('line_spacing', self.lineSpBox.value())

    def _onLineSpReturn(self):
        self._emit('line_spacing', self.lineSpBox.value())
