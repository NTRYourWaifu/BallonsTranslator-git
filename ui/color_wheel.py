"""
浮動圓形調色盤 (ColorWheelPopup)
- 圓形 HSV 色輪 + 中央飽和度/亮度方塊
- 頂部兩個色塊：主色（文字）/副色（輪廓），點擊切換焦點
- 底部最近使用顏色橫列
- 滑鼠移出面板後游標變滴管，左鍵攝取螢幕顏色
"""

import math
from typing import Callable, List, Optional

from qtpy.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QApplication, QSizePolicy
from qtpy.QtCore import Qt, QPoint, QPointF, QRectF, QSize, Signal, QEvent, QObject
from qtpy.QtGui import (
    QPainter, QColor, QConicalGradient, QRadialGradient, QLinearGradient,
    QBrush, QPen, QImage, QPixmap, QMouseEvent, QPainterPath, QCursor
)

_RECENT_MAX = 10
_WHEEL_OUTER = 110   # 色輪外徑（px）
_WHEEL_WIDTH = 18    # 色輪環寬
_SV_SIZE = _WHEEL_OUTER - _WHEEL_WIDTH * 2 - 8  # 中央方塊邊長


def _global_pos(event) -> QPoint:
    try:
        return event.globalPosition().toPoint()
    except AttributeError:
        return event.globalPos()


# ─────────────────────────────────────────────
# 1. 色輪 Widget
# ─────────────────────────────────────────────
class _ColorWheel(QWidget):
    """純色輪環，回傳點擊位置的 Hue (0-359)"""
    hueChanged = Signal(int)

    def __init__(self, diameter: int, ring_width: int, parent=None):
        super().__init__(parent)
        self._d = diameter
        self._rw = ring_width
        self.setFixedSize(diameter, diameter)
        self._hue = 0
        self._dragging = False

    def setHue(self, h: int):
        self._hue = h % 360
        self.update()

    def hue(self) -> int:
        return self._hue

    def _hue_at(self, pos: QPoint) -> Optional[int]:
        cx, cy = self._d / 2, self._d / 2
        dx, dy = pos.x() - cx, pos.y() - cy
        r = math.hypot(dx, dy)
        outer = self._d / 2
        inner = outer - self._rw
        if inner <= r <= outer:
            angle = math.degrees(math.atan2(-dy, dx)) % 360
            return int(angle)
        return None

    def _in_ring(self, pos: QPoint) -> bool:
        return self._hue_at(pos) is not None

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self._d / 2, self._d / 2
        outer = self._d / 2
        inner = outer - self._rw

        # 色輪環：用 Conical Gradient
        grad = QConicalGradient(cx, cy, 0)
        for i in range(361):
            c = QColor.fromHsvF(i / 360, 1.0, 1.0)
            grad.setColorAt(i / 360, c)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(0, 0, self._d, self._d))

        # 挖空中心
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.drawEllipse(QRectF(cx - inner, cy - inner, inner * 2, inner * 2))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # Hue 游標
        rad = math.radians(self._hue)
        mid_r = (outer + inner) / 2
        hx = cx + mid_r * math.cos(rad)
        hy = cy - mid_r * math.sin(rad)
        p.setPen(QPen(Qt.GlobalColor.white, 2))
        p.setBrush(QColor.fromHsvF(self._hue / 360, 1.0, 1.0))
        p.drawEllipse(QRectF(hx - 7, hy - 7, 14, 14))

    def mousePressEvent(self, event: QMouseEvent):
        h = self._hue_at(event.pos())
        if h is not None:
            self._dragging = True
            self._hue = h
            self.hueChanged.emit(h)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            h = self._hue_at(event.pos())
            if h is not None:
                self._hue = h
                self.hueChanged.emit(h)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False


# ─────────────────────────────────────────────
# 2. SV 方塊 Widget
# ─────────────────────────────────────────────
class _SVSquare(QWidget):
    """飽和度/亮度選擇方塊，右上 = 純色，左 = 白，下 = 黑"""
    svChanged = Signal(float, float)

    def __init__(self, size: int, parent=None):
        super().__init__(parent)
        self._sz = size
        self.setFixedSize(size, size)
        self._hue = 0
        self._s = 1.0
        self._v = 1.0
        self._dragging = False

    def setHue(self, h: int):
        self._hue = h
        self.update()

    def setSV(self, s: float, v: float):
        self._s = max(0.0, min(1.0, s))
        self._v = max(0.0, min(1.0, v))
        self.update()

    def s(self) -> float:
        return self._s

    def v(self) -> float:
        return self._v

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 底色：純 Hue
        p.fillRect(0, 0, self._sz, self._sz, QColor.fromHsvF(self._hue / 360, 1.0, 1.0))

        # 白色漸層（左→右）
        wg = QLinearGradient(0, 0, self._sz, 0)
        wg.setColorAt(0, QColor(255, 255, 255, 255))
        wg.setColorAt(1, QColor(255, 255, 255, 0))
        p.fillRect(0, 0, self._sz, self._sz, QBrush(wg))

        # 黑色漸層（上→下）
        bg = QLinearGradient(0, 0, 0, self._sz)
        bg.setColorAt(0, QColor(0, 0, 0, 0))
        bg.setColorAt(1, QColor(0, 0, 0, 255))
        p.fillRect(0, 0, self._sz, self._sz, QBrush(bg))

        # 游標
        cx = self._s * self._sz
        cy = (1.0 - self._v) * self._sz
        p.setPen(QPen(Qt.GlobalColor.white, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(cx - 6, cy - 6, 12, 12))
        p.setPen(QPen(Qt.GlobalColor.black, 1))
        p.drawEllipse(QRectF(cx - 5, cy - 5, 10, 10))

    def _pick(self, pos: QPoint):
        s = max(0.0, min(1.0, pos.x() / self._sz))
        v = max(0.0, min(1.0, 1.0 - pos.y() / self._sz))
        self._s, self._v = s, v
        self.svChanged.emit(s, v)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        self._dragging = True
        self._pick(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            self._pick(event.pos())

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False


# ─────────────────────────────────────────────
# 3. 主色/副色切換色塊
# ─────────────────────────────────────────────
class _DualColorSwatch(QWidget):
    """
    主色（文字）在左上，副色（輪廓）在右下，點擊切換當前焦點。
    focus=0 → 主色，focus=1 → 副色
    """
    focusChanged = Signal(int)  # 0=主色, 1=副色

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(56, 40)
        self._primary = QColor(0, 0, 0)
        self._secondary = QColor(255, 255, 255)
        self._focus = 0  # 0=primary, 1=secondary

    def setColors(self, primary: QColor, secondary: QColor):
        self._primary = primary
        self._secondary = secondary
        self.update()

    def setPrimary(self, c: QColor):
        self._primary = c
        self.update()

    def setSecondary(self, c: QColor):
        self._secondary = c
        self.update()

    def primary(self) -> QColor:
        return self._primary

    def secondary(self) -> QColor:
        return self._secondary

    def focus(self) -> int:
        return self._focus

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        sw = int(W * 0.65)
        sh = int(H * 0.75)
        ox, oy = W - sw, H - sh

        # 副色（後層）
        pen_sec = QPen(Qt.GlobalColor.white if self._focus == 0 else QColor(30, 147, 229), 2)
        p.setPen(pen_sec)
        p.setBrush(self._secondary)
        p.drawRoundedRect(ox, oy, sw, sh, 3, 3)

        # 主色（前層）
        pen_pri = QPen(QColor(30, 147, 229) if self._focus == 0 else Qt.GlobalColor.white, 2)
        p.setPen(pen_pri)
        p.setBrush(self._primary)
        p.drawRoundedRect(0, 0, sw, sh, 3, 3)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        W, H = self.width(), self.height()
        sw = int(W * 0.65)
        sh = int(H * 0.75)
        ox, oy = W - sw, H - sh
        # 點副色區域
        if event.x() > ox and event.y() > oy:
            self._focus = 1
        else:
            self._focus = 0
        self.update()
        self.focusChanged.emit(self._focus)


# ─────────────────────────────────────────────
# 4. 最近使用顏色列
# ─────────────────────────────────────────────
class _RecentColors(QWidget):
    colorPicked = Signal(QColor)

    def __init__(self, max_count=10, parent=None):
        super().__init__(parent)
        self._colors: List[QColor] = []
        self._max = max_count
        self._swatch_size = 18
        self._spacing = 3
        self.setFixedHeight(self._swatch_size + 4)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def addColor(self, c: QColor):
        # 去重（相同 RGB 不重複加）
        for existing in self._colors:
            if existing.rgb() == c.rgb():
                self._colors.remove(existing)
                break
        self._colors.insert(0, QColor(c))
        if len(self._colors) > self._max:
            self._colors.pop()
        self.update()

    def colors(self) -> List[QColor]:
        return list(self._colors)

    def setColors(self, colors: List[QColor]):
        self._colors = [QColor(c) for c in colors[:self._max]]
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        sz = self._swatch_size
        sp = self._spacing
        x = 2
        for c in self._colors:
            p.setBrush(c)
            p.setPen(QPen(QColor(120, 120, 120), 1))
            p.drawRoundedRect(x, 2, sz, sz, 2, 2)
            x += sz + sp

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        sz = self._swatch_size
        sp = self._spacing
        x = 2
        for c in self._colors:
            if x <= event.x() <= x + sz and 2 <= event.y() <= 2 + sz:
                self.colorPicked.emit(QColor(c))
                return
            x += sz + sp

    def sizeHint(self) -> QSize:
        n = len(self._colors)
        w = 4 + n * (self._swatch_size + self._spacing)
        return QSize(max(w, 40), self._swatch_size + 4)


# ─────────────────────────────────────────────
# 5. 滴管 Event Filter（裝在 gv 上）
# ─────────────────────────────────────────────
_EYEDROPPER_CURSOR = None

def _get_eyedropper_cursor() -> QCursor:
    global _EYEDROPPER_CURSOR
    if _EYEDROPPER_CURSOR is None:
        # 嘗試用系統游標，否則自製
        try:
            _EYEDROPPER_CURSOR = QCursor(Qt.CursorShape.CrossCursor)
        except Exception:
            _EYEDROPPER_CURSOR = QCursor(Qt.CursorShape.CrossCursor)
    return _EYEDROPPER_CURSOR


class _EyedropperFilter(QObject):
    """安裝在 gv.viewport() 上，啟用時截取左鍵點擊的螢幕顏色；右鍵取消"""
    colorPicked = Signal(QColor)
    cancelRequested = Signal()

    def __init__(self, gv, parent=None):
        super().__init__(parent)
        self._gv = gv
        self._active = False

    def setActive(self, active: bool):
        self._active = active
        if active:
            self._gv.viewport().installEventFilter(self)
            QApplication.setOverrideCursor(Qt.CursorShape.CrossCursor)
        else:
            self._gv.viewport().removeEventFilter(self)
            QApplication.restoreOverrideCursor()

    def eventFilter(self, obj, event: QEvent) -> bool:
        if not self._active:
            return False
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                gpos = _global_pos(event)
                screen = QApplication.primaryScreen()
                img = screen.grabWindow(0, gpos.x(), gpos.y(), 1, 1).toImage()
                c = QColor(img.pixel(0, 0))
                self.colorPicked.emit(c)
                return True
            if event.button() == Qt.MouseButton.RightButton:
                self.cancelRequested.emit()
                return True
        return False


# ─────────────────────────────────────────────
# 6. 主浮動調色盤
# ─────────────────────────────────────────────
class ColorWheelPopup(QFrame):
    """
    浮動圓形調色盤，父 widget 為 gv。
    primaryChanged(QColor)  → 文字顏色改變
    secondaryChanged(QColor) → 輪廓顏色改變
    """
    primaryChanged = Signal(QColor)
    secondaryChanged = Signal(QColor)

    def __init__(self, gv, parent=None):
        super().__init__(gv)
        self._gv = gv
        self.setObjectName("ColorWheelPopup")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._recent: List[QColor] = []
        self._eyedropper = _EyedropperFilter(gv, self)
        self._eyedropper.colorPicked.connect(self._onEyedropperPick)
        self._eyedropper.cancelRequested.connect(self._onEyedropperCancel)
        self._eyedropper_mode = False

        vl = QVBoxLayout(self)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)

        # ── 頂部：標題 + 雙色塊 ──
        top_row = QHBoxLayout()
        title = QLabel("調色盤", self)
        title.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        top_row.addWidget(title)
        top_row.addStretch()

        self._swatch = _DualColorSwatch(self)
        self._swatch.setToolTip("左上=文字色，右下=輪廓色\n點擊切換當前編輯目標")
        self._swatch.focusChanged.connect(self._onFocusChanged)
        top_row.addWidget(self._swatch)

        # 滴管按鈕（文字按鈕）
        self._eyedropper_label = QLabel("💧", self)
        self._eyedropper_label.setFixedSize(24, 24)
        self._eyedropper_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._eyedropper_label.setToolTip("滴管：移出面板後點擊攝取顏色")
        self._eyedropper_label.setStyleSheet("font-size:16px; cursor:pointer;")
        self._eyedropper_label.mousePressEvent = lambda e: self._toggleEyedropper()
        top_row.addWidget(self._eyedropper_label)
        vl.addLayout(top_row)

        # ── 色輪 + SV 方塊（疊加在色輪中央）──
        wheel_container = QWidget(self)
        wheel_container.setFixedSize(_WHEEL_OUTER, _WHEEL_OUTER)

        self._wheel = _ColorWheel(_WHEEL_OUTER, _WHEEL_WIDTH, wheel_container)
        self._wheel.hueChanged.connect(self._onHueChanged)

        sv_offset = _WHEEL_WIDTH + 4
        self._sv = _SVSquare(_SV_SIZE, wheel_container)
        self._sv.move(sv_offset, sv_offset)
        self._sv.svChanged.connect(self._onSVChanged)

        vl.addWidget(wheel_container, alignment=Qt.AlignmentFlag.AlignHCenter)

        # ── 十六進位輸入 ──
        hex_row = QHBoxLayout()
        hex_label = QLabel("十六進位", self)
        hex_row.addWidget(hex_label)
        from qtpy.QtWidgets import QLineEdit
        self._hexEdit = QLineEdit(self)
        self._hexEdit.setMaximumWidth(80)
        self._hexEdit.setPlaceholderText("#RRGGBB")
        self._hexEdit.returnPressed.connect(self._onHexInput)
        hex_row.addWidget(self._hexEdit)
        hex_row.addStretch()
        vl.addLayout(hex_row)

        # ── 最近使用顏色 ──
        recent_label = QLabel("最近使用", self)
        vl.addWidget(recent_label)
        self._recent_widget = _RecentColors(_RECENT_MAX, self)
        self._recent_widget.colorPicked.connect(self._onRecentPick)
        vl.addWidget(self._recent_widget)

        self._loading = False
        self.adjustSize()
        self.hide()

        self.setMouseTracking(True)

    # ── 公開 API ──────────────────────────────

    def show_at(self, global_pos: QPoint, primary: QColor, secondary: QColor):
        """在 global_pos 附近顯示，載入初始顏色"""
        self._loading = True
        self._swatch.setColors(primary, secondary)
        self._applyColorToWidgets(primary if self._swatch.focus() == 0 else secondary)
        self._loading = False

        gv = self._gv
        local_pos = gv.mapFromGlobal(global_pos)
        self.adjustSize()
        x = max(0, min(local_pos.x(), gv.width() - self.width()))
        y = max(0, local_pos.y() - self.height() - 4)
        self.move(x, y)
        self.show()
        self.raise_()

    def setPrimary(self, c: QColor):
        self._swatch.setPrimary(c)
        if self._swatch.focus() == 0:
            self._applyColorToWidgets(c)

    def setSecondary(self, c: QColor):
        self._swatch.setSecondary(c)
        if self._swatch.focus() == 1:
            self._applyColorToWidgets(c)

    def recentColors(self) -> List[QColor]:
        return self._recent_widget.colors()

    def setRecentColors(self, colors: List[QColor]):
        self._recent_widget.setColors(colors)

    # ── 內部 ──────────────────────────────────

    def _currentColor(self) -> QColor:
        h = self._wheel.hue()
        s = self._sv.s()
        v = self._sv.v()
        return QColor.fromHsvF(h / 360, s, v)

    def _applyColorToWidgets(self, c: QColor):
        """把顏色同步到色輪和 SV 方塊"""
        h, s, v, _ = c.getHsvF()
        if h < 0:
            h = 0
        self._wheel.setHue(int(h * 360))
        self._sv.setHue(int(h * 360))
        self._sv.setSV(s, v)
        self._hexEdit.setText(c.name().upper())

    def _emitCurrent(self):
        c = self._currentColor()
        self._hexEdit.setText(c.name().upper())
        if self._swatch.focus() == 0:
            self._swatch.setPrimary(c)
            self.primaryChanged.emit(c)
        else:
            self._swatch.setSecondary(c)
            self.secondaryChanged.emit(c)

    def _onHueChanged(self, h: int):
        self._sv.setHue(h)
        self._emitCurrent()

    def _onSVChanged(self, s: float, v: float):
        self._emitCurrent()

    def _onFocusChanged(self, focus: int):
        if focus == 0:
            self._applyColorToWidgets(self._swatch.primary())
        else:
            self._applyColorToWidgets(self._swatch.secondary())

    def _onHexInput(self):
        txt = self._hexEdit.text().strip()
        if not txt.startswith('#'):
            txt = '#' + txt
        c = QColor(txt)
        if c.isValid():
            self._applyColorToWidgets(c)
            if self._swatch.focus() == 0:
                self._swatch.setPrimary(c)
                self.primaryChanged.emit(c)
            else:
                self._swatch.setSecondary(c)
                self.secondaryChanged.emit(c)

    def _onRecentPick(self, c: QColor):
        self._applyColorToWidgets(c)
        if self._swatch.focus() == 0:
            self._swatch.setPrimary(c)
            self.primaryChanged.emit(c)
        else:
            self._swatch.setSecondary(c)
            self.secondaryChanged.emit(c)

    def _commitColorToRecent(self, c: QColor):
        self._recent_widget.addColor(c)

    def _toggleEyedropper(self):
        self._eyedropper_mode = not self._eyedropper_mode
        self._eyedropper.setActive(self._eyedropper_mode)
        if self._eyedropper_mode:
            self._eyedropper_label.setStyleSheet(
                "font-size:16px; cursor:pointer; background: rgba(30,147,229,80); border-radius:3px;"
            )
        else:
            self._eyedropper_label.setStyleSheet("font-size:16px; cursor:pointer;")

    def _onEyedropperPick(self, c: QColor):
        self._eyedropper_mode = False
        self._eyedropper.setActive(False)
        self._eyedropper_label.setStyleSheet("font-size:16px; cursor:pointer;")
        self._applyColorToWidgets(c)
        self._commitColorToRecent(c)
        if self._swatch.focus() == 0:
            self._swatch.setPrimary(c)
            self.primaryChanged.emit(c)
        else:
            self._swatch.setSecondary(c)
            self.secondaryChanged.emit(c)

    def _onEyedropperCancel(self):
        """右鍵取消滴管並關閉整個調色盤"""
        self._eyedropper_mode = False
        self._eyedropper.setActive(False)
        self._eyedropper_label.setStyleSheet("font-size:16px; cursor:pointer;")
        self.hide()

    def leaveEvent(self, event):
        """滑鼠離開面板 → 自動啟動滴管模式"""
        if self.isVisible() and not self._eyedropper_mode:
            self._eyedropper_mode = True
            self._eyedropper.setActive(True)
            self._eyedropper_label.setStyleSheet(
                "font-size:16px; cursor:pointer; background: rgba(30,147,229,80); border-radius:3px;"
            )
        super().leaveEvent(event)

    def enterEvent(self, event):
        """滑鼠回到面板 → 取消滴管"""
        if self._eyedropper_mode:
            self._eyedropper_mode = False
            self._eyedropper.setActive(False)
            self._eyedropper_label.setStyleSheet("font-size:16px; cursor:pointer;")
        super().enterEvent(event)

    def hideEvent(self, event):
        """面板隱藏時確保滴管關閉"""
        if self._eyedropper_mode:
            self._eyedropper_mode = False
            self._eyedropper.setActive(False)
        super().hideEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        # 點擊面板任意處時，把當前顏色加入最近使用
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        c = self._currentColor()
        self._commitColorToRecent(c)
        super().mouseReleaseEvent(event)
