"""
ui/custom_widget/ocr_stats_bar.py

輕量狀態列，顯示每次 Run 的 OCR Plan 命中次數。
放在 BottomBar 的 spacer 位置（左側區域）。

圖示風格參考 VS Code 狀態列：
  ✓  Plan A 成功（原圖）    綠
  △  Plan B 成功（黑圖）    黃
  ✂  Plan C 成功（切片）    橘
  ◆  Plan D 成功（Grok）    粉
  ✕  Error 異常             紅
"""

from qtpy.QtWidgets import QHBoxLayout, QWidget
from qtpy.QtCore import Qt, Slot, QPoint, QPointF
from qtpy.QtGui import (
    QFont, QPainter, QColor, QBrush, QPen,
    QPainterPath, QPolygonF,
)

from .widget import Widget


# ── 圖示繪製函式 ─────────────────────────────────────────────
# 每個函式在 (cx, cy) 為中心、半徑 r 的範圍內繪製圖示

def _draw_check(p: QPainter, cx: float, cy: float, r: float, color: QColor):
    """✓ 打勾"""
    pen = QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(cx - r * 0.65, cy + r * 0.05)
    path.lineTo(cx - r * 0.1, cy + r * 0.6)
    path.lineTo(cx + r * 0.7, cy - r * 0.55)
    p.drawPath(path)


def _draw_triangle(p: QPainter, cx: float, cy: float, r: float, color: QColor):
    """△ 三角形（警告風格）"""
    pen = QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    poly = QPolygonF([
        QPointF(cx, cy - r * 0.7),
        QPointF(cx - r * 0.7, cy + r * 0.55),
        QPointF(cx + r * 0.7, cy + r * 0.55),
        QPointF(cx, cy - r * 0.7),
    ])
    p.drawPolyline(poly)


def _draw_scissors(p: QPainter, cx: float, cy: float, r: float, color: QColor):
    """✂ 剪刀"""
    pen = QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    # 上圓
    p.drawEllipse(QPointF(cx - r * 0.3, cy - r * 0.35), r * 0.3, r * 0.3)
    # 下圓
    p.drawEllipse(QPointF(cx - r * 0.3, cy + r * 0.35), r * 0.3, r * 0.3)
    # 交叉線
    p.drawLine(QPointF(cx - r * 0.05, cy - r * 0.15), QPointF(cx + r * 0.7, cy + r * 0.5))
    p.drawLine(QPointF(cx - r * 0.05, cy + r * 0.15), QPointF(cx + r * 0.7, cy - r * 0.5))


def _draw_diamond(p: QPainter, cx: float, cy: float, r: float, color: QColor):
    """◆ 菱形（實心）"""
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(color))
    poly = QPolygonF([
        QPointF(cx, cy - r * 0.65),
        QPointF(cx + r * 0.5, cy),
        QPointF(cx, cy + r * 0.65),
        QPointF(cx - r * 0.5, cy),
    ])
    p.drawPolygon(poly)


def _draw_cross(p: QPainter, cx: float, cy: float, r: float, color: QColor):
    """✕ 叉叉"""
    pen = QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    d = r * 0.5
    p.drawLine(QPointF(cx - d, cy - d), QPointF(cx + d, cy + d))
    p.drawLine(QPointF(cx - d, cy + d), QPointF(cx + d, cy - d))


# ── Plan 設定 ────────────────────────────────────────────────
# (event_key, draw_func, 顏色, tooltip)
_PLAN_CONFIG = [
    ('plan_a_ok',  _draw_check,    '#4caf50', 'Plan A（原圖）成功'),
    ('plan_a2_ok', _draw_triangle, '#ffc107', 'Plan B（黑圖）成功'),
    ('slice_ok',   _draw_scissors, '#ff9800', 'Plan C（切片）成功'),
    ('grok_ok',    _draw_diamond,  '#e91e96', 'Plan D（Grok）成功'),
    ('error',      _draw_cross,    '#f44336', '異常 / 放棄'),
]


# ── 單一指標 Widget ──────────────────────────────────────────
class _StatIcon(QWidget):
    """單一 Plan 的 QPainter 圖示 + 計數"""

    ICON_SIZE = 11
    WIDGET_W  = 46
    WIDGET_H  = 22

    def __init__(self, draw_func, color_hex: str, tooltip: str, parent=None):
        super().__init__(parent)
        self._draw_func = draw_func
        self._color = QColor(color_hex)
        self._count = 0
        self.setToolTip(tooltip)
        self.setFixedSize(self.WIDGET_W, self.WIDGET_H)

    def increment(self):
        self._count += 1
        self.update()

    def reset(self):
        self._count = 0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        h = self.height()
        cy = h / 2.0
        icon_cx = self.ICON_SIZE / 2.0 + 2
        icon_r = self.ICON_SIZE / 2.0

        # 繪製圖示
        self._draw_func(p, icon_cx, cy, icon_r, self._color)

        # 計數文字
        font = QFont()
        font.setPointSizeF(9.5)
        p.setFont(font)
        p.setPen(QPen(QColor('#cccccc')))
        text_x = int(icon_cx + icon_r + 4)
        p.drawText(text_x, 0, self.WIDGET_W - text_x, h,
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   str(self._count))

        p.end()


# ── 狀態列主體 ────────────────────────────────────────────────
class OcrStatsBar(Widget):
    """
    插入 BottomBar 的 OCR 統計條。

    外部呼叫：
        connect_ocr(stats_signals)   # 綁定 OcrStatsSignals
        reset()                      # Run 開始前清零
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._icons: dict[str, _StatIcon] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(2)

        for key, draw_func, color, tip in _PLAN_CONFIG:
            icon = _StatIcon(draw_func, color, tip, self)
            self._icons[key] = icon
            layout.addWidget(icon)

        self.setFixedHeight(24)

    # ── 公開 API ──────────────────────────────────────────────

    def connect_ocr(self, stats_signals):
        """綁定 OcrStatsSignals.event signal"""
        stats_signals.event.connect(self._on_event)

    def reset(self):
        """Run 開始時清零所有計數"""
        for icon in self._icons.values():
            icon.reset()

    # ── slot ─────────────────────────────────────────────────

    @Slot(str)
    def _on_event(self, event_type: str):
        if event_type in self._icons:
            self._icons[event_type].increment()
