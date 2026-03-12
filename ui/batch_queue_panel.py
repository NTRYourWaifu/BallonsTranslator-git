"""
ui/batch_queue_panel.py

批量翻譯佇列面板
- 外部拖入資料夾加入佇列
- 列表內拖動排序
- 點擊圓點展開/收合 5 色 OCR 統計（完整 press+release 才觸發，不與拖曳衝突）
"""

import os
import os.path as osp

from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QAbstractItemView, QMenu,
    QWidget, QStyledItemDelegate, QStyleOptionViewItem, QStyle,
)
from qtpy.QtCore import Qt, Signal, QSize, QRect, QPoint, QPointF, QModelIndex
from qtpy.QtGui import (
    QDragEnterEvent, QDropEvent, QPainter, QColor, QFont,
    QPen, QBrush, QPainterPath, QPolygonF, QMouseEvent,
)

from .custom_widget.widget import Widget


# ── 狀態 ──────────────────────────────────────────────────────
class _Status:
    PENDING  = 'pending'
    RUNNING  = 'running'
    DONE     = 'done'
    ERROR    = 'error'

_STATUS_COLORS = {
    _Status.PENDING:  QColor('#555555'),
    _Status.RUNNING:  QColor('#ffc107'),
    _Status.DONE:     QColor('#4caf50'),
    _Status.ERROR:    QColor('#f44336'),
}

_STATUS_LABEL = {
    _Status.PENDING:  '等待中',
    _Status.RUNNING:  '執行中',
    _Status.DONE:     '完成',
    _Status.ERROR:    '異常',
}

ROLE_PATH       = Qt.ItemDataRole.UserRole
ROLE_STATUS     = Qt.ItemDataRole.UserRole + 1
ROLE_OCR_STATS  = Qt.ItemDataRole.UserRole + 2
ROLE_EXPANDED   = Qt.ItemDataRole.UserRole + 3

_OCR_PLANS = [
    ('plan_a_ok',  '#4caf50'),
    ('plan_a2_ok', '#ffc107'),
    ('slice_ok',   '#ff9800'),
    ('grok_ok',    '#e91e96'),
    ('error',      '#f44336'),
]

_OCR_TOOLTIPS = [
    'Plan A 原圖成功', 'Plan B 黑圖成功', 'Plan C 切片成功',
    'Plan D Grok 備援', '異常 / 放棄',
]


# ── 小圖示繪製 ────────────────────────────────────────────────
def _draw_check(p, cx, cy, r, color):
    pen = QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(cx - r*0.6, cy + r*0.05)
    path.lineTo(cx - r*0.1, cy + r*0.55)
    path.lineTo(cx + r*0.65, cy - r*0.5)
    p.drawPath(path)

def _draw_triangle(p, cx, cy, r, color):
    pen = QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    poly = QPolygonF([QPointF(cx, cy-r*0.65), QPointF(cx-r*0.65, cy+r*0.5),
                      QPointF(cx+r*0.65, cy+r*0.5), QPointF(cx, cy-r*0.65)])
    p.drawPolyline(poly)

def _draw_scissors(p, cx, cy, r, color):
    pen = QPen(color, 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(cx-r*0.25, cy-r*0.3), r*0.28, r*0.28)
    p.drawEllipse(QPointF(cx-r*0.25, cy+r*0.3), r*0.28, r*0.28)
    p.drawLine(QPointF(cx, cy-r*0.1), QPointF(cx+r*0.65, cy+r*0.5))
    p.drawLine(QPointF(cx, cy+r*0.1), QPointF(cx+r*0.65, cy-r*0.5))

def _draw_diamond(p, cx, cy, r, color):
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(color))
    poly = QPolygonF([QPointF(cx, cy-r*0.6), QPointF(cx+r*0.45, cy),
                      QPointF(cx, cy+r*0.6), QPointF(cx-r*0.45, cy)])
    p.drawPolygon(poly)

def _draw_cross(p, cx, cy, r, color):
    pen = QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    d = r * 0.45
    p.drawLine(QPointF(cx-d, cy-d), QPointF(cx+d, cy+d))
    p.drawLine(QPointF(cx-d, cy+d), QPointF(cx+d, cy-d))

_OCR_DRAW = [_draw_check, _draw_triangle, _draw_scissors, _draw_diamond, _draw_cross]


# ── Delegate ──────────────────────────────────────────────────
ROW_COLLAPSED = 42
ROW_EXPANDED  = 80
DOT_LEFT = 12
DOT_R    = 6
DOT_HIT  = DOT_LEFT + DOT_R + 6   # 點擊判定區

class _FolderItemDelegate(QStyledItemDelegate):

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect

        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, QColor('#3a3a4a'))
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(rect, QColor('#2a2a38'))

        path     = index.data(ROLE_PATH) or ''
        status   = index.data(ROLE_STATUS) or _Status.PENDING
        stats    = index.data(ROLE_OCR_STATS) or {}
        expanded = index.data(ROLE_EXPANDED) or False
        name     = osp.basename(path) if path else '???'

        dot_color = _STATUS_COLORS.get(status, _STATUS_COLORS[_Status.PENDING])
        status_text = _STATUS_LABEL.get(status, '')

        row1_cy = rect.top() + 21

        # 圓點 or ▼
        if expanded:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(dot_color))
            tri = QPolygonF([
                QPointF(rect.left() + DOT_LEFT - 4, row1_cy - 3),
                QPointF(rect.left() + DOT_LEFT + 4, row1_cy - 3),
                QPointF(rect.left() + DOT_LEFT, row1_cy + 3),
            ])
            painter.drawPolygon(tri)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(dot_color))
            painter.drawEllipse(QPoint(rect.left() + DOT_LEFT, row1_cy), DOT_R, DOT_R)

        text_x = rect.left() + DOT_LEFT + DOT_R + 10
        title_font = QFont(option.font)
        title_font.setPointSizeF(10.5)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor('#e0e0e0'))
        painter.drawText(text_x, rect.top() + 6, rect.width() - text_x - 80, 20,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)

        st_font = QFont(option.font)
        st_font.setPointSizeF(9)
        painter.setFont(st_font)
        painter.setPen(dot_color)
        painter.drawText(rect.right() - 70, rect.top() + 6, 65, 20,
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, status_text)

        if expanded:
            icon_y = rect.top() + 44
            icon_r = 5.0
            gap = 52
            base_x = text_x
            num_font = QFont(option.font)
            num_font.setPointSizeF(9)

            for i, (key, color_hex) in enumerate(_OCR_PLANS):
                ix = base_x + i * gap
                color = QColor(color_hex)
                count = stats.get(key, 0)
                _OCR_DRAW[i](painter, ix, icon_y, icon_r, color)
                painter.setFont(num_font)
                painter.setPen(QColor('#999') if count == 0 else color)
                painter.drawText(int(ix + icon_r + 3), int(icon_y - 8), 24, 16,
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 str(count))

            path_font = QFont(option.font)
            path_font.setPointSizeF(8)
            painter.setFont(path_font)
            painter.setPen(QColor('#666'))
            painter.drawText(text_x, rect.top() + 58, rect.width() - text_x - 8, 16,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, path)

        painter.restore()

    def sizeHint(self, option, index):
        expanded = index.data(ROLE_EXPANDED) or False
        return QSize(option.rect.width(), ROW_EXPANDED if expanded else ROW_COLLAPSED)


# ── ListWidget（拖入 + 內部排序 + 點擊展開）───────────────────
class FolderListWidget(QListWidget):

    folders_dropped = Signal(list)
    folder_clicked = Signal(str)   # 點擊文字區域時發射資料夾路徑

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # 同時支援外部拖入和內部排序
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setItemDelegate(_FolderItemDelegate(self))
        self.setMouseTracking(True)

        self._press_pos = None
        self._press_item = None

        self.setStyleSheet("""
            QListWidget {
                border: 2px dashed #444; border-radius: 6px;
                background: transparent; outline: none;
            }
            QListWidget:hover { border-color: #666; }
            QListWidget::item { border-bottom: 1px solid #2a2a3a; padding: 0; }
        """)

    # ── 點擊展開：用 press+release 判斷，不干擾拖曳 ──

    def mousePressEvent(self, event: QMouseEvent):
        index = self.indexAt(event.pos())
        if index.isValid() and event.button() == Qt.MouseButton.LeftButton:
            item = self.itemFromIndex(index)
            item_rect = self.visualItemRect(item)
            click_x = event.pos().x() - item_rect.left()
            if click_x <= DOT_HIT:
                # 圓點區域：攔截，不走 super（避免啟動拖曳/選取）
                self._press_pos = event.pos()
                self._press_item = item
                self._press_is_dot = True
                return
            else:
                # 文字區域：記錄位置，但正常走 super（保持選取/拖放功能）
                self._press_pos = event.pos()
                self._press_item = item
                self._press_is_dot = False
                # 不 return，繼續走 super
        else:
            self._press_pos = None
            self._press_item = None
            self._press_is_dot = False
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if (self._press_pos is not None
                and self._press_item is not None
                and event.button() == Qt.MouseButton.LeftButton):
            delta = (event.pos() - self._press_pos).manhattanLength()
            if delta < 8:
                item = self._press_item
                if self._press_is_dot:
                    # 圓點區域：展開/收合
                    expanded = item.data(ROLE_EXPANDED) or False
                    item.setData(ROLE_EXPANDED, not expanded)
                    item.setSizeHint(QSize(0, ROW_EXPANDED if not expanded else ROW_COLLAPSED))
                    self.viewport().update()
                    self._press_pos = None
                    self._press_item = None
                    self._press_is_dot = False
                    return
                else:
                    # 文字區域：簡單點擊（非拖曳）→ 切換資料夾
                    path = item.data(ROLE_PATH)
                    if path:
                        self.folder_clicked.emit(path)
        self._press_pos = None
        self._press_item = None
        self._press_is_dot = False
        super().mouseReleaseEvent(event)

    # ── 外部拖入資料夾 ──

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        elif event.source() is self:
            # 內部拖動排序
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() or event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            # 外部拖入資料夾
            dirs = []
            for url in event.mimeData().urls():
                p = url.toLocalFile()
                if osp.isdir(p):
                    dirs.append(p)
            if dirs:
                self.folders_dropped.emit(dirs)
                event.acceptProposedAction()
            else:
                event.ignore()
        elif event.source() is self:
            # 內部排序：交給 QListWidget 處理
            super().dropEvent(event)
        else:
            event.ignore()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        remove_act = menu.addAction(self.tr('移除選中項'))
        clear_act = menu.addAction(self.tr('清空全部'))
        rst = menu.exec_(event.globalPos())
        if rst == remove_act:
            for item in self.selectedItems():
                self.takeItem(self.row(item))
        elif rst == clear_act:
            self.clear()


# ── 主面板 ────────────────────────────────────────────────────
class BatchQueuePanel(Widget):

    run_batch = Signal(list)
    queue_changed = Signal(int)
    open_folder = Signal(str)    # 點擊項目時切換到該資料夾

    MIN_WIDTH = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(self.MIN_WIDTH)

        title = QLabel(self.tr('批量翻譯佇列'))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-size: 14px; font-weight: bold; padding: 8px 0;')

        self.hintLabel = QLabel(self.tr('將資料夾拖入此處'))
        self.hintLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hintLabel.setWordWrap(True)
        self.hintLabel.setStyleSheet("""
            color: #666; font-size: 12px; padding: 20px 8px;
            border: 2px dashed #444; border-radius: 8px;
        """)

        self.folderList = FolderListWidget(self)
        self.folderList.folders_dropped.connect(self.addFolders)
        self.folderList.folder_clicked.connect(self.open_folder.emit)

        legend = QLabel(
            '<span style="color:#555">\u25cf 等待</span>'
            '&nbsp;&nbsp;'
            '<span style="color:#ffc107">\u25cf 執行</span>'
            '&nbsp;&nbsp;'
            '<span style="color:#4caf50">\u25cf 完成</span>'
            '&nbsp;&nbsp;'
            '<span style="color:#f44336">\u25cf 異常</span>'
            '&nbsp;&nbsp;&nbsp;'
            '<span style="color:#888">點擊圓點展開 | 拖動排序</span>'
        )
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend.setStyleSheet('font-size: 9px; padding: 2px 0;')

        btn_style = """
            QPushButton {
                padding: 5px 10px; border: 1px solid #555;
                border-radius: 4px; font-size: 12px;
            }
            QPushButton:hover { background-color: #3a3a4a; border-color: #777; }
            QPushButton:disabled { color: #666; border-color: #444; }
        """
        self.addBtn = QPushButton(self.tr('+ 新增'))
        self.addBtn.setStyleSheet(btn_style)
        self.addBtn.clicked.connect(self.onAddFolder)

        self.removeBtn = QPushButton(self.tr('- 移除'))
        self.removeBtn.setStyleSheet(btn_style)
        self.removeBtn.clicked.connect(self.onRemoveSelected)

        self.clearBtn = QPushButton(self.tr('清空'))
        self.clearBtn.setStyleSheet(btn_style)
        self.clearBtn.clicked.connect(self.onClear)

        self.runAllBtn = QPushButton(self.tr('\u25b6  全部執行'))
        self.runAllBtn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; color: white;
                font-weight: bold; font-size: 13px; padding: 8px;
                border-radius: 5px; border: none;
            }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:disabled { background-color: #444; color: #777; }
        """)
        self.runAllBtn.clicked.connect(self.onRunAll)

        self.countLabel = QLabel()
        self.countLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countLabel.setStyleSheet('color: #888; font-size: 11px;')
        self._updateCount()

        btnRow = QHBoxLayout()
        btnRow.setSpacing(4)
        btnRow.addWidget(self.addBtn)
        btnRow.addWidget(self.removeBtn)
        btnRow.addWidget(self.clearBtn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(self.hintLabel)
        layout.addWidget(self.folderList, stretch=1)
        layout.addWidget(legend)
        layout.addWidget(self.countLabel)
        layout.addLayout(btnRow)
        layout.addWidget(self.runAllBtn)

    # ── 公開方法 ──

    def addFolders(self, dirs: list):
        existing = set(self._getAllPaths())
        added = 0
        for d in dirs:
            d = osp.normpath(d)
            if d not in existing and osp.isdir(d):
                item = QListWidgetItem()
                item.setData(ROLE_PATH, d)
                item.setData(ROLE_STATUS, _Status.PENDING)
                item.setData(ROLE_OCR_STATS, {})
                item.setData(ROLE_EXPANDED, False)
                item.setSizeHint(QSize(0, ROW_COLLAPSED))
                item.setToolTip(d)
                # 啟用拖動（內部排序用）
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
                self.folderList.addItem(item)
                existing.add(d)
                added += 1
        if added:
            self.hintLabel.setVisible(False)
            self._updateCount()

    def markRunning(self, folder):
        self._setStatus(folder, _Status.RUNNING)

    def markDone(self, folder):
        self._setStatus(folder, _Status.DONE)

    def markError(self, folder):
        self._setStatus(folder, _Status.ERROR)

    def updateOcrStats(self, folder: str, event_type: str):
        folder = osp.normpath(folder)
        for i in range(self.folderList.count()):
            item = self.folderList.item(i)
            if osp.normpath(item.data(ROLE_PATH)) == folder:
                stats = item.data(ROLE_OCR_STATS) or {}
                stats[event_type] = stats.get(event_type, 0) + 1
                item.setData(ROLE_OCR_STATS, stats)
                break
        self.folderList.viewport().update()

    def setRunning(self, running: bool):
        self.addBtn.setEnabled(not running)
        self.removeBtn.setEnabled(not running)
        self.clearBtn.setEnabled(not running)
        self.runAllBtn.setEnabled(not running)
        # 執行中禁止拖動排序
        self.folderList.setDragEnabled(not running)
        self.runAllBtn.setText(
            self.tr('\u23f3 執行中...') if running else self.tr('\u25b6  全部執行')
        )

    # ── 內部 ──

    def _getAllPaths(self):
        return [self.folderList.item(i).data(ROLE_PATH)
                for i in range(self.folderList.count())]

    def _setStatus(self, folder, status):
        folder = osp.normpath(folder)
        for i in range(self.folderList.count()):
            item = self.folderList.item(i)
            if osp.normpath(item.data(ROLE_PATH)) == folder:
                item.setData(ROLE_STATUS, status)
                break
        self.folderList.viewport().update()

    def _updateCount(self):
        n = self.folderList.count()
        self.countLabel.setText(f'{n} 個資料夾' if n > 0 else '')
        self.hintLabel.setVisible(n == 0)
        self.queue_changed.emit(n)

    def onAddFolder(self):
        d = QFileDialog.getExistingDirectory(self, self.tr('選擇資料夾'))
        if d: self.addFolders([d])

    def onRemoveSelected(self):
        for item in self.folderList.selectedItems():
            self.folderList.takeItem(self.folderList.row(item))
        self._updateCount()

    def onClear(self):
        self.folderList.clear()
        self._updateCount()

    def onRunAll(self):
        dirs = self._getAllPaths()
        if dirs:
            for i in range(self.folderList.count()):
                self.folderList.item(i).setData(ROLE_STATUS, _Status.PENDING)
                self.folderList.item(i).setData(ROLE_OCR_STATS, {})
            self.folderList.viewport().update()
            self.run_batch.emit(dirs)
