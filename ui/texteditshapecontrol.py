import math
import numpy as np

from qtpy.QtWidgets import QGraphicsPixmapItem, QGraphicsItem, QWidget, QGraphicsSceneHoverEvent, QLabel, QStyleOptionGraphicsItem, QGraphicsSceneMouseEvent, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem
from qtpy.QtCore import Qt, QRect, QRectF, QPointF, QPoint
from qtpy.QtGui import QPainter, QPen, QColor, QBrush
from utils.imgproc_utils import xywh2xyxypoly, rotate_polygons
from typing import List, Union, Tuple

from .cursor import resizeCursorList
from .textitem import TextBlkItem

CBEDGE_WIDTH = 20   # 縮小控制點（原為 30）
ROTATE_HANDLE_DIST = 28   # 旋轉把手距框頂端的距離（px，在 scene 座標）
ROTATE_HANDLE_R = 6       # 旋轉把手圓半徑（px）
RESET_HANDLE_DIST = 28    # 重置把手距框底端的距離（px）
RESET_HANDLE_R = 6        # 重置把手圓半徑（px）
SNAP_ANGLES = [0, 90, 180, 270, 360]   # 吸附角度
SNAP_THRESHOLD = 5.0                    # 吸附閾值（度）


class ControlBlockItem(QGraphicsRectItem):
    DRAG_NONE = 0
    DRAG_RESHAPE = 1
    CURSOR_IDX = -1

    def __init__(self, parent, idx: int):
        super().__init__(parent)
        self.idx = idx
        self.ctrl: TextBlkShapeControl = parent
        self.edge_width = 0
        self.drag_mode = self.DRAG_NONE
        self.setAcceptHoverEvents(True)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.updateEdgeWidth(CBEDGE_WIDTH)

    def updateEdgeWidth(self, edge_width: float):
        self.edge_width = edge_width
        self.visible_len = self.edge_width // 2
        self.pen_width = edge_width / CBEDGE_WIDTH * 2
        offset = self.edge_width // 4 + self.pen_width / 2
        self.visible_rect = QRectF(offset, offset, self.visible_len, self.visible_len)
        self.setRect(0, 0, self.edge_width, self.edge_width)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        rect = QRectF(self.visible_rect)
        rect.setTopLeft(self.boundingRect().topLeft() + rect.topLeft())
        painter.setPen(QPen(QColor(75, 75, 75), self.pen_width, Qt.PenStyle.SolidLine, Qt.SquareCap))
        painter.fillRect(rect, QColor(200, 200, 200, 125))
        painter.drawRect(rect)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        return super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        self.drag_mode = self.DRAG_NONE
        self.CURSOR_IDX = -1
        if self.drag_mode == self.DRAG_NONE:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        return super().hoverLeaveEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        angle = self.ctrl.rotation() + 45 * self.idx
        idx = self.get_angle_idx(angle)
        self.setCursor(resizeCursorList[idx % 4])
        self.CURSOR_IDX = idx
        return super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.ctrl.ctrlblockPressed()
        blk_item = self.ctrl.blk_item
        if event.button() == Qt.MouseButton.LeftButton:
            self.ctrl.reshaping = True
            self.drag_mode = self.DRAG_RESHAPE
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            blk_item.startReshape()
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseMoveEvent(event)
        blk_item = self.ctrl.blk_item
        if blk_item is None:
            return

        if self.drag_mode == self.DRAG_RESHAPE:
            block_group = self.ctrl.ctrlblock_group
            crect = self.ctrl.rect()
            pos_x, pos_y = 0, 0
            opposite_block = block_group[(self.idx + 4) % 8]
            oppo_pos = opposite_block.pos()
            if self.idx % 2 == 0:
                if self.idx == 0:
                    pos_x = min(self.pos().x(), oppo_pos.x())
                    pos_y = min(self.pos().y(), oppo_pos.y())
                    crect.setX(pos_x + self.visible_len)
                    crect.setY(pos_y + self.visible_len)
                elif self.idx == 2:
                    pos_x = max(self.pos().x(), oppo_pos.x())
                    pos_y = min(self.pos().y(), oppo_pos.y())
                    crect.setWidth(pos_x - oppo_pos.x())
                    crect.setY(pos_y + self.visible_len)
                elif self.idx == 4:
                    pos_x = max(self.pos().x(), oppo_pos.x())
                    pos_y = max(self.pos().y(), oppo_pos.y())
                    crect.setWidth(pos_x - oppo_pos.x())
                    crect.setHeight(pos_y - oppo_pos.y())
                else:   # idx == 6
                    pos_x = min(self.pos().x(), oppo_pos.x())
                    pos_y = max(self.pos().y(), oppo_pos.y())
                    crect.setX(pos_x + self.visible_len)
                    crect.setHeight(pos_y - oppo_pos.y())
            else:
                if self.idx == 1:
                    pos_y = min(self.pos().y(), oppo_pos.y())
                    crect.setY(pos_y + self.visible_len)
                elif self.idx == 3:
                    pos_x = max(self.pos().x(), oppo_pos.x())
                    crect.setWidth(pos_x - oppo_pos.x())
                elif self.idx == 5:
                    pos_y = max(self.pos().y(), oppo_pos.y())
                    crect.setHeight(pos_y - oppo_pos.y())
                else:   # idx == 7
                    pos_x = min(self.pos().x(), oppo_pos.x())
                    crect.setX(pos_x + self.visible_len)

            self.ctrl.setRect(crect)
            scale = self.ctrl.current_scale
            new_center = self.ctrl.sceneBoundingRect().center()
            new_xy = QPointF(new_center.x() / scale - crect.width() / 2, new_center.y() / scale - crect.height() / 2)
            rect = QRectF(new_xy.x(), new_xy.y(), crect.width(), crect.height())
            blk_item.setRect(rect)

    def get_angle_idx(self, angle) -> int:
        idx = int((angle + 22.5) % 360 / 45)
        return idx

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.ctrl.reshaping = False
            if self.drag_mode == self.DRAG_RESHAPE:
                self.ctrl.blk_item.endReshape()
            self.drag_mode = self.DRAG_NONE

            self.ctrl.previewPixmap.setVisible(False)
            self.ctrl.angleLabel.setVisible(False)
            self.ctrl.blk_item.update()
            self.ctrl.updateBoundingRect()
            return super().mouseReleaseEvent(event)


# ─────────────────────────────────────────────────────────────────
# 旋轉把手（框正上方的圓點）
# ─────────────────────────────────────────────────────────────────
class RotateHandleItem(QGraphicsEllipseItem):
    """正上方旋轉圓點，拖曳旋轉整個框"""

    def __init__(self, ctrl: 'TextBlkShapeControl'):
        r = ROTATE_HANDLE_R
        super().__init__(-r, -r, r * 2, r * 2, ctrl)
        self.ctrl = ctrl
        self._dragging = False
        self._rotate_start = 0.0
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self._update_style(False)

    def _update_style(self, hover: bool):
        if hover:
            self.setBrush(QBrush(QColor(30, 147, 229)))
            self.setPen(QPen(QColor(255, 255, 255), 1.5))
        else:
            self.setBrush(QBrush(QColor(200, 200, 200, 200)))
            self.setPen(QPen(QColor(75, 75, 75), 1.5))

    def hoverEnterEvent(self, event):
        self._update_style(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self._dragging:
            self._update_style(False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.ctrl.ctrlblockPressed()
            blk_item = self.ctrl.blk_item
            if blk_item is not None:
                preview = self.ctrl.previewPixmap
                preview.setPixmap(blk_item.toPixmap().copy(blk_item.unpadRect(blk_item.boundingRect()).toRect()))
                preview.setOpacity(0.7)
                preview.setVisible(True)
            rotate_vec = event.scenePos() - self.ctrl.sceneBoundingRect().center()
            rotation = np.rad2deg(math.atan2(rotate_vec.y(), rotate_vec.x()))
            self._rotate_start = -rotation + self.ctrl.rotation()
            self._updateAngleLabel()
            event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self._dragging:
            rotate_vec = event.scenePos() - self.ctrl.sceneBoundingRect().center()
            rotation = np.rad2deg(math.atan2(rotate_vec.y(), rotate_vec.x()))
            raw_angle = (rotation + self._rotate_start) % 360

            # 吸附到整數角度
            snapped = _snap_angle(raw_angle)
            self.ctrl.setAngle(snapped)
            self._updateAngleLabel()
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._update_style(False)
            self.ctrl.previewPixmap.setVisible(False)
            self.ctrl.angleLabel.setVisible(False)
            blk_item = self.ctrl.blk_item
            if blk_item is not None:
                blk_item.rotated.emit(self.ctrl.rotation())
                blk_item.update()
            self.ctrl.updateBoundingRect()
        event.accept()

    def _updateAngleLabel(self):
        angleLabel = self.ctrl.angleLabel
        sp = self.scenePos()
        gv = angleLabel.parent()
        pos = gv.mapFromScene(sp)
        x = max(min(pos.x(), gv.width() - angleLabel.width()), 0)
        y = max(min(pos.y() - angleLabel.height() - 8, gv.height() - angleLabel.height()), 0)
        angleLabel.move(QPoint(x, y))
        angleLabel.setText("{:.1f}°".format(self.ctrl.rotation()))
        if not angleLabel.isVisible():
            angleLabel.setVisible(True)
            angleLabel.raise_()


# ─────────────────────────────────────────────────────────────────
# 重置把手（框正下方的圓點，點擊歸零角度）
# ─────────────────────────────────────────────────────────────────
class ResetHandleItem(QGraphicsEllipseItem):
    """正下方重置圓點，點擊後角度歸零"""

    def __init__(self, ctrl: 'TextBlkShapeControl'):
        r = RESET_HANDLE_R
        super().__init__(-r, -r, r * 2, r * 2, ctrl)
        self.ctrl = ctrl
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self._update_style(False)

    def _update_style(self, hover: bool):
        if hover:
            self.setBrush(QBrush(QColor(229, 80, 57)))
            self.setPen(QPen(QColor(255, 255, 255), 1.5))
        else:
            self.setBrush(QBrush(QColor(200, 200, 200, 200)))
            self.setPen(QPen(QColor(75, 75, 75), 1.5))

    def hoverEnterEvent(self, event):
        self._update_style(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._update_style(False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.ctrl.ctrlblockPressed()
            self.ctrl.setAngle(0)
            blk_item = self.ctrl.blk_item
            if blk_item is not None:
                blk_item.rotated.emit(0.0)
                blk_item.update()
            self.ctrl.updateBoundingRect()
        event.accept()


# ─────────────────────────────────────────────────────────────────
# 角度吸附輔助函數
# ─────────────────────────────────────────────────────────────────
def _snap_angle(angle: float) -> float:
    """接近 SNAP_ANGLES 中任一角度時吸附"""
    angle = angle % 360
    for snap in SNAP_ANGLES:
        diff = abs(angle - snap)
        diff = min(diff, 360 - diff)
        if diff <= SNAP_THRESHOLD:
            return float(snap % 360)
    return angle


class TextBlkShapeControl(QGraphicsRectItem):
    blk_item: TextBlkItem = None
    ctrl_block: ControlBlockItem = None
    reshaping: bool = False

    def __init__(self, parent) -> None:
        super().__init__()
        self.gv = parent
        self.ctrlblock_group = [
            ControlBlockItem(self, idx) for idx in range(8)
        ]

        # 旋轉把手（正上方圓點）
        self._rotate_handle = RotateHandleItem(self)
        self._rotate_handle.hide()

        # 正上方連接線
        self._rotate_line = QGraphicsLineItem(self)
        self._rotate_line.setPen(QPen(QColor(100, 100, 100, 180), 1, Qt.PenStyle.DashLine))
        self._rotate_line.hide()

        # 重置把手（正下方圓點）
        self._reset_handle = ResetHandleItem(self)
        self._reset_handle.hide()

        # 正下方連接線
        self._reset_line = QGraphicsLineItem(self)
        self._reset_line.setPen(QPen(QColor(100, 100, 100, 180), 1, Qt.PenStyle.DashLine))
        self._reset_line.hide()

        self.previewPixmap = QGraphicsPixmapItem(self)
        self.previewPixmap.setVisible(False)
        pen = QPen(QColor(69, 71, 87), 2, Qt.PenStyle.SolidLine)
        pen.setDashPattern([7, 14])
        self.setPen(pen)
        self.setVisible(False)

        self.angleLabel = QLabel(parent)
        self.angleLabel.setText("{:.1f}°".format(self.rotation()))
        self.angleLabel.setObjectName("angleLabel")
        self.angleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angleLabel.setHidden(True)

        self.current_scale = 1.
        self.need_rescale = False
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def setBlkItem(self, blk_item: TextBlkItem):
        if self.blk_item == blk_item and self.isVisible():
            return
        if self.blk_item is not None:
            self.blk_item.under_ctrl = False
            if self.blk_item.isEditing():
                self.blk_item.endEdit()
            self.blk_item.update()

        self.blk_item = blk_item
        if blk_item is None:
            self.hide()
            return
        blk_item.under_ctrl = True
        blk_item.update()
        self.updateBoundingRect()
        self.show()

    def updateBoundingRect(self):
        if self.blk_item is None:
            return
        abr = self.blk_item.absBoundingRect(qrect=True)
        br = QRectF(0, 0, abr.width(), abr.height())
        self.setRect(br)
        self.blk_item.setCenterTransform()
        self.setTransformOriginPoint(self.blk_item.transformOriginPoint())
        self.setPos(abr.x(), abr.y())
        self.setAngle(self.blk_item.angle)

    def setRect(self, *args):
        super().setRect(*args)
        self.updateControlBlocks()

    def updateControlBlocks(self):
        b_rect = self.rect()
        b_rect = [b_rect.x(), b_rect.y(), b_rect.width(), b_rect.height()]
        corner_pnts = xywh2xyxypoly(np.array([b_rect])).reshape(-1, 2)
        edge_pnts = (corner_pnts[[1, 2, 3, 0]] + corner_pnts) / 2
        pnts = [edge_pnts, corner_pnts]
        for ii, ctrlblock in enumerate(self.ctrlblock_group):
            is_corner = not ii % 2
            idx = ii // 2
            pos = pnts[is_corner][idx] - 0.5 * ctrlblock.edge_width
            ctrlblock.setPos(pos[0], pos[1])

        # 更新旋轉把手位置（框頂邊中點正上方）
        r = self.rect()
        scale = self.current_scale
        dist = ROTATE_HANDLE_DIST / scale
        top_center_x = r.x() + r.width() / 2
        top_center_y = r.y()
        self._rotate_handle.setPos(top_center_x, top_center_y - dist)
        self._rotate_line.setLine(top_center_x, top_center_y, top_center_x, top_center_y - dist + ROTATE_HANDLE_R / scale)

        # 更新重置把手位置（框底邊中點正下方）
        dist_reset = RESET_HANDLE_DIST / scale
        bot_center_x = r.x() + r.width() / 2
        bot_center_y = r.y() + r.height()
        self._reset_handle.setPos(bot_center_x, bot_center_y + dist_reset)
        self._reset_line.setLine(bot_center_x, bot_center_y, bot_center_x, bot_center_y + dist_reset - RESET_HANDLE_R / scale)

    def setAngle(self, angle: int) -> None:
        center = self.boundingRect().center()
        self.setTransformOriginPoint(center)
        self.setRotation(angle)

    def ctrlblockPressed(self):
        self.scene().clearSelection()
        if self.blk_item is not None:
            self.blk_item.endEdit()

    def paint(self, painter: QPainter, option: 'QStyleOptionGraphicsItem', widget=...) -> None:
        painter.setCompositionMode(QPainter.CompositionMode.RasterOp_NotDestination)
        super().paint(painter, option, widget)

    def hideControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.hide()
        self._rotate_handle.hide()
        self._rotate_line.hide()
        self._reset_handle.hide()
        self._reset_line.hide()

    def showControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.show()
        self._rotate_handle.show()
        self._rotate_line.show()
        self._reset_handle.show()
        self._reset_line.show()

    def updateScale(self, scale: float):
        if not self.isVisible():
            if scale != self.current_scale:
                self.need_rescale = True
                self.current_scale = scale
            return

        self.current_scale = scale
        scale_inv = 1 / scale
        pen = self.pen()
        pen.setWidthF(2 * scale_inv)
        self.setPen(pen)
        for ctrl in self.ctrlblock_group:
            ctrl.updateEdgeWidth(CBEDGE_WIDTH * scale_inv)

        # 旋轉/重置把手跟 line 的粗細也要跟著縮放
        lpen = QPen(QColor(100, 100, 100, 180), scale_inv, Qt.PenStyle.DashLine)
        self._rotate_line.setPen(lpen)
        self._reset_line.setPen(QPen(lpen))

        r_size = ROTATE_HANDLE_R * scale_inv
        self._rotate_handle.setRect(-r_size, -r_size, r_size * 2, r_size * 2)
        rs_size = RESET_HANDLE_R * scale_inv
        self._reset_handle.setRect(-rs_size, -rs_size, rs_size * 2, rs_size * 2)

        # 重新計算把手位置（距離也要 scale-compensate）
        self.updateControlBlocks()

    def show(self) -> None:
        super().show()
        if self.need_rescale:
            self.updateScale(self.current_scale)
            self.need_rescale = False
        self.setZValue(1)
        self._rotate_handle.show()
        self._rotate_line.show()
        self._reset_handle.show()
        self._reset_line.show()

    def hide(self) -> None:
        super().hide()
        self._rotate_handle.hide()
        self._rotate_line.hide()
        self._reset_handle.hide()
        self._reset_line.hide()

    def startEditing(self):
        self.setCursor(Qt.CursorShape.IBeamCursor)
        for ctrlb in self.ctrlblock_group:
            ctrlb.hide()
        self._rotate_handle.hide()
        self._rotate_line.hide()
        self._reset_handle.hide()
        self._reset_line.hide()

    def endEditing(self):
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        if self.isVisible():
            for ctrlb in self.ctrlblock_group:
                ctrlb.show()
            self._rotate_handle.show()
            self._rotate_line.show()
            self._reset_handle.show()
            self._reset_line.show()
