from qtpy.QtCore import Signal, Qt, QPointF, QSize, QLineF, QDateTime, QRectF, QPoint
from qtpy.QtGui import QPen, QColor, QCursor, QPainter, QPixmap, QBrush, QFontMetrics, QImage
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from typing import Union, Tuple, List
import numpy as np
from utils.logger import logger
from utils.fontformat import px2pt

from .image_edit import ImageEditMode, PixmapItem, DrawingLayer, StrokeImgItem
from .canvas import Canvas, TextBlkItem
from .textedit_area import TransPairWidget


class StrokeItemUndoCommand(QUndoCommand):
    def __init__(self, target_layer: DrawingLayer, rect: Tuple[int], qimg: QImage, erasing=False):
        super().__init__()
        self.qimg = qimg
        self.x = rect[0]
        self.y = rect[1]
        self.target_layer = target_layer
        self.key = str(QDateTime.currentMSecsSinceEpoch())
        if erasing:
            self.compose_mode = QPainter.CompositionMode.CompositionMode_DestinationOut
        else:
            self.compose_mode = QPainter.CompositionMode.CompositionMode_SourceOver
        
    def undo(self):
        if self.qimg is not None:
            self.target_layer.removeQImage(self.key)
            self.target_layer.update()

    def redo(self):
        if self.qimg is not None:
            self.target_layer.addQImage(self.x, self.y, self.qimg, self.compose_mode, self.key)
            self.target_layer.scene().update()


class InpaintUndoCommand(QUndoCommand):
    def __init__(self, canvas: Canvas, inpainted: np.ndarray, mask: np.ndarray, inpaint_rect: List[int]):
        super().__init__()
        self.canvas = canvas
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        self.undo_img = np.copy(img_view)
        self.undo_mask = np.copy(mask_view)
        self.redo_img = inpainted
        self.redo_mask = mask
        self.inpaint_rect = inpaint_rect

    def redo(self) -> None:
        inpaint_rect = self.inpaint_rect
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        img_view[:] = self.redo_img
        mask_view[:] = self.redo_mask
        self.canvas.updateLayers()

    def undo(self) -> None:
        inpaint_rect = self.inpaint_rect
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        img_view[:] = self.undo_img
        mask_view[:] = self.undo_mask
        self.canvas.updateLayers()


class EmptyCommand(QUndoCommand):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
    

class RunBlkTransCommand(QUndoCommand):
    def __init__(self, canvas: Canvas, blkitems: List[TextBlkItem], transpairw_list: List[TransPairWidget],  mode: int):
        super().__init__()

        # mode 0: OCR only
        # mode 1: OCR + translate
        # mode 2: OCR + translate + inpaint
        # mode 3: inpaint only
        # mode 4: OCR + inpaint (no translate)
        do_ocr = mode in (0, 1, 2, 4)
        do_translate = mode in (1, 2)
        do_inpaint = mode in (2, 3, 4)

        self.empty_command = None
        if do_inpaint:
            self.empty_command = EmptyCommand()
            canvas.push_draw_command(self.empty_command)

        self.op_counter = -1
        self.blkitems = blkitems
        self.transpairw_list = transpairw_list
        self.has_trans = set()  # 記錄哪些 blkitem 有寫入翻譯欄

        if do_ocr or do_translate:
            for blkitem, transpairw in zip(self.blkitems, self.transpairw_list):
                if do_translate:
                    trs = blkitem.blk.translation
                    if trs:
                        transpairw.e_trans.setPlainTextAndKeepUndoStack(trs)
                        blkitem.setPlainTextAndKeepUndoStack(trs)
                        self.has_trans.add(id(blkitem))
                    blkitem.blk.rich_text = ''
                if do_ocr:
                    transpairw.e_source.setPlainTextAndKeepUndoStack(blkitem.blk.get_text())
                    # LLM OCR 引擎在 OCR 階段就把譯文塞進 blk.translation，順手寫到譯文欄
                    if not do_translate and blkitem.blk.translation:
                        transpairw.e_trans.setPlainTextAndKeepUndoStack(blkitem.blk.translation)
                        blkitem.setPlainTextAndKeepUndoStack(blkitem.blk.translation)
                        self.has_trans.add(id(blkitem))
                        blkitem.blk.rich_text = ''
                    blkitem.setVertical(blkitem.blk.vertical)
                    from ui.textitem import calc_font_size_by_render
                    blkitem.blk.font_size = calc_font_size_by_render(blkitem.blk)
                    blkitem.setFontSize(px2pt(blkitem.blk.font_size))
                    if blkitem.blk.angle != 0:
                        blkitem.setRotation(blkitem.blk.angle)

        self.canvas = canvas
        self.mode = mode
        self.do_ocr = do_ocr
        self.do_translate = do_translate
        self.do_inpaint = do_inpaint
        if do_inpaint:
            self.undo_img_list = []
            self.undo_mask_list = []
            self.redo_img_list = []
            self.redo_mask_list = []
            self.inpaint_rect_lst = []
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            self.num_inpainted = 0
            for item in self.blkitems:
                inpainted_dict = item.blk.region_inpaint_dict
                item.blk.region_inpaint_dict = None
                if inpainted_dict is None:
                    self.undo_img_list.append(None)
                    self.undo_mask_list.append(None)
                    self.redo_mask_list.append(None)
                    self.redo_img_list.append(None)
                    self.inpaint_rect_lst.append(None)
                else:
                    inpaint_rect = inpainted_dict['inpaint_rect']
                    img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                    mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                    self.undo_img_list.append(np.copy(img_view))
                    self.undo_mask_list.append(np.copy(mask_view))
                    self.redo_img_list.append(inpainted_dict['inpainted'])
                    self.redo_mask_list.append(inpainted_dict['mask'])
                    self.inpaint_rect_lst.append(inpaint_rect)
                    self.num_inpainted += 1

    def redo(self) -> None:

        if self.empty_command is not None:
            self.empty_command.redo()

        if self.do_inpaint and self.num_inpainted > 0:
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for inpaint_rect, redo_img, redo_mask in zip(self.inpaint_rect_lst, self.redo_img_list, self.redo_mask_list):
                if inpaint_rect is None:
                    continue
                img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                img_view[:] = redo_img
                mask_view[:] = redo_mask
            self.canvas.updateLayers()

        if self.op_counter < 0:
            self.op_counter += 1
            return

        if self.do_ocr or self.do_translate:
            for blkitem, transpairw in zip(self.blkitems, self.transpairw_list):
                if id(blkitem) in self.has_trans:
                    transpairw.e_trans.redo()
                    blkitem.redo()
                if self.do_ocr:
                    transpairw.e_source.redo()

    def undo(self) -> None:

        if self.empty_command is not None:
            self.empty_command.undo()

        if self.do_inpaint and self.num_inpainted > 0:
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for inpaint_rect, undo_img, undo_mask in zip(self.inpaint_rect_lst, self.undo_img_list, self.undo_mask_list):
                if inpaint_rect is None:
                    continue
                img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                img_view[:] = undo_img
                mask_view[:] = undo_mask
            self.canvas.updateLayers()

        if self.do_ocr or self.do_translate:
            for blkitem, transpairw in zip(self.blkitems, self.transpairw_list):
                if id(blkitem) in self.has_trans:
                    transpairw.e_trans.undo()
                    blkitem.undo()
                if self.do_ocr:
                    transpairw.e_source.undo()