# import numpy as np
#
#
# class BottleEvalutor():
#     def __init__(self):
#         self.weights = [0.15, 0.09, 0.09, 0.05, 0.13, 0.05, 0.12, 0.13, 0.07, 0.12]
#
#     def iou_threshold(self, gt_width, gt_height):
#         short_size = min(gt_width, gt_height)
#         if short_size < 40:
#             return 0.2
#         elif short_size < 120:
#             return short_size / 200
#         elif short_size < 420:
#             return short_size /1500 + 0.52
#         else:
#             return 0.8
#
#     def get_ap(self, pre_boxes, gt_boxes, gt_ious):
#
#
#
#
#     def __call__(self, predictions, targets, threshold):
#         predict_boxes = predictions["boxes"].to("cpu").numpy()
#         predict_classes = predictions["labels"].to("cpu").numpy()
#         predict_scores = predictions["scores"].to("cpu").numpy()
#         # 根据 threshold
#
#         gt_boxes = targets["boxes"].to("cpu").numpy()
#         gt_labels = targets["labels"].to("cpu").numpy()
#         gt_ious = [self.iou_threshold(min(box[2] - box[0], box[3] - box[1])) for box in gt_boxes]
#
#         # 计算各个类别的ap
#         ap = list()
#         for id in range(1, len(self.weights) + 1):
#             pre_index = predict_classes.index(id)
#             gt_index = gt_labels.index(id)
#             if pre_index
#             id_pre_boxes = predict_boxes[pre_index]
#             id_gt_ious = gt_ious[gt_index]
#             id_gt_boxes = gt_boxes[gt_index]
#             ap.append(self.get_ap(id_pre_boxes, id_gt_boxes, id_gt_ious))
#
#
#
#
#
