"""
@File  : ssd_match.py
@Author: CodeCat
@Time  : 2021/6/9 16:28
"""
import torch


class SSDMatcher(object):
    BELOW_IOU_THRESHOLD = -1

    __annotations__ = {
        'BELOW_IOU_THRESHOLD': int
    }

    def __init__(self, iou_threshold):
        self.BELOW_IOU_THRESHOLD = -1
        self.iou_threshold = iou_threshold

    def __call__(self, match_quality_matrix):
        matched_vals, matches = match_quality_matrix.max(dim=0)
        below_iou_threshold = matched_vals <= self.iou_threshold
        matches[below_iou_threshold] = self.BELOW_IOU_THRESHOLD
        _, highest_quality_pred_foreach_gt = match_quality_matrix.max(dim=1)
        matches[highest_quality_pred_foreach_gt] = torch.arange(highest_quality_pred_foreach_gt.size(0),
                                                                dtype=torch.int64,
                                                                device=highest_quality_pred_foreach_gt.device)
        return matches