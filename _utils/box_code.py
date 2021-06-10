"""
@File  : box_code.py
@Author: CodeCat
@Time  : 2021/6/9 15:27
"""
import math
import torch


class BoxCoder(object):
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        bboxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(bboxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """计算回归参数"""
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)

        wx = weights[0]
        wy = weights[1]
        ww = weights[2]
        wh = weights[3]

        proposals_x1 = proposals[:, 0].unsqueeze(1)
        proposals_y1 = proposals[:, 1].unsqueeze(1)
        proposals_x2 = proposals[:, 2].unsqueeze(1)
        proposals_y2 = proposals[:, 3].unsqueeze(1)

        reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
        reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
        reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
        reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

        db_widths = proposals_x2 - proposals_x1
        db_heights = proposals_y2 - proposals_y1
        db_center_x = proposals_x1 + 0.5 * db_widths
        db_center_y = proposals_y1 + 0.5 * db_heights

        gt_widths = reference_boxes_x2 - reference_boxes_x1
        gt_heights = reference_boxes_y2 - reference_boxes_y1
        gt_center_x = reference_boxes_x1 + 0.5 * gt_widths
        gt_center_y = reference_boxes_y1 + 0.5 * gt_heights

        # gt_box: (x_g, y_g, w_g, h_g)  default_box: (x_d, y_d, w_d, h_d)
        # 平移操作：x_g = x_d + dx * w_d; y_g = y_d + dy * h_d  =>  dx = (x_g - x_d) / w_d; dy = (y_g - y_d) / h_d
        # 放缩操作: w_g = exp(dw) * w_d; h_g = exp(dh) * h_d => dw = log(w_g / w_d); dh = log(h_g / h_d)
        targets_dx = wx * (gt_center_x - db_center_x) / db_widths
        targets_dy = wy * (gt_center_y - db_center_y) / db_heights
        target_dw = ww * torch.log(gt_widths / db_widths)
        target_dh = wh * torch.log(gt_heights / db_heights)

        targets = torch.cat((targets_dx, targets_dy, target_dw, target_dh), dim=1)

        return targets

    def decode(self, rel_codes, boxes):
        boxes_per_image = [len(b) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        center_x = boxes[:, 0] + 0.5 * widths
        center_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights

        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_center_x = center_x[:, None] + dx * widths[:, None]
        pred_center_y = center_y[:, None] + dy * heights[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes1 = pred_center_x - torch.tensor(0.5, dtype=pred_center_x.dtype, device=pred_w.device) * pred_w
        pred_boxes2 = pred_center_y - torch.tensor(0.5, dtype=pred_center_y.dtype, device=pred_h.device) * pred_h
        pred_boxes3 = pred_center_x + torch.tensor(0.5, dtype=pred_center_x.dtype, device=pred_w.device) * pred_w
        pred_boxes4 = pred_center_y + torch.tensor(0.5, dtype=pred_center_y.dtype, device=pred_h.device) * pred_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)

        return pred_boxes