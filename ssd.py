"""
@File  : ssd.py
@Author: CodeCat
@Time  : 2021/6/9 21:52
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Any, Dict, List, Optional, Tuple

from _utils import boxes as box_ops
from _utils.box_code import BoxCoder
from _utils.defaultbox import DefaultBoxGenerator
from _utils.ssd_match import SSDMatcher
from _utils.ssd_transform import SSDTransform


def _xavier_init(conv):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)


class SSDScoringHead(nn.Module):
    def __init__(self, module_list, num_columns):
        super(SSDScoringHead, self).__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def forward(self, x):
        all_results = []

        for i, features in enumerate(x):
            results = self.module_list[i](features)

            # permute output from (N， A*K, H, W) to (N, HWA, K)
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels, num_anchors, num_classes):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(nn.Conv2d(channels, num_classes * anchors, kernel_size=3, padding=1))
        _xavier_init(cls_logits)
        super(SSDClassificationHead, self).__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels, num_anchors):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 4 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super(SSDRegressionHead, self).__init__(bbox_reg, 4)


class SSDHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(SSDHead, self).__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, x):
        return {
            'bbox_regression': self.regression_head(x),
            'cls_logits': self.classification_head(x),
        }


class SSD(nn.Module):
    __annotations__ = {
        'box_coder': BoxCoder,
        'proposal_matcher': SSDMatcher,
    }

    def __init__(self, backbone: nn.Module, anchor_generator: DefaultBoxGenerator,
                 size: Tuple[int, int], num_classes: int,
                 image_mean: Optional[List[float]] = None, image_std: Optional[List[float]] = None,
                 head: Optional[nn.Module] = None,
                 score_thresh: float = 0.01,
                 nms_thresh: float = 0.45,
                 detections_per_img: int = 200,
                 iou_thresh: float = 0.5,
                 topk_candidates: int = 400,
                 positive_fraction: float = 0.25):
        """
        :param backbone: 提取特征
        :param anchor_generator: 生成default boxes对于每个特征图
        :param size: 在输入backbone之前，图像需调整到尺寸大小
        :param num_classes: 模型输出类别数
        :param image_mean: 图像预处理时标准化时的均值
        :param image_std: 图像预处理时标准化时的方差
        :param head: 回归器和分类器
        :param score_thresh: 用于后处理检测时的阈值
        :param nms_thresh: 用于后处理nms的阈值
        :param detections_per_img: 每张图片检测回归框的至多数量
        :param iou_thresh: iou阈值用于在训练时判断该anchor是否为正样本
        :param topk_candidates: nms后选取最好的回归框数量
        :param positive_fraction: 在训练期间正负样本的比例
        """
        super(SSD, self).__init__()

        self.backbone = backbone

        self.anchor_generator = anchor_generator

        self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        if head is None:
            out_channels = backbone.out_channels
            assert len(out_channels) == len(anchor_generator.aspect_ratios)
            num_anchors = self.anchor_generator.num_anchors_per_location()
            head = SSDHead(out_channels, num_anchors, num_classes)
        self.head = head

        self.proposal_matcher = SSDMatcher(iou_thresh)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.transform = SSDTransform(size=size, image_mean=image_mean, image_std=image_std)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction

    def forward(self, images, targets):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)

        features = list(features.values())

        head_outputs = self.head(features)

        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None

            matched_idxs = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image['boxes'].numel() == 0:
                    matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                                   device=anchors_per_image.device))
                    continue

                match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))
            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
            return losses
        else:
            detections = self.postprocess_detection(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        bbox_regression = head_outputs['bbox_regression']
        cls_logits = head_outputs['cls_logits']

        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (targets_per_image, bbox_regression_per_image, cls_logits_per_image, anchors_per_image,
             matched_idxs_per_image) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_images = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_images.numel()

            # 计算回归损失
            matched_gt_boxes_per_image = targets_per_image['boxes'][foreground_matched_idxs_per_images]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(F.smooth_l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ))

            gt_classes_target = torch.zeros((cls_logits_per_image.size(0), ), dtype=targets_per_image['labels'].dtype,
                                            device=targets_per_image['labels'].device)
            gt_classes_target[foreground_idxs_per_image] = targets_per_image['labels'][foreground_matched_idxs_per_images]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # 计算分类损失
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes),
            cls_targets.view(-1),
            reduction='none'
        ).view(cls_targets.size())

        # hard 负采样
        forground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * forground_idxs.sum(1, keepdim=True)
        negative_loss = cls_loss.clone()
        # 将正样本损失值置为 -无穷
        negative_loss[forground_idxs] = -float('-inf')
        # 将损失按降序排列，选取损失最大的作为负样本
        values, idx = negative_loss.sort(1, descending=True)
        # 很巧妙，idx代表样本索引，我们要选取前num_negative个，相当于idx[:, num_negative]
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            'bbox_regression': bbox_loss.sum() / N,
            'classification': (cls_loss[forground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }

    def postprocess_detection(self, head_outputs, image_anchors, image_shapes):
        bbox_regression = head_outputs['bbox_regression']
        pred_scores = F.softmax(head_outputs['cls_logits'], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = self.box_coder.decode(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]
                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                num_topk = min(self.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })
        return detections