"""
@File  : ssd_transform.py
@Author: CodeCat
@Time  : 2021/6/9 19:40
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .image_list import ImageList


class SSDTransform(nn.Module):
    def __init__(self, size, image_mean, image_std):
        super(SSDTransform, self).__init__()
        self.size = size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors of shape [C, H, W], got {}".format(image.shape))

            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = torch.stack(images, dim=0)
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append(image_size)

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        h, w = image.shape[-2:]

        image = self.resize_image(image)

        if target is None:
            return image, target

        bbox = target['boxes']
        bbox = self.resize_boxes(bbox, [h, w], image.shape[-2:])
        target['boxes'] = bbox
        return image, target


    def resize_image(self, image):
        image = F.interpolate(image[None], size=self.size, mode='bilinear', align_corners=False)[0]
        return image

    @staticmethod
    def resize_boxes(boxes, original_size, new_size):
        ratios = [torch.tensor(s, dtype=torch.float32, device=boxes.device) /
                  torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
                  for s, s_orig in zip(new_size, original_size)]

        ratio_height, ratio_width = ratios

        xmin = boxes[:, 0] * ratio_width
        ymin = boxes[:, 1] * ratio_height
        xmax = boxes[:, 2] * ratio_width
        ymax = boxes[:, 3] * ratio_height

        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred['boxes']
            boxes = self.resize_boxes(boxes, im_s, o_im_s)
            result[i]['boxes'] = boxes
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(({1}), mode='bilinear')".format(_indent, self.size)
        format_string += '\n)'
        return format_string