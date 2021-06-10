"""
@File  : defaultbox.py
@Author: CodeCat
@Time  : 2021/6/8 22:36
"""
import math
import torch
from torch import nn


class DefaultBoxGenerator(nn.Module):
    def __init__(self, aspect_ratios, min_ratio, max_ratio, scales, steps, clip):
        super(DefaultBoxGenerator, self).__init__()
        if steps is not None:
            assert len(aspect_ratios) == len(steps)
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = aspect_ratios
        # [8, 16, 32, 64, 100, 300]
        self.steps = steps
        # True
        self.clip = clip
        num_outputs = len(aspect_ratios)

        if scales is None:
            if num_outputs > 1:
                range_ratios = max_ratio - min_ratio  # 0.9 - 0.15
                self.scales = [min_ratio + range_ratios * k / (num_outputs - 1.0) for k in range(num_outputs)]
                self.scales.append(1.0)
            else:
                self.scales = [min_ratio, max_ratio]
        else:
            # [0.07, 0.15, 0.33, 0.51, 0.69, 0.87. 1.05]
            self.scales = scales

        self._wh_paris = self._generate_wh_pairs(num_outputs)

    def _generate_wh_pairs(self, num_outputs, dtype=torch.float32, device=torch.device('cpu')):
        _wh_pairs = []
        for k in range(num_outputs):
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            for ar in self.aspect_ratios[k]:
                sq_ar = math.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h], [h, w]])

            _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype, device=device))
        return _wh_pairs

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        default_boxes = self._grid_default_boxes(grid_sizes, image_size, dtype=dtype)
        default_boxes = default_boxes.to(device)

        dboxes = []
        for _ in image_list.image_sizes:
            dboxes_in_image = default_boxes
            # (x, y, w, h) -> (xmin, ymin, xmax, ymax)
            dboxes_in_image = torch.cat([dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:],
                                         dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]], dim=-1)
            # 将相对坐标转换为图像中的实际坐标
            dboxes_in_image[:, 0::2] *= image_size[1]
            dboxes_in_image[:, 1::2] *= image_size[0]
            dboxes.append(dboxes_in_image)
        return dboxes


    def num_anchors_per_location(self):
        return [2 + 2 * len(r) for r in self.aspect_ratios]

    def _grid_default_boxes(self, grid_sizes, image_size, dtype):
        default_boxes = []
        for k, f_k in enumerate(grid_sizes):
            if self.steps is not None:
                x_f_k, y_f_k = [img_shape / self.steps[k] for img_shape in image_size]
            else:
                y_f_k, x_f_k = f_k

            shifts_x = ((torch.arange(0, f_k[1])) / x_f_k).to(dtype=dtype)
            shifts_y = ((torch.arange(0, f_k[0])) / y_f_k).to(dtype=dtype)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 每个default box的中心坐标，每个中心坐标有_wh_pairs个fefault box
            shifts = torch.stack((shift_x, shift_y) * len(self._wh_paris[k]), dim=-1).reshape(-1, 2)
            _wh_pairs = self._wh_paris[k].clamp(min=0, max=1) if self.clip else self._wh_paris[k]

            # 对于每个default box添加对应的高和宽
            wh_pairs = _wh_pairs.repeat(len(shift_x), 1)

            default_box = torch.cat((shifts, wh_pairs), dim=1)

            default_boxes.append(default_box)

        return torch.cat(default_boxes, dim=0)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'aspect_ratios={}'.format(self.aspect_ratios)
        s += ', clip={}'.format(self.clip)
        s += ', scales={}'.format(self.scales)
        s += ', steps={}'.format(self.steps)
        s += ')'
        return s
