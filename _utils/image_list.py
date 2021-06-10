"""
@File  : image_list.py
@Author: CodeCat
@Time  : 2021/6/9 20:29
"""


class ImageList(object):
    def __init__(self, tensors, image_sizes):
        """
        :param tensors: (Tensor)
        :param image_sizes: (List[Tuple[int, int])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)