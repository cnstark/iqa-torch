import torch
from torch import nn

from ..utils import rgb2gray, rgb2y, crop_border, convert_image_dtype, reorder_image
from .niqe_core import _apply_niqe


__all__ = ['niqe', 'NIQE']


def niqe(
    img: torch.Tensor,
    border_crop_size: int = 0,
    test_y_channel: bool = False,
    input_order: str = 'NCHW',
) -> torch.Tensor:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Ref: https://github.com/xinntao/BasicSR/blob/master/basicsr/metrics/niqe.py

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Note:
        The dtype and the range of images should be:

            1. torch.uint8 type with range [0, 255];
            2. torch.float32 type with range [0, 1].

    Args:
        img (torch.Tensor): Input image whose quality needs to be computed.
        border_crop_size (int, optional): Crop border for each end of height and width.
            Defaults to 0.
        test_y_channel (bool, optional): Test on Y channel of YCbCr.
            Defaults to False, which means that the image will test on gray.
        input_order (str, optional): _description_. Defaults to 'NCHW'.

    Returns:
        torch.Tensor: NIQE result.
    """
    if border_crop_size != 0:
        img = crop_border(img, border_crop_size, input_order)

    img = convert_image_dtype(img, torch.float32)

    img = reorder_image(img, input_order, 'HWC')

    if img.size(-1) != 1:
        if test_y_channel:
            img = rgb2y(img)
        else:
            img = rgb2gray(img)

    img = convert_image_dtype(img, torch.uint8)

    img = reorder_image(img, 'HWC', 'HW')

    return torch.Tensor([_apply_niqe(img)])


class NIQE(nn.Module):
    def __init__(
        self,
        border_crop_size: int = 0,
        test_y_channel: bool = False,
        input_order: str = None,
    ):
        """NIQE (Natural Image Quality Evaluator) calculator

        Ref: Making a "Completely Blind" Image Quality Analyzer.
        This implementation could produce almost the same results as the official
        MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

        Args:
            border_crop_size (int, optional): Crop border for each end of height and width.
            Defaults to 0.
            test_y_channel (bool, optional): Test on Y channel of YCbCr.
                Defaults to False, which means that the image will test on gray.
            input_order (str, optional): _description_. Defaults to 'NCHW'.
        """
        super().__init__()
        self.border_crop_size = border_crop_size
        self.test_y_channel = test_y_channel
        self.input_order = input_order

    def forward(self, img1: torch.Tensor, img2: torch.Tensor = None) -> torch.Tensor:
        """Calculate NIQE (Natural Image Quality Evaluator) metric.

        Note:
            The dtype and the range of images should be:

                1. torch.uint8 type with range [0, 255];
                2. torch.float32 type with range [0, 1].

        Args:
            img1 (torch.Tensor): Input image whose quality needs to be computed.
            img2 (torch.Tensor): Variable placeholders for uniform format. Defaults to None.

        Returns:
            torch.Tensor: NIQE result.
        """
        return niqe(img1, self.border_crop_size, self.test_y_channel, self.input_order)
