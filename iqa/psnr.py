import torch
from torch import nn
from torch.nn import functional as F

from .utils import rgb2y, crop_border, convert_image_dtype, reorder_image


__all__ = ['psnr', 'PSNR']


def _apply_psnr(img1: torch.Tensor, img2: torch.Tensor, peak: float = 1.0) -> torch.Tensor:
    """PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (torch.Tensor): Image with range [0, `peak`].
        img2 (torch.Tensor): Image with range [0, `peak`].
        peak (float, optional): Peak of images. Defaults to 1.0.

    Returns:
        [torch.Tensor]: PSNR value.
    """
    return 10 * torch.log10((peak ** 2) / F.mse_loss(img1, img2))


def psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    border_crop_size: int = 0,
    test_y_channel: bool = False,
    input_order: str = None,
) -> torch.Tensor:
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Note:
        The dtype and the range of images should be:

            1. torch.uint8 type with range [0, 255];
            2. torch.float32 type with range [0, 1].

    Note:
        The order of images must be specified when `border_crop_size` is
        not 0 or `test_y_channel` is True.

    Note:
        If `test_y_channel` is True, `img1` and `img2` should be:

            1. RGB image with 3 channel;
            2. Gray image with 1 channel.

    Args:
        img1 (torch.Tensor): Image 1.
        img2 (torch.Tensor): Image 2.
        border_crop_size (int, optional): Crop border for each end of height and weight. Defaults to 0.
        test_y_channel (bool, optional): Test on Y channel of YCbCr. Defaults to False.
        input_order (str, optional): The order of input image. Defaults to None.

    Returns:
        torch.Tensor: PSNR value.
    """
    if img1.size() != img2.size():
        raise ValueError(f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if border_crop_size != 0:
        if input_order is None:
            raise ValueError(f'Please specify the order of images.')
        img1 = crop_border(img1, border_crop_size, input_order)
        img2 = crop_border(img2, border_crop_size, input_order)

    img1 = convert_image_dtype(img1, torch.float32)
    img2 = convert_image_dtype(img2, torch.float32)

    if test_y_channel:
        if input_order is None:
            raise ValueError(f'Please specify the order of images.')
        img1 = reorder_image(img1, input_order, 'NHWC')
        img2 = reorder_image(img2, input_order, 'NHWC')

        if img1.size(-1) != 1:
            img1 = rgb2y(img1)
        if img2.size(-1) != 1:
            img2 = rgb2y(img2)

    return _apply_psnr(img1, img2, 1.)


class PSNR(nn.Module):
    def __init__(
        self,
        border_crop_size: int = 0,
        test_y_channel: bool = False,
        input_order: str = None,
    ):
        """PSNR calculator

        Note:
            The order of images must be specified when `border_crop_size` is
            not 0 or `test_y_channel` is True.

        Note:
        If `test_y_channel` is True, `img1` and `img2` should be:

            1. RGB image with 3 channel;
            2. Gray image with 1 channel.

        Args:
            border_crop_size (int, optional): Crop border for each end of height and weight. Defaults to 0.
            test_y_channel (bool, optional): Test on Y channel of YCbCr. Defaults to False.
            input_order (str, optional): The order of input image. Defaults to None.
        """
        super().__init__()
        self.border_crop_size = border_crop_size
        self.test_y_channel = test_y_channel
        self.input_order = input_order

        if border_crop_size != 0 or test_y_channel:
            if input_order is None:
                raise ValueError(f'Please specify the order of images.')

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Calculate PSNR (Peak Signal-to-Noise Ratio).

        Note:
            The dtype and the range of images should be:

                1. torch.uint8 type with range [0, 255];
                2. torch.float32 type with range [0, 1].

        Args:
            img1 (torch.Tensor): Image 1.
            img2 (torch.Tensor): Image 2.

        Returns:
            torch.Tensor: PSNR value.
        """
        return psnr(img1, img2, self.border_crop_size, self.test_y_channel, self.input_order)
