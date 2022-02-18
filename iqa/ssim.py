from math import exp

import torch
from torch import nn
import torch.nn.functional as F

from .utils import rgb2y, crop_border, convert_image_dtype, reorder_image


__all__ = ['ssim', 'SSIM']


def _create_ssim_window(window_size: int) -> torch.Tensor:
    def gaussian(_window_size, sigma):
        gauss = torch.Tensor(
            [
                exp(-((x - _window_size // 2) ** 2) / float(2 * sigma ** 2))
                for x in range(_window_size)
            ]
        )
        return gauss / gauss.sum()

    window_1d = gaussian(window_size, 1.5).float().unsqueeze(1)
    window_2d = torch.mm(window_1d, window_1d.t()).unsqueeze(0).unsqueeze(0)
    return window_2d


def _apply_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    peak: float = 1.0,
    size_average: bool = True,
) -> torch.Tensor:
    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2

    channel = img1.size()[1]

    mu1 = F.conv2d(img1, window, groups=channel)
    mu2 = F.conv2d(img2, window, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return ssim_map.mean() if size_average else ssim_map


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    border_crop_size: int = 0,
    test_y_channel: bool = False,
    input_order: str = 'NCHW',
    window: torch.Tensor = None,
) -> torch.Tensor:
    if img1.size() != img2.size():
        raise ValueError(f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if border_crop_size != 0:
        img1 = crop_border(img1, border_crop_size, input_order)
        img2 = crop_border(img2, border_crop_size, input_order)

    img1 = convert_image_dtype(img1, torch.float32)
    img2 = convert_image_dtype(img2, torch.float32)

    if test_y_channel:
        order = 'NHWC'
        img1 = reorder_image(img1, input_order, order)
        img2 = reorder_image(img2, input_order, order)

        if img1.size(-1) != 1:
            img1 = rgb2y(img1)
        if img2.size(-1) != 1:
            img2 = rgb2y(img2)
    else:
        order = input_order

    img1 = reorder_image(img1, order, 'NCHW')
    img2 = reorder_image(img2, order, 'NCHW')

    if window is None:
        window = _create_ssim_window(11).to(img1.device)

    window = window.repeat([img1.size(1), 1, 1, 1])

    return _apply_ssim(img1, img2, window, 1., True)


class SSIM(nn.Module):
    def __init__(
        self,
        border_crop_size: int = 0,
        test_y_channel: bool = False,
        input_order: str = 'NCHW',
    ):
        super().__init__()
        self.border_crop_size = border_crop_size
        self.test_y_channel = test_y_channel
        self.input_order = input_order

        self.register_buffer('window', _create_ssim_window(11))

    def forward(self, img1, img2):
        return ssim(
            img1,
            img2,
            self.border_crop_size,
            self.test_y_channel,
            self.input_order,
            self.window,
        )
