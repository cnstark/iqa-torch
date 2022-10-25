from typing import Tuple
import torch
from torch import nn

from .lpips_model import LPIPSModel
from ..utils import crop_border, convert_image_dtype, reorder_image


__all__ = ['lpips', 'LPIPS']


def _preprocess_imgs(
    img1: torch.Tensor,
    img2: torch.Tensor,
    border_crop_size: int = 0,
    input_order: str = 'NCHW',
) -> Tuple[torch.Tensor]:
    """Preprocessing images for calculate LPIPS.

    Args:
        img1 (torch.Tensor): Image 1.
        img2 (torch.Tensor): Image 2.
        border_crop_size (int, optional): Crop border for each end of height and weight. Defaults to 0.
        input_order (str, optional): The order of input image. Defaults to 'NCHW'.

    Returns:
        Tuple[torch.Tensor]: Preprocessed images.
    """
    if img1.size() != img2.size():
        raise ValueError(f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if border_crop_size != 0:
        img1 = crop_border(img1, border_crop_size, input_order)
        img2 = crop_border(img2, border_crop_size, input_order)

    img1 = convert_image_dtype(img1, torch.float32)
    img2 = convert_image_dtype(img2, torch.float32)

    img1 = reorder_image(img1, input_order, 'NCHW')
    img2 = reorder_image(img2, input_order, 'NCHW')

    return img1, img2


def lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    net: str = 'alex',
    version: str = '0.1',
    border_crop_size: int = 0,
    input_order: str = 'NCHW',
) -> torch.Tensor:
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Ref: https://github.com/richzhang/PerceptualSimilarity

    Note:
        Since `nn.Module` are included in LPIPS, it is recommended to use `class LPIPS`.

    Note:
        The dtype and the range of images should be:

            1. torch.uint8 type with range [0, 255];
            2. torch.float32 type with range [0, 1].

    Args:
        img1 (torch.Tensor): Image 1.
        img2 (torch.Tensor): Image 2.
        net (str, optional): ['alex','vgg','squeeze'] are the base/trunk networks
            available. Defaults to 'alex'.
        version (str, optional): LPIPS version, corresponds to old arxiv v1
            (https://arxiv.org/abs/1801.03924v1). Defaults to '0.1'.
        border_crop_size (int, optional): Crop border for each end of height and weight. Defaults to 0.
        input_order (str, optional): The order of input image. Defaults to 'NCHW'.

    Returns:
        torch.Tensor: LPIPS result.
    """
    img1, img2 = _preprocess_imgs(img1, img2, border_crop_size, input_order)

    lpips_model = LPIPSModel(net=net, version=version)
    lpips_model.to(img1.device)

    return lpips_model(img1, img2)


class LPIPS(nn.Module):
    def __init__(
        self,
        net: str = 'alex',
        version: str = '0.1',
        border_crop_size: int = 0,
        input_order: str = 'NCHW',
    ):
        """LPIPS (Learned Perceptual Image Patch Similarity) calculator.

        Ref: https://github.com/richzhang/PerceptualSimilarity

        Args:
            net (str, optional): ['alex','vgg','squeeze'] are the base/trunk networks
                available. Defaults to 'alex'.
            version (str, optional): LPIPS version, corresponds to old arxiv v1
                (https://arxiv.org/abs/1801.03924v1). Defaults to '0.1'.
            border_crop_size (int, optional): Crop border for each end of height and weight. Defaults to 0.
            input_order (str, optional): The order of input image. Defaults to 'NCHW'.
        """
        super().__init__()

        self.border_crop_size = border_crop_size
        self.input_order = input_order

        self.lpips_model = LPIPSModel(net=net, version=version)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

        Note:
            The dtype and the range of images should be:

                1. torch.uint8 type with range [0, 255];
                2. torch.float32 type with range [0, 1].

        Args:
            img1 (torch.Tensor): Image 1.
            img2 (torch.Tensor): Image 2.

        Returns:
            torch.Tensor: LPIPS result.
        """
        img1, img2 = _preprocess_imgs(img1, img2, self.border_crop_size, self.input_order)

        return self.lpips_model(img1, img2)
