import torch


__all__ = ['rgb2gray', 'rgb2y', 'rgb2ycbcr', 'ycbcr2rgb']


RGB2YCBCR_MAT = torch.Tensor([
    [65.481, -37.797, 112.0],
    [128.553, -74.203, -93.786],
    [24.966, 112.0, -18.214]
]) / 255.
RGB2YCBCR_BIAS = torch.Tensor([16, 128, 128]) / 255.
RGB2GRAY_MAT = torch.Tensor([0.299, 0.587, 0.114])[:, None]


def rgb2gray(img: torch.Tensor) -> torch.Tensor:
    """Convert a RGB image to a gray image.

    Ref: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray

    Args:
        img (torch.Tensor): The input image. It accepts:
            dtype: torch.float32
            range: [0, 1]
            order: 'HWC' or 'NHWC'

    Returns:
        torch.Tensor: The converted gray image.
    """
    out_img = torch.mm(img.reshape(-1, 3), RGB2GRAY_MAT)
    size = list(img.size())
    size[-1] = 1
    return out_img.reshape(size)


def rgb2y(img: torch.Tensor) -> torch.Tensor:
    """Convert a RGB image to Y channel of YCbCr.

    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (torch.Tensor): The input image. It accepts:
            dtype: torch.float32
            range: [0, 1]
            order: 'HWC' or 'NHWC'

    Returns:
        torch.Tensor: The converted Y channel image.
    """
    out_img = torch.mm(img.reshape(-1, 3), RGB2YCBCR_MAT[:, 0:1]) + RGB2YCBCR_BIAS[0]
    size = list(img.size())
    size[-1] = 1
    return out_img.reshape(size)


def rgb2ycbcr(img: torch.Tensor) -> torch.Tensor:
    """Convert a RGB image to YCbCr image.

    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (torch.Tensor): The input image. It accepts:
            dtype: torch.float32
            range: [0, 1]
            order: 'HWC' or 'NHWC'

    Returns:
        torch.Tensor: The converted YCbCr image.
    """
    out_img = torch.mm(img.reshape(-1, 3), RGB2YCBCR_MAT) + RGB2YCBCR_BIAS
    return out_img.reshape(img.size())


def ycbcr2rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert a YCbCr image to RGB image.

    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (torch.Tensor): The input image. It accepts:
            dtype: torch.float32
            range: [0, 1]
            order: 'HWC' or 'NHWC'

    Returns:
        torch.Tensor: The converted RGB image.
    """
    out_img = torch.mm(img.reshape(-1, 3) - RGB2YCBCR_BIAS, RGB2YCBCR_MAT.inverse())
    return out_img.reshape(img.size())
