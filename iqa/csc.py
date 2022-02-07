import torch


def _convert_input_type_range(img: torch.Tensor) -> torch.Tensor:
    """Convert the type and range of the input image.
    It converts the input image to torch.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
        img (torch.Tensor): The input image. It accepts:
            1. torch.uint8 type with range [0, 255];
            2. torch.float32 type with range [0, 1].
    Returns:
        (torch.Tensor): The converted image with type of torch.float32 and range of
            [0, 1].
    """
    if img.dtype == torch.float32:
        return img
    elif img.dtype == torch.uint8:
        return img.float() / 255.
    else:
        raise TypeError(f'The img type should be torch.float32 or torch.uint8, but got {img.dtype}')


def _convert_output_type_range(img: torch.Tensor, dst_type: torch.dtype) -> torch.Tensor:
    """Convert the type and range of the image according to dst_type.
    It converts the image to desired type and range. If `dst_type` is torch.uint8,
    images will be converted to torch.uint8 type with range [0, 255]. If
    `dst_type` is torch.float32, it converts the image to torch.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
        img (torch.Tensor): The image to be converted with torch.float32 type and
            range [0, 255].
        dst_type (torch.uint8 | torch.float32): If dst_type is torch.uint8, it
            converts the image to torch.uint8 type with range [0, 255]. If
            dst_type is torch.float32, it converts the image to torch.float32 type
            with range [0, 1].
    Returns:
        (torch.Tensor): The converted image with desired type and range.
    """
    if dst_type not in (torch.uint8, torch.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == torch.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.to(dst_type)


def rgb2ycbcr(img: torch.Tensor, y_only: bool = False) -> torch.Tensor:
    """Convert a RGB image to YCbCr image.
    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (torch.Tensor): The input image. It accepts:
            1. torch.uint8 type with range [0, 255];
            2. torch.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        torch.Tensor: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = torch.dot(img, torch.Tensor([65.481, 128.553, 24.966])) + 16.0
    else:
        out_img = torch.matmul(
            img, torch.Tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]])) + torch.Tensor([16, 128, 128])
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img: torch.Tensor, y_only: bool = False) -> torch.Tensor:
    """Convert a BGR image to YCbCr image.
    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (torch.Tensor): The input image. It accepts:
            1. torch.uint8 type with range [0, 255];
            2. torch.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        torch.Tensor: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = torch.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = torch.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert a YCbCr image to RGB image.
    This function produces the same results as Matlab's ycbcr2rgb function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `YCrCb <-> RGB`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (torch.Tensor): The input image. It accepts:
            1. torch.uint8 type with range [0, 255];
            2. torch.float32 type with range [0, 1].
    Returns:
        torch.Tensor: The converted RGB image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = torch.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img: torch.Tensor) -> torch.Tensor:
    """Convert a YCbCr image to BGR image.
    The bgr version of ycbcr2rgb.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `YCrCb <-> BGR`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (torch.Tensor): The input image. It accepts:
            1. torch.uint8 type with range [0, 255];
            2. torch.float32 type with range [0, 1].
    Returns:
        torch.Tensor: The converted BGR image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = torch.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img
