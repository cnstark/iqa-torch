from typing import Dict, Tuple
import torch
import einops


__all__ = ['convert_image_dtype', 'reorder_image', 'crop_border']


def convert_image_dtype(img: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert the type and range of the image according to `dtype`.
    It converts the image to desired type and range. If `dtype` is torch.uint8,
    images will be converted to torch.uint8 type with range [0, 255]. If
    `dtype` is torch.float32, it converts the image to torch.float32 type with
    range [0, 1].

    Args:
        img (torch.Tensor): Input image.
        dtype (torch.dtype): Dtype of output image.

    Returns:
        torch.Tensor: output image with `dtype`
    """
    if img.dtype not in (torch.uint8, torch.float32):
        raise TypeError(f'The img type should be torch.float32 or torch.uint8, but got {img.dtype}')
    
    if img.dtype == dtype:
        return img
    elif dtype == torch.uint8:
        return (img.clamp(0., 1.) * 255.).round().to(torch.uint8)
    else:
        return img.to(torch.float32) / 255.


def reorder_image(img: torch.Tensor, input_order: str, output_order: str) -> torch.Tensor:
    """Reorder input image to other order.

    Valid orders include 'HW', 'CHW', 'HWC', 'NHW', 'NCHW', 'NHWC'.

    Args:
        img (torch.Tensor): input image.
        input_order (str): the order of input image.
        output_order (str): the order of output image.

    Returns:
        torch.Tensor: output image with output order.
    """

    def convert_order(order: str) -> Tuple[str, Dict[str, int]]:
        valid_orders = ('HW', 'CHW', 'HWC', 'NHW', 'NCHW', 'NHWC')
        if order not in valid_orders:
            raise ValueError(f'Wrong order {order}. Valid orders include {valid_orders}')
        missing_order = [d for d in 'NCHW' if d not in order]
        if len(missing_order) == 0:
            return ' '.join(order), {}
        else:
            order_list = list(order)
            order_list[0] = '({})'.format(' '.join(missing_order + [order_list[0]]))
            return ' '.join(order_list), dict([[d, 1] for d in missing_order])

    if img.dim() != len(input_order):
        raise ValueError(f'Wrong img dim.')

    input_order_ein, input_order_info = convert_order(input_order)
    output_order_ein, output_order_info = convert_order(output_order)

    return einops.rearrange(
        img,
        '{} -> {}'.format(input_order_ein, output_order_ein),
        **dict(input_order_info, **output_order_info)
    )


def crop_border(img: torch.Tensor, border_crop_size: int, order: str = 'NCHW') -> torch.Tensor:
    """Crop borders of image.

    Args:
        img (torch.Tensor): Input image.
        border_crop_size (int): Crop border for each end of height and weight.
        order (str, optional): The order of input image. Defaults to 'NCHW'.

    Returns:
        torch.Tensor: Output image.
    """
    if border_crop_size == 0:
        return img
    else:
        if order in ('HW, CHW, NHW, NCHW'):
            return img[..., border_crop_size:-border_crop_size, border_crop_size:-border_crop_size]
        else:
            return img[..., border_crop_size:-border_crop_size, border_crop_size:-border_crop_size, :]
