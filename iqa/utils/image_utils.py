import torch


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
    valid_orders = ('HW', 'CHW', 'HWC', 'NHW', 'NCHW', 'NHWC')
    if input_order not in valid_orders:
        raise ValueError(f'Wrong input_order {input_order}. Valid orders include {valid_orders}')
    if output_order not in valid_orders:
        raise ValueError(f'Wrong output_order {output_order}. Valid orders include {valid_orders}')
    if img.dim() != len(input_order):
        raise ValueError(f'Wrong img dim.')

    out_img = None
    if input_order == output_order:
        out_img = img
    elif input_order.find(output_order) != -1 or input_order.replace('C', '') == output_order:
        # NCHW->(HW, CHW, NHW), NHWC->(HW, HWC, NHW), (CHW, HWC, NHW)->HW
        out_img = img
        if output_order.find('N') == -1 and input_order.find('N') != -1:
            # NCHW->CHW, NHWC->HWC, NHW->HW
            if out_img.size(0) != 1:
                raise ValueError('The dim N must be 1.')
            out_img = out_img[0, ...]
        if output_order.find('C') == -1 and input_order.find('C') != -1:
            if input_order in ('NCHW', 'CHW'):
                # (NCHW, CHW)->(NHW, HW)
                if out_img.size(-3) != 1:
                    raise ValueError('The dim C must be 1.')
                out_img = out_img[..., 0, :, :]
            elif input_order in ('NHWC', 'HWC'):
                # (NHWC, HWC)->(NHW, HW)
                if out_img.size(-1) != 1:
                    raise ValueError('The dim C must be 1.')
                out_img = out_img[..., 0]
    elif output_order.find(input_order) != -1 or output_order.replace('C', '') == input_order:
        # HW->('CHW', 'HWC', 'NHW', 'NCHW', 'NHWC'), CHW->NCHW, HWC->NHWC, NHW->(NCHW, NHWC)
        out_img = img
        if input_order.find('N') == -1 and output_order.find('N') != -1:
            # (HW, CHW, HWC)->(NHW, NCHW, NHWC)
            out_img = out_img[None, ...]
        if input_order.find('C') == -1 and output_order.find('C') != -1:
            if output_order in ('NCHW', 'CHW'):
                # (HW, NHW)->(CHW, NCHW)
                out_img = out_img[..., None, :, :]
            elif output_order in ('NHWC', 'HWC'):
                # (HW, NHW)->(HWC, NHWC)
                out_img = out_img[..., None]
    elif input_order == 'NCHW' and output_order == 'NHWC':
        out_img = img.permute(0, 2, 3, 1)
    elif input_order == 'NHWC' and output_order == 'NCHW':
        out_img = img.permute(0, 3, 1, 2)
    else:
        out_img = img
        if input_order.find('C') == -1:
            # NHW->(CHW, HWC)
            out_img = reorder_image(out_img, input_order, 'N' + output_order)
            out_img = reorder_image(out_img, 'N' + output_order, output_order)
        else:
            # CHW->(HWC, NHW, NHWC), HWC->(CHW, NHW, NCHW), NCHW->HWC, NHWC->CHW
            if input_order.find('N') == -1:
                temp_order = 'N' + input_order
                out_img = reorder_image(out_img, input_order, temp_order)
            else:
                temp_order = input_order
            out_img = reorder_image(out_img, temp_order, 'N' + output_order.replace('N', ''))
            if not 'N' + output_order.replace('N', '') == output_order:
                out_img = reorder_image(out_img, 'N' + output_order.replace('N', ''), output_order)
    return out_img


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
