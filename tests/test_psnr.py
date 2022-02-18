import os
import sys
import unittest
sys.path.append(os.getcwd())

import numpy as np
import torch

from iqa import psnr
from libs.basicsr_psnr_ssim import calculate_psnr

IMAGE_SHAPES = (
    ('CHW', (3, 128, 128)),
    ('CHW', (1, 128, 128)),
    ('HWC', (128, 128, 3)),
    ('HWC', (128, 128, 1)),
)


class PSNRTestCase(unittest.TestCase):
    def test_psnr_rgb_float32(self):
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img1_np = np.random.rand(*shape).astype(np.float32)
                img2_np = np.random.rand(*shape).astype(np.float32)
                img1_torch = torch.from_numpy(img1_np)
                img2_torch = torch.from_numpy(img2_np)

                psnr_bsr = calculate_psnr(img1_np * 255., img2_np * 255., 4, order)
                psnr_iqa = psnr(img1_torch, img2_torch, 4, input_order=order)

                self.assertLess(psnr_bsr - psnr_iqa.item(), 1e-5)

    def test_psnr_rgb_uint8(self):
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img1_np = (np.random.rand(*shape) * 255.).astype(np.uint8)
                img2_np = (np.random.rand(*shape) * 255.).astype(np.uint8)
                img1_torch = torch.from_numpy(img1_np)
                img2_torch = torch.from_numpy(img2_np)

                psnr_bsr = calculate_psnr(img1_np, img2_np, 4, order)
                psnr_iqa = psnr(img1_torch, img2_torch, 4, input_order=order)

                self.assertLess(psnr_bsr - psnr_iqa.item(), 1e-5)

    def test_psnr_y_float32(self):
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img1_np = np.random.rand(*shape).astype(np.float32)
                img2_np = np.random.rand(*shape).astype(np.float32)
                img1_torch = torch.from_numpy(img1_np)
                img2_torch = torch.from_numpy(img2_np)

                psnr_bsr = calculate_psnr(img1_np * 255., img2_np * 255., 4, order, True)
                psnr_iqa = psnr(img1_torch, img2_torch, 4, test_y_channel=True, input_order=order)

                self.assertLess(psnr_bsr - psnr_iqa.item(), 1e-5)

    def test_psnr_y_uint8(self):
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img1_np = (np.random.rand(*shape) * 255.).astype(np.uint8)
                img2_np = (np.random.rand(*shape) * 255.).astype(np.uint8)
                img1_torch = torch.from_numpy(img1_np)
                img2_torch = torch.from_numpy(img2_np)

                psnr_bsr = calculate_psnr(img1_np, img2_np, 4, order, True)
                psnr_iqa = psnr(img1_torch, img2_torch, 4, test_y_channel=True, input_order=order)

                self.assertLess(psnr_bsr - psnr_iqa.item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
