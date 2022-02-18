import os
import sys
import unittest
sys.path.append(os.getcwd())

import numpy as np
import torch

from iqa.niqe import niqe
from libs.basicsr_niqe import calculate_niqe


IMAGE_SHAPES = (
    ('CHW', (3, 128, 128)),
    ('CHW', (1, 128, 128)),
    ('HWC', (128, 128, 3)),
    ('HWC', (128, 128, 1)),
)


# TODO: precision
class NIQETestCase(unittest.TestCase):
    def test_niqe_gray_float32(self):
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img_np = np.random.rand(*shape).astype(np.float32)
                img_torch = torch.from_numpy(img_np)

                niqe_bsr = calculate_niqe(img_np * 255, 4, order, 'gray')
                niqe_iqa = niqe(img_torch, 4, False, order)

                self.assertLess(abs(niqe_bsr - niqe_iqa.item()), 1e-4)
    
    def test_niqe_gray_uint8(self):
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img_np = (np.random.rand(*shape) * 255.).astype(np.uint8)
                img_torch = torch.from_numpy(img_np)

                niqe_bsr = calculate_niqe(img_np, 4, order, 'gray')
                niqe_iqa = niqe(img_torch, 4, False, order)

                self.assertLess(abs(niqe_bsr - niqe_iqa.item()), 1e-4)

    def test_niqe_y_float32(self):
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img_np = np.random.rand(*shape).astype(np.float32)
                img_torch = torch.from_numpy(img_np)

                niqe_bsr = calculate_niqe(img_np * 255, 4, order, 'y')
                niqe_iqa = niqe(img_torch, 4, True, order)

                self.assertLess(abs(niqe_bsr - niqe_iqa.item()), 1e-4)
    
    def test_niqe_y_uint8(self):
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img_np = (np.random.rand(*shape) * 255.).astype(np.uint8)
                img_torch = torch.from_numpy(img_np)

                niqe_bsr = calculate_niqe(img_np, 4, order, 'y')
                niqe_iqa = niqe(img_torch, 4, True, order)

                self.assertLess(abs(niqe_bsr - niqe_iqa.item()), 1e-4)


if __name__ == '__main__':
    unittest.main()
