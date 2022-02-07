import os
import sys
import unittest
import functools
sys.path.append(os.getcwd())

import numpy as np
import torch

from iqa.csc import rgb2ycbcr, rgb2y, ycbcr2rgb
from libs.basicsr_matlab_functions import rgb2ycbcr as rgb2ycbcr_bsr
from libs.basicsr_matlab_functions import ycbcr2rgb as ycbcr2rgb_bsr


IMAGE_SHAPES = (
    (128, 128, 3),
    (4, 128, 128, 3)
)


class CSCTestCase(unittest.TestCase):
    def _test_csc(self, fn_torch, fn_bsr):
        for shape in IMAGE_SHAPES:
            with self.subTest(shape=shape):
                img_in = np.random.rand(*shape).astype(np.float32)
                img_in_torch = torch.from_numpy(img_in)

                img_out = fn_bsr(img_in)
                img_out_torch = fn_torch(img_in_torch)

                if img_out.ndim != len(shape):
                    img_out = img_out[..., None]
                diff = np.abs(img_out - img_out_torch.numpy())
                # TODO: ycbcr2rgb precision
                self.assertLess(diff.max(), 1e-5)
                self.assertLess(diff.mean(), 1e-5)

    def test_rgb2ycbcr(self):
        self._test_csc(rgb2ycbcr, rgb2ycbcr_bsr)

    def test_ycbcr2rgb(self):
        self._test_csc(ycbcr2rgb, ycbcr2rgb_bsr)

    def test_rgb2y(self):
        rgb2y_bsr = functools.partial(rgb2ycbcr_bsr, y_only=True)
        self._test_csc(rgb2y, rgb2y_bsr)


if __name__ == '__main__':
    unittest.main()
