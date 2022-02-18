import os
import sys
import unittest

sys.path.append(os.getcwd())

import torch

from lpips import LPIPS as LPIPS_Official
from iqa import lpips, LPIPS


IMAGE_SHAPES = (('NCHW', (1, 3, 128, 128)),)


class LPIPSTestCase(unittest.TestCase):
    def test_lpips_func(self):
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img1 = torch.rand(shape)
                img2 = torch.rand(shape)

                lpips_iqa = lpips(img1, img2)
                lpips_official = LPIPS_Official()(img1, img2)

                self.assertLess((lpips_iqa - lpips_official).item(), 1e-8)

    def test_lpips_module(self):
        _lpips = LPIPS()
        _lpips_official = LPIPS_Official()
        for order, shape in IMAGE_SHAPES:
            with self.subTest(order=order, shape=shape):
                img1 = torch.rand(shape)
                img2 = torch.rand(shape)

                lpips_iqa = _lpips(img1, img2)
                lpips_official = _lpips_official(img1, img2)

                self.assertLess((lpips_iqa - lpips_official).item(), 1e-8)


if __name__ == '__main__':
    unittest.main()
