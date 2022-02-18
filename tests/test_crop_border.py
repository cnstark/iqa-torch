import os
import sys
import unittest
sys.path.append(os.getcwd())

import torch

from iqa.utils import crop_border


TEST_SET = (
    ('HW', (16, 16), 2, (12, 12)),
    ('CHW', (3, 16, 16), 2, (3, 12, 12)),
    ('HWC', (16, 16, 3), 2, (12, 12, 3)),
    ('NHW', (4, 16, 16), 2, (4, 12, 12)),
    ('NCHW', (4, 3, 16, 16), 2, (4, 3, 12, 12)),
    ('NHWC', (4, 16, 16, 3), 2, (4, 12, 12, 3)),
)


class TestCropBorder(unittest.TestCase):
    def test_crop_border(self):
        for order, input_shape, border_crop_size, output_shape in TEST_SET:
            with self.subTest(order=order, input_shape=input_shape, border_crop_size=border_crop_size):
                src = torch.randn(input_shape)
                dst = crop_border(src, border_crop_size, order)
                self.assertEqual(tuple(dst.size()), output_shape)


if __name__ == '__main__':
    unittest.main()
