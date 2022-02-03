import os
import sys
import unittest
sys.path.append(os.getcwd())

import torch

from iqa.utils import reorder_image


ORDER_SHAPE = {
    'HW': (16, 16),
    'CHW': (1, 16, 16),
    'HWC': (16, 16, 1),
    'NHW': (1, 16, 16),
    'NCHW': (1, 1, 16, 16),
    'NHWC': (1, 16, 16, 1),
}

ORDER_SHAPE_C = {
    'CHW': (3, 16, 16),
    'HWC': (16, 16, 3),
    'NCHW': (1, 3, 16, 16),
    'NHWC': (1, 16, 16, 3),
}

ORDER_SHAPE_N = {
    'NHW': (4, 16, 16),
    'NCHW': (4, 1, 16, 16),
    'NHWC': (4, 16, 16, 1),
}

ORDER_SHAPE_NC = {
    'NCHW': (4, 3, 16, 16),
    'NHWC': (4, 16, 16, 3),
}


class TestCase(unittest.TestCase):
    def _test_order(self, order_shape: dict):
        for input_order, input_shape in order_shape.items():
            for output_order, output_shape in order_shape.items():
                with self.subTest(input_order=input_order, output_order=output_order):
                    src = torch.randn(input_shape)
                    dst = reorder_image(src, input_order, output_order)
                    self.assertEqual(tuple(dst.size()), output_shape)

    def test_reorder_all(self):
        self._test_order(ORDER_SHAPE)
    
    def test_reorder_c(self):
        self._test_order(ORDER_SHAPE_C)
    
    def test_reorder_n(self):
        self._test_order(ORDER_SHAPE_N)
    
    def test_reorder_nc(self):
        self._test_order(ORDER_SHAPE_NC)


if __name__ == '__main__':
    unittest.main()
