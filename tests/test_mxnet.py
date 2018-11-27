import unittest

import mxnet as mx

from common import gpu_test

class TestMxNet(unittest.TestCase):
    def test_array(self):
        x = mx.nd.array([[1, 2, 3], [4, 5, 6]])

        self.assertEqual((2, 3), x.shape)

    @gpu_test
    def test_gpu(self):
        # Use a GPU context.
        ctx = mx.gpu(device_id=0)
        x1 = mx.nd.array([[1, 5, 3], [6, 1, 2]], ctx)
        x2 = mx.nd.array([[5, 3], [2, 18], [3, 4]], ctx)
        # Matrix multiplication.
        x3 = mx.ndarray.linalg.gemm2(x1, x2)

        self.assertEqual('gpu', x1.context.device_type)
        self.assertEqual('gpu', x2.context.device_type)
        self.assertEqual('gpu', x3.context.device_type)
        self.assertEqual((2, 2), x3.shape)
