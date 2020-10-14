#!/usr/bin/env python3.7

import numpy as np

import mindspore as ms

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")


class Mul(ms.nn.Cell):
    def __init__(self):
        super(Mul, self).__init__()
        self.mul = ms.ops.operations.Mul()

    def construct(self, x, y):
        return self.mul(x, y)


def test_mul():
    x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

    mul = Mul()
    print(mul(x, y))


class AllReduce(ms.nn.Cell):
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce = ms.ops.operations.KungFuAllReduce()

    def construct(self, x):
        return self.all_reduce(x)


def test_all_reduce():
    x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))

    all_reduce = AllReduce()
    y = all_reduce(x)
    print(y)


test_mul()
# test_all_reduce()
