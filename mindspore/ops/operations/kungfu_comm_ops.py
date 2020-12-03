"""kungfu comm_ops"""
from ..._c_expression import (kungfu_finalize, kungfu_init,
                              kungfu_nccl_finalize, kungfu_nccl_init)
from ..primitive import PrimitiveWithInfer, prim_attr_register


class KungFuAllReduce(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        return x


class KungFuBroadcast(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        return x


class KungFuResize(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['n'], outputs=['changed', 'detached'])

    def infer_shape(self, *args):
        return ([], [])

    def infer_dtype(self, *args):
        return (ms.bool_, ms.bool_)
