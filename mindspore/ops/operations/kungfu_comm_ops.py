"""kungfu comm_ops"""
from ..._c_expression import (kungfu_current_cluster_size, kungfu_current_rank,
                              kungfu_finalize, kungfu_init,
                              kungfu_nccl_finalize, kungfu_nccl_init)
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, prim_attr_register
from .comm_ops import ReduceOp


class KungFuAllReduce(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, op=ReduceOp.SUM):
        self.op = op
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
        return (mstype.bool_, mstype.bool_)
