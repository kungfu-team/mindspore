import mindspore as ms


class KungFuAllReduce(ms.ops.PrimitiveWithInfer):
    @ms.ops.prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        return x


class KungFuBroadcast(ms.ops.PrimitiveWithInfer):
    @ms.ops.prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        return x


class KungFuResize(ms.ops.PrimitiveWithInfer):
    @ms.ops.prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['n'], outputs=['changed', 'detached'])

    def infer_shape(self, *args):
        return ([], [])

    def infer_dtype(self, *args):
        return (ms.bool_, ms.bool_)
