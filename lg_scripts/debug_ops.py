from mindspore.ops.primitive import PrimitiveWithInfer, prim_attr_register


class KungFuLogTensor(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        return x
