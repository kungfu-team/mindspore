import argparse

import mindspore as ms
import numpy as np
from mindspore._c_expression import (kungfu_current_rank, kungfu_finalize,
                                     kungfu_init, kungfu_nccl_finalize,
                                     kungfu_nccl_init)
from mindspore.ops.operations.kungfu_comm_ops import (KungFuAllReduce,
                                                      KungFuBroadcast)

dtype_map = {
    'i32': np.int32,
    'f32': np.float32,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='CPU', choices=['GPU', 'CPU'])
    p.add_argument('--dtype', type=str, default='f32', choices=['i32', 'f32'])
    return p.parse_args()


def main():
    args = parse_args()
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)

    kungfu_init()
    if args.device == 'GPU':
        kungfu_nccl_init()

    broadcast = KungFuBroadcast()

    size = 10
    value = kungfu_current_rank()
    dtype = dtype_map[args.dtype]

    x = ms.Tensor(np.array([value] * size).astype(dtype))
    print('x=%s' % (x))
    y = broadcast(x)
    print('y=%s' % (y))

    if args.device == 'GPU':
        kungfu_nccl_finalize()
    kungfu_finalize()


main()
