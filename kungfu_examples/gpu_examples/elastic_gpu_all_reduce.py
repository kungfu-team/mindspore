import argparse
import os
import time

import mindspore as ms
import numpy as np
from mindspore._c_expression import (kungfu_current_cluster_size,
                                     kungfu_current_rank, kungfu_finalize,
                                     kungfu_init, kungfu_nccl_finalize,
                                     kungfu_nccl_init)
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.ops.operations.kungfu_comm_ops import (KungFuAllReduce,
                                                      KungFuBroadcast,
                                                      KungFuResize)

resnet50 = [
    1000, 2048000, 2048, 2048, 2048, 1048576, 512, 512, 512, 2359296, 512, 512,
    512, 1048576, 2048, 2048, 2048, 1048576, 512, 512, 512, 2359296, 512, 512,
    512, 1048576, 2048, 2048, 2048, 2048, 2048, 2048, 1048576, 512, 512, 512,
    2097152, 2359296, 512, 512, 512, 524288, 1024, 1024, 1024, 262144, 256,
    256, 256, 589824, 256, 256, 256, 262144, 1024, 1024, 1024, 262144, 256,
    256, 256, 589824, 256, 256, 256, 262144, 1024, 1024, 1024, 262144, 256,
    256, 256, 589824, 256, 256, 256, 262144, 1024, 1024, 1024, 262144, 256,
    256, 256, 589824, 256, 256, 256, 262144, 1024, 1024, 1024, 262144, 256,
    256, 256, 589824, 256, 256, 256, 262144, 1024, 1024, 1024, 1024, 1024,
    1024, 262144, 524288, 256, 256, 256, 589824, 256, 256, 256, 131072, 512,
    512, 512, 65536, 128, 128, 128, 147456, 128, 128, 128, 65536, 512, 512,
    512, 65536, 128, 128, 128, 147456, 128, 128, 128, 65536, 512, 512, 512,
    65536, 128, 128, 128, 147456, 128, 128, 128, 65536, 512, 512, 512, 512,
    512, 512, 65536, 131072, 128, 128, 128, 147456, 128, 128, 128, 32768, 256,
    256, 256, 16384, 64, 64, 64, 36864, 64, 64, 64, 16384, 256, 256, 256,
    16384, 64, 64, 64, 36864, 64, 64, 64, 16384, 256, 256, 256, 256, 256, 256,
    16384, 16384, 64, 64, 64, 36864, 64, 64, 64, 4096, 64, 64, 64, 9408
]

vgg16 = [
    1728, 64, 36864, 64, 73728, 128, 147456, 128, 294912, 256, 589824, 256,
    589824, 256, 1179648, 512, 2359296, 512, 2359296, 512, 2359296, 512,
    2359296, 512, 2359296, 512, 102760448, 4096, 16777216, 4096, 4096000, 1000
]

model_grad_sizes = {
    'one': [1024],
    'resnet50': resnet50,
    'vgg16': vgg16,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',
                   type=str,
                   default="CPU",
                   choices=['Ascend', 'GPU', 'CPU'])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--model',
                   type=str,
                   default='resnet50',
                   choices=model_grad_sizes.keys())
    args = p.parse_args()
    return args


def sync_step(step, sync_op):
    step_tensor = ms.Tensor(step, dtype=ms.float32)
    cluster_step = sync_op(step_tensor)
    new_step = int(cluster_step.asnumpy())
    print('sync step %d -> %d' % (step, new_step))
    return new_step


def main():
    args = parse_args()
    grad_sizes = model_grad_sizes[args.model]

    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)

    schedule = {
        3: 2,
        6: 3,
        9: 4,
        12: 1,
    }

    kungfu_init()
    kungfu_nccl_init()

    all_reduce = KungFuAllReduce()
    all_reduce_max = KungFuAllReduce(op=ReduceOp.MAX)
    resize = KungFuResize()

    xs = [
        ms.Tensor(np.array([1.0] * size).astype(np.float32))
        for size in grad_sizes
    ]

    step = ms.Tensor(0)
    print(step)

    step = 0
    while True:
        step = sync_step(step, all_reduce_max)
        print('step: %d' % (step))
        t0 = time.time()
        ys = [all_reduce(x) for x in xs]
        t1 = time.time()
        d = t1 - t0

        if step in schedule:
            new_size = ms.Tensor(schedule[step], dtype=ms.uint32)
            print('step=%d, will resize to %d' % (step, schedule[step]))
            changed, detached = resize(new_size)
            print(changed)
            print(detached)
            if changed:
                need_sync = True
            if detached:
                break

        step += 1
        if step > args.steps:
            break
    kungfu_nccl_finalize()
    kungfu_finalize()


main()
