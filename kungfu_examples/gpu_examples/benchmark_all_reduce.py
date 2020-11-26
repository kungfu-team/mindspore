import argparse
import os
import time

import mindspore as ms
import numpy as np
from mindspore.communication.management import get_group_size, get_rank, init

import kungfu_mindspore_ops as kf

grad_sizes = [
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',
                   type=str,
                   default="CPU",
                   choices=['Ascend', 'GPU', 'CPU'])
    p.add_argument('--warmup-steps', type=int, default=3)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--collective',
                   type=str,
                   default='mindspore',
                   choices=['mindspore', 'kungfu'])
    return p.parse_args()


def print_env():
    # for e in sorted(os.environ):
    #     print(e)
    print(os.getenv('LD_LIBRARY_PATH'))


def parse_kungfu_size():
    val = os.getenv('KUNGFU_INIT_PEERS')
    return len(val.split(','))


def parse_kungfu_port():
    val = os.getenv('KUNGFU_SELF_SPEC')
    print(val)
    return int(val.split(':')[1])


def main():
    args = parse_args()
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)

    if args.collective == 'mindspore':
        init()
        cluster_size = get_group_size()
        rank = get_rank()
    else:
        cluster_size = parse_kungfu_size()
        rank = parse_kungfu_port() - 10000

    print('rank: %d, size: %d' % (rank, cluster_size))

    if args.collective == 'mindspore':
        all_reduce = ms.ops.operations.AllReduce()
    elif args.collective == 'kungfu':
        all_reduce = kf.AllReduce()
    else:
        raise RuntimeError('invalid collective')

    xs = [
        ms.Tensor(np.array([1.0] * size).astype(np.float32))
        for size in grad_sizes
    ]

    data_size = sum(grad_sizes) * 4  # 1 float is 4 bytes
    multiplier = 4 * (cluster_size - 1)
    Gi = 1024 * 1024 * 1024

    def run_stage(name, steps):
        for i in range(steps):
            t0 = time.time()
            ys = [all_reduce(x) for x in xs]
            t1 = time.time()
            d = t1 - t0
            rate = float(data_size) * multiplier / Gi / d
            if rank == 0:
                print('%s %d took %.3fms, data rate: %.3fGiB/s' %
                      (name, i + 1, d * 1e3, rate))

    run_stage('warmup', args.warmup_steps)
    run_stage('step', args.steps)


main()
