import argparse
import time

import mindspore as ms
import numpy as np

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
    p.add_argument('--steps', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)

    all_reduce = kf.AllReduce()

    xs = [
        ms.Tensor(np.array([1.0] * size).astype(np.float32))
        for size in grad_sizes
    ]

    data_size = sum(grad_sizes) * 4
    cluster_size = 4  # TODO: get from API
    multiplier = 4 * (cluster_size - 1)
    Gi = 1024 * 1024 * 1024

    for i in range(args.steps):
        t0 = time.time()
        ys = [all_reduce(x) for x in xs]
        t1 = time.time()
        d = t1 - t0
        rate = float(data_size) * multiplier / Gi / d
        print('took %.3fms, data rate: %.3fGiB/s' % (d * 1e3, rate))


main()
