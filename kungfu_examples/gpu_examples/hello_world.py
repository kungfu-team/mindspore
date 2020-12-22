#!/usr/bin/env python3.7

import argparse

import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kf
import numpy as np
from mindspore._c_expression import (kungfu_current_cluster_size,
                                     kungfu_current_rank, kungfu_finalize,
                                     kungfu_init, kungfu_nccl_finalize,
                                     kungfu_nccl_init)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',
                   type=str,
                   default="CPU",
                   choices=['Ascend', 'GPU', 'CPU'])
    return p.parse_args()


def main():
    args = parse_args()
    kungfu_init()
    kungfu_nccl_init()
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)

    all_reduce = kf.KungFuAllReduce()

    x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    print(x)
    y = all_reduce(x)
    print(y)
    kungfu_nccl_finalize()
    kungfu_finalize()


main()
