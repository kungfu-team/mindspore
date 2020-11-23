#!/usr/bin/env python3.7

import argparse

import mindspore as ms
import numpy as np

import kungfu_mindspore_ops as kf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',
                   type=str,
                   default="CPU",
                   choices=['Ascend', 'GPU', 'CPU'])
    return p.parse_args()


def main():
    args = parse_args()
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)

    all_reduce = kf.AllReduce()

    x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    print(x)
    y = all_reduce(x)
    print(y)


main()
