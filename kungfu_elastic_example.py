#!/usr/bin/env python3.7

import argparse

import numpy as np

import mindspore as ms
from kungfu_ops import KungFuAllReduce, KungFuBroadcast, KungFuResize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max-step', type=int, default='100')
    return p.parse_args()


class AllReduce(ms.nn.Cell):
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce = KungFuAllReduce()

    def construct(self, x):
        return self.all_reduce(x)


class Broadcast(ms.nn.Cell):
    def __init__(self):
        super(Broadcast, self).__init__()
        self.broadcast = KungFuBroadcast()

    def construct(self, x):
        return self.broadcast(x)


class Resize(ms.nn.Cell):
    def __init__(self):
        super(Resize, self).__init__()
        self.resize = KungFuResize()

    def construct(self, n):
        return self.resize(n)


def elastic_run(max_step, schedule):
    broadcast = Broadcast()
    all_reduce = AllReduce()
    resize = Resize()

    need_sync = True
    step = 0
    while True:
        if need_sync:
            step_tensor = ms.Tensor(step, dtype=ms.float32)
            cluster_step = broadcast(step_tensor)
            new_step = int(cluster_step.asnumpy())
            print('sync step %d -> %d' % (step, new_step))
            step = new_step
            need_sync = False

        x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
        y = all_reduce(x)
        print('step %d, y=%s' % (step, y))

        if step in schedule:
            new_size = ms.Tensor(schedule[step], dtype=ms.uint32)
            changed, keep = resize(new_size)
            print("changed: ", changed)
            print("keep: ", keep)
            if changed:
                need_sync = True
            if not keep:
                break
        step += 1
        if step >= max_step:
            break


def main():
    args = parse_args()
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")
    schedule = {
        10: 2,
        20: 3,
        30: 4,
        40: 1,
        50: 2,
        60: 3,
        70: 4,
        80: 1,
    }
    elastic_run(args.max_step, schedule)


main()
