# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Lenet Tutorial
The sample can be run on CPU, GPU and Ascend 910 AI processor.
"""
import argparse
import os

import mindspore as ms
from mindspore.common.initializer import Normal
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint)
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from download import download_dataset
from kungfu_mindspore_optimizer import build_optimizer
from mnist import create_dataset


def parse_args():
    p = argparse.ArgumentParser(description='MindSpore LeNet Example')
    p.add_argument(
        '--device',
        type=str,
        default="CPU",
        choices=['Ascend', 'GPU', 'CPU'],
        help='device where the code will be implemented (default: CPU)')
    p.add_argument('--data-dir', type=str, default="MNIST_Data")
    p.add_argument('--epoch-size', type=int, default=1)
    p.add_argument('--repeat-size', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--use-kungfu', action='store_true', default=False)
    return p.parse_args()


class LeNet5(ms.nn.Cell):
    """Lenet network structure."""

    # define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = ms.nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = ms.nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = ms.nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = ms.nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = ms.nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = ms.nn.ReLU()
        self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = ms.nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_net(model, epoch_size, mnist_path, repeat_size, ckpoint_cb,
              sink_mode):
    """Define the training method."""
    print("============== Starting Training ==============")
    # load training dataset
    ds_train = create_dataset(os.path.join(mnist_path, "train"), 32,
                              repeat_size)
    callbacks = [
        # ckpoint_cb,
        LossMonitor(),
    ]
    model.train(epoch_size,
                ds_train,
                callbacks=callbacks,
                dataset_sink_mode=sink_mode)


def test_net(network, model, mnist_path):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    # load the saved model for evaluation
    param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
    # load parameter to the network
    load_param_into_net(network, param_dict)
    # load testing dataset
    ds_eval = create_dataset(os.path.join(mnist_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))


def main():
    args = parse_args()

    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)
    dataset_sink_mode = not args.device == "CPU"

    download_dataset(args.data_dir)

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # create the network
    network = LeNet5()
    # define the optimizer
    net_opt = build_optimizer(args, network)
    config_ck = CheckpointConfig(save_checkpoint_steps=1875,
                                 keep_checkpoint_max=10)
    # save the network model and parameters for subsequence fine-tuning
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    # group layers into an object with training and evaluation features
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    train_net(model, args.epoch_size, args.data_dir, args.repeat_size,
              ckpoint_cb, dataset_sink_mode)
    test_net(network, model, args.data_dir)


if __name__ == "__main__":
    main()
