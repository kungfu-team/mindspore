KungFu Mindspore
======

KungFu is a communication library designed for machine learning.
It provides low level collective communicative APIs (such as `AllReduce`) which can be used by different ML frameworks,
e.g. Tensorflow, PyTorch and Mindspore.

To enable KungFu elastic training in mindspore, we provided kungfu operator as part of mindspore extension.
User should implement customized optimizer and callback (extending the standard mindspore `Optimizer` and `Callback`).

We also provide some commonly used optimizers as part of mindspore extension.

## Build

Currently we build and install kungfu in the source tree of mindspore.
User can use the script <../install-kungfu.sh> to download and build kungfu.

## Implementing Customized Optimizer and Callback

1. Implement distributed optimizer

```python
import mindspore as ms
from mindspore.ops import composite as C
from mindspore.ops.operations.kungfu_comm_ops import KungFuAllReduce

class KungFuMomentum(ms.nn.Momentum):
    def __init__(self, *args, **kwargs):
        super(KungFuMomentum, self).__init__(*args, **kwargs)
        self.map_ = C.Map()
        self.all_reduce = KungFuAllReduce()

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        return super(KungFuMomentum, self).construct(gradients)
```


2. Implement Elastic Callback

```python
import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops


def sync_net_parameters(network: ms.nn.Cell):
    broadcast = kfops.KungFuBroadcast()
    network.init_parameters_data()
    for _name, param in network.parameters_and_names():
        x = ms.Tensor(param.data)
        x = broadcast(x)
        param.set_data(x)


class KungFuElasticCallback(ms.train.callback.Callback):
    def __init__(self, schedule):
        self.schedule = schedule
        self.need_sync = True

        self.broadcast = kfops.KungFuBroadcast()
        # TODO: use integer
        self._kungfu_global_step = ms.Tensor(0.0, dtype=ms.float32)

        self.resize = kfops.KungFuResize()

    def _advance_step(self):
        old_step = int(self._kungfu_global_step.asnumpy())
        self._kungfu_global_step = ms.Tensor(old_step + 1.0, dtype=ms.float32)

    def _sync_step(self):
        old_step = int(self._kungfu_global_step.asnumpy())
        self._kungfu_global_step = self.broadcast(self._kungfu_global_step)
        new_step = int(self._kungfu_global_step.asnumpy())
        print('sync step %d -> %d' % (old_step, new_step))

    def begin(self, run_context):
        pass

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context):
        pass

    def step_begin(self, run_context):
        self._advance_step()

        if self.need_sync:
            self._sync_step()
            cb_params = run_context.original_args()
            sync_net_parameters(cb_params.train_network)
            self.need_sync = False

    def step_end(self, run_context):
        step = int(self._kungfu_global_step.asnumpy())
        if step in self.schedule:
            new_size = self.schedule[step]
            new_size_tensor = ms.Tensor(new_size, dtype=ms.uint32)
            print('calling resize with %d at step %d' % (new_size, step))
            changed, detached = self.resize(new_size_tensor)
            if changed:
                self.need_sync = True
            if detached:
                print('detached, requesting stop')
                run_context.request_stop()
                print('requested stop')

    def end(self, run_context):
        print('stopped')
```


## Enabling Elastic Training

To enable elastic training in mindspore, user need to

1. replace their original optimizer by the KungFu distributed optimizer.

```python
opt = KungFuMomentum(group_params,
                     lr,
                     config.momentum,
                     loss_scale=config.loss_scale)
```

2. append KungFuElasticCallback to the original callback list

```python
kungfu_elastic_callback = KungFuElasticCallback(schedule)

model.train(train_epoch,
            dataset,
            callbacks=cb + [kungfu_elastic_callback],
            sink_size=dataset.get_dataset_size(),
            dataset_sink_mode=False)
```

3. launch the training program with `kungfu-run`

```bash
KUNGFU_LIB_PATH=$ROOT/third_party/kungfu/lib

export LD_LIBRARY_PATH=$KUNGFU_LIB_PATH:$ROOT/mindspore/lib:$ROOT/build/mindspore/_deps/ompi-src/ompi/.libs

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 1

    echo -w  # enable elastic mode
    echo -builtin-config-port 9100
    echo -config-server http://127.0.0.1:9100/config
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

kungfu_run /usr/bin/python3.7 train.py
```

See <./lenet-elastic/run-elastic.sh> for a full example.
