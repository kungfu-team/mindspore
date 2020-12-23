import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops


def sync_net_parameters(network: ms.nn.Cell):
    print('BEGIN sync_net_parameters')
    broadcast = kfops.KungFuBroadcast()
    network.init_parameters_data()
    for _name, param in network.parameters_and_names():
        x = ms.Tensor(param.data)
        x = broadcast(x)
        param.set_data(x)
    print('END sync_net_parameters')


# TODO: use KungFu AP interface
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
