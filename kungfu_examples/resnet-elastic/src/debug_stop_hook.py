import mindspore as ms


class DebugStopHook(ms.train.callback.Callback):
    def __init__(self, stop_after=1):
        self.stop_after = stop_after
        self.step = 0

    def begin(self, run_context):
        pass

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context):
        pass

    def step_begin(self, run_context):
        self.step += 1

    def step_end(self, run_context):
        if self.step >= self.stop_after:
            run_context.request_stop()
            print('requested stop')

    def end(self, run_context):
        print('stopped')
