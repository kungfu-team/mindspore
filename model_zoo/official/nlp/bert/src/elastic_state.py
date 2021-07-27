class ElasticState:
    def __init__(self, max_progress=None):
        self._progress = 0
        self._max_progress = max_progress
        self._synced = False
        self._stop_reason = None

        import pystdml as ml
        self._sess = ml.init_elastic()

    def begin(self):
        should_sync = not self._synced
        if should_sync:
            new_progress = self._sess.all_reduce_max(self._progress)
            self._step = new_progress
            self._synced = True
        return should_sync

    def end(self, progress=1):
        self._progress += progress
        if self._max_progress:
            if self._progress >= self._max_progress:
                self._stop_reason = 'finished'
                return

        result = self._sess.resize()
        if result.changed:
            if result.detached:
                self._stop_reason = 'detached'
                return
            self._synced = False

    def stopped(self):
        return self._stop_reason is not None

    def stop_reason(self):
        return self._stop_reason


class ElasticContext:
    def __init__(self, elastic_state):
        self._elastic_state = elastic_state

    def __enter__(self):
        return self._elastic_state.begin()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._elastic_state.end()


import mindspore as ms


class ElasticCallback(ms.train.callback.Callback):
    def __init__(self, elastic_state):
        self._elastic_state = elastic_state

    def begin(self, run_context):
        pass
        print('ElasticCallback::begin')

    def epoch_begin(self, run_context):
        pass
        print('ElasticCallback::epoch_begin')

    def epoch_end(self, run_context):
        pass
        print('ElasticCallback::epoch_end')

    def step_begin(self, run_context):
        print('ElasticCallback::step_begin')
        should_sync = self._elastic_state.begin()
        if should_sync:
            print('TODO: sync states')
        print('progress: %d' % (self._elastic_state._progress))

    def step_end(self, run_context):
        print('ElasticCallback::step_end')
        self._elastic_state.end()
        if self._elastic_state.stopped():
            print('_elastic_state stopped, requesting run_context to stop')
            run_context.request_stop()

    def end(self, run_context):
        pass
        print('StopCallback::end')
