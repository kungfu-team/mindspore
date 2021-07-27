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

