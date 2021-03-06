import time


class Profiler(object):
    def __init__(self):
        self._profile_data = {}
        self._running = {}

    def add(self, name, value):
        if name not in self._profile_data:
            self._profile_data[name] = 0
        self._profile_data[name] += value

    def start(self, name):
        if name in self._running:
            raise RuntimeError(f"profiling '{name}' already started")
        self._running[name] = time.time()

    def stop(self, name):
        if name not in self._running:
            raise RuntimeError(f"profiling '{name}' has not been started")
        dur = time.time() - self._running[name]
        del self._running[name]
        self.add(name, dur)

    def get_and_reset(self) -> str:
        ret = []
        for name, value in self.get_and_reset_dict().items():
            ret.append(f"{name}: {value}")
        return ", ".join(ret)

    def get_and_reset_dict(self) -> dict:
        if len(self._running) > 0:
            print('the following profiles are still running: {}'.format(", ".join(self._running.keys())))
        ret = self._profile_data.copy()
        self.__init__()
        return ret


def profiler_pipe(p):
    if p is None:
        return Profiler()
    t = type(p)
    if t == Profiler:
        return p
    raise RuntimeError(f"unknown profile type '{t}'")
