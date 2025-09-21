import threading

class ResultThread(threading.Thread):
    """带返回值的线程类"""
    def __init__(self, target, args=()):
        super().__init__(target=target, args=args)
        self.result = None

    def run(self):
        self.result = self._target(*self._args)