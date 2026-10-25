"""Bounded controller tasks with isolated contexts and nonblocking cancellation."""
from concurrent.futures import Future, CancelledError
from contextvars import copy_context
from contextlib import nullcontext
from queue import Queue, Empty
from threading import Thread, Event


class AssignmentClient:
    """Fence role results before tools/source changes after controller cancellation."""
    def __init__(self, client, cancelled):
        self.client, self.cancelled = client, cancelled

    def __getattr__(self, name):
        return getattr(self.client,name)

    def generate(self, *args, **kwargs):
        if self.cancelled.is_set(): raise CancelledError('Controller stopped')
        options={key:self.__dict__[key] for key in ('deadline','max_steps') if key in self.__dict__}
        scope=self.client.assignment_scope(**options) if hasattr(self.client,'assignment_scope') else nullcontext()
        with scope:
            result=self.client.generate(*args,**kwargs)
        if self.cancelled.is_set(): raise CancelledError('Controller stopped')
        return result


class PreparationPool:
    def __init__(self, workers):
        self.queue = Queue()
        self.stopped = Event()
        for index in range(workers):
            Thread(target=self._work, name=f'candidate-prepare-{index}', daemon=True).start()

    def submit(self, function, *args, **kwargs):
        future = Future()
        if self.stopped.is_set():
            future.cancel()
        else:
            self.queue.put((future, copy_context(), function, args, kwargs))
        return future

    def _work(self):
        while not self.stopped.is_set():
            try:
                future, context, function, args, kwargs = self.queue.get(timeout=.1)
            except Empty:
                continue
            if self.stopped.is_set():
                future.cancel()
                continue
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(context.run(function, *args, **kwargs))
                except BaseException as error:
                    future.set_exception(error)

    def close(self):
        self.stopped.set()
        while True:
            try:
                self.queue.get_nowait()[0].cancel()
            except Empty:
                break
