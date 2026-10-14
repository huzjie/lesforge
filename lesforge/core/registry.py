"""Tiny registry used for backends and benchmarks."""


class Registry:
    def __init__(self, name):
        self.name = name
        self._items = {}

    def register(self, key):
        def deco(fn):
            self._items[key] = fn
            return fn
        return deco

    def get(self, key):
        return self._items.get(key)

    def keys(self):
        return list(self._items.keys())


BACKENDS = Registry("backends")
BENCHMARKS = Registry("benchmarks")
