import unittest

from lesforge.core.registry import Registry


class TestRegistry(unittest.TestCase):
    def test_register_get(self):
        reg = Registry("t")
        @reg.register("a")
        def f():
            return 1
        self.assertEqual(reg.get("a")(), 1)
        self.assertEqual(reg.keys(), ["a"])


if __name__ == "__main__":
    unittest.main()
