import unittest

from lesforge.utils.yamlish import parse


class TestYamlish(unittest.TestCase):
    def test_nested_map(self):
        d = parse("a:\n  b: 1\n  c: hello\n")
        self.assertEqual(d["a"]["b"], 1)
        self.assertEqual(d["a"]["c"], "hello")

    def test_block_list(self):
        d = parse("items:\n  - x\n  - y\n")
        self.assertEqual(d["items"], ["x", "y"])

    def test_inline_comment(self):
        d = parse("name: mock  # backend\n")
        self.assertEqual(d["name"], "mock")

    def test_inline_list(self):
        d = parse("dims: [8, 16, 32]\n")
        self.assertEqual(d["dims"], [8, 16, 32])


if __name__ == "__main__":
    unittest.main()
