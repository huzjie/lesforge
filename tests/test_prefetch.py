import unittest

from lesforge.scheduler.prefetch import PrefetchPipeline
from lesforge.scheduler.offload import OffloadStore
from lesforge.scheduler.tier import ResidencyTable


class TestPrefetch(unittest.TestCase):
    def test_issue_and_drain(self):
        p = PrefetchPipeline(depth=2, offload=OffloadStore())
        r = ResidencyTable()
        p.issue(0, 1)
        p.issue(0, 2)
        self.assertFalse(p.can_issue())
        admitted = p.drain(r)
        self.assertEqual(len(admitted), 2)
        self.assertTrue(p.can_issue())
        self.assertTrue(r.is_resident(0, 1))
        self.assertTrue(r.is_resident(0, 2))


if __name__ == "__main__":
    unittest.main()
