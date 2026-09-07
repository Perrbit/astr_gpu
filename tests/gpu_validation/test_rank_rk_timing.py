import unittest

from summarize_rank_rk_timing import slowest_rank_samples


class RankTimingTests(unittest.TestCase):
    def test_stepwise_maximum_handles_unordered_ranks(self):
        log = '\n'.join([
            'ASTR_GPU_RANK_RK_TIMING 1 1 0.1 0.4 0.5',
            'ASTR_GPU_RANK_RK_TIMING 0 0 0.1 0.2 0.3',
            'ASTR_GPU_RANK_RK_TIMING 1 0 0.1 0.1 0.2',
            'ASTR_GPU_RANK_RK_TIMING 0 1 0.1 0.3 0.4',
        ])
        self.assertEqual(slowest_rank_samples(log, 2, 2), [0.3, 0.5])

    def test_incomplete_duplicate_or_nonfinite_records_fail(self):
        valid = 'ASTR_GPU_RANK_RK_TIMING 0 0 0.1 0.2 0.3'
        for text, ranks in [('', 1), (valid, 2), (valid+'\n'+valid, 1),
                            (valid.replace('0.3', 'nan'), 1),
                            (valid.replace('0.3', '0.9'), 1)]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                slowest_rank_samples(text, ranks, 1)
