"""tests/test_data_generator.py

DataGenerator のユニットテスト。
"""

import unittest
import numpy as np
from kf_auto_tuning.config import SystemConfig
from kf_auto_tuning.data_generator import DataGenerator


def _make_config(steps=50, seed=0):
    return SystemConfig(
        F=np.array([[1, 0.1], [0, 1]]),
        G=np.array([[0], [1]]),
        H=np.array([[1, 0]]),
        Q_true=np.diag([0.1, 0.1]),  # 非ゼロにしてシードによる差異を確認できるようにする
        R_true=0.5,
        x0=np.array([0.0, 1.0]),
        P0=np.eye(2),
        u=0,
        steps=steps,
        nx=2,
        nz=1,
        data_seed=seed,
    )


class TestDataGenerator(unittest.TestCase):
    def test_output_shapes(self):
        cfg = _make_config(steps=100)
        gen = DataGenerator(cfg)
        states, obs = gen.generate()
        self.assertEqual(states.shape, (100, 2))
        self.assertEqual(obs.shape, (100, 1))

    def test_reproducibility_with_seed(self):
        cfg = _make_config(seed=42)
        s1, o1 = DataGenerator(cfg).generate()
        s2, o2 = DataGenerator(cfg).generate()
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        cfg0 = _make_config(seed=0)
        cfg1 = _make_config(seed=1)
        s0, _ = DataGenerator(cfg0).generate()
        s1, _ = DataGenerator(cfg1).generate()
        self.assertFalse(np.allclose(s0, s1))

    def test_no_seed_runs_without_error(self):
        cfg = _make_config()
        cfg.data_seed = None
        states, obs = DataGenerator(cfg).generate()
        self.assertEqual(len(states), 50)
