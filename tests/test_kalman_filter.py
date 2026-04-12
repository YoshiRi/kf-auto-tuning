"""tests/test_kalman_filter.py

KalmanFilter のユニットテスト。
"""

import unittest
import numpy as np
from kf_auto_tuning.config import SystemConfig
from kf_auto_tuning.data_generator import DataGenerator
from kf_auto_tuning.kalman_filter import KalmanFilter


def _make_config(steps=100, seed=42):
    return SystemConfig(
        F=np.array([[1, 0.1], [0, 1]]),
        G=np.array([[0], [1]]),
        H=np.array([[1, 0]]),
        Q_true=np.diag([0.01, 0.01]),
        R_true=0.5,
        x0=np.array([0.0, 1.0]),
        P0=np.eye(2),
        u=0,
        steps=steps,
        nx=2,
        nz=1,
        data_seed=seed,
    )


class TestKalmanFilter(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_config()
        gen = DataGenerator(self.cfg)
        self.true_states, self.obs = gen.generate()
        self.kf = KalmanFilter(self.cfg)
        self.Q = np.diag([0.01])
        self.R = 0.5

    def test_estimate_shape(self):
        estimates, _ = self.kf.run(self.Q, self.R, self.obs)
        self.assertEqual(estimates.shape, (100, 2))

    def test_metrics_keys(self):
        _, metrics = self.kf.run(self.Q, self.R, self.obs)
        for key in ("mean_nees", "var_nees", "mean_nis", "var_nis"):
            self.assertIn(key, metrics)

    def test_nis_positive(self):
        _, metrics = self.kf.run(self.Q, self.R, self.obs)
        self.assertGreater(metrics["mean_nis"], 0)

    def test_nees_computed_with_true_states(self):
        _, metrics = self.kf.run(self.Q, self.R, self.obs, true_states=self.true_states)
        self.assertGreater(metrics["mean_nees"], 0)

    def test_nees_zero_without_true_states(self):
        _, metrics = self.kf.run(self.Q, self.R, self.obs, true_states=None)
        self.assertEqual(metrics["mean_nees"], 0)
        self.assertEqual(metrics["var_nees"], 0)

    def test_larger_Q_increases_nis(self):
        """Q を過大に設定すると NIS が理論値 1.0 から大きく外れることを確認する。"""
        _, m_good = self.kf.run(np.diag([0.01]), 0.5, self.obs)
        _, m_bad = self.kf.run(np.diag([100.0]), 0.5, self.obs)
        # 理論値 nz=1 からの乖離は bad のほうが大きいはず（絶対値で評価）
        dev_good = abs(m_good["mean_nis"] - 1.0)
        dev_bad = abs(m_bad["mean_nis"] - 1.0)
        self.assertGreater(dev_bad, dev_good)
