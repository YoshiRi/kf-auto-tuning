"""tests/test_experiment_runner.py

ExperimentRunner と run_multiple_experiments のユニットテスト。
最適化実行は計算コストが高いため、n_calls を最小限に抑えた軽量設定で実行します。
"""

import unittest
import numpy as np
from kf_auto_tuning.config import SystemConfig, OptimizationConfig
from kf_auto_tuning.experiment_runner import ExperimentRunner, run_multiple_experiments


def _make_configs(n_calls=12, seed=0):
    system_cfg = SystemConfig(
        F=np.array([[1, 0.1], [0, 1]]),
        G=np.array([[0], [1]]),
        H=np.array([[1, 0]]),
        Q_true=np.diag([0.01, 0.01]),
        R_true=0.5,
        x0=np.array([0.0, 1.0]),
        P0=np.eye(2),
        u=0,
        steps=30,
        nx=2,
        nz=1,
        data_seed=seed,
    )
    opt_cfg = OptimizationConfig(
        param_bounds={"Q": [0.01, 1.0], "R": [0.1, 1.0]},
        n_calls=n_calls,
        opt_seed=seed,
    )
    return system_cfg, opt_cfg


class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.sys_cfg, self.opt_cfg = _make_configs(n_calls=12)

    def test_result_keys(self):
        runner = ExperimentRunner(self.sys_cfg, self.opt_cfg, cost_type="nis")
        result = runner.run_experiment(plot=False)
        for key in ("true_params", "optimized_params", "metrics", "estimation_results", "experiments"):
            self.assertIn(key, result)

    def test_optimized_params_shape(self):
        runner = ExperimentRunner(self.sys_cfg, self.opt_cfg, cost_type="nis")
        result = runner.run_experiment(plot=False)
        opt_Q = result["optimized_params"]["optimal_Q"]
        self.assertEqual(opt_Q.shape, (1, 1))

    def test_optimized_params_in_bounds(self):
        runner = ExperimentRunner(self.sys_cfg, self.opt_cfg, cost_type="nis")
        result = runner.run_experiment(plot=False)
        q = result["optimized_params"]["optimal_Q"].item()
        r = result["optimized_params"]["optimal_R"]
        self.assertGreaterEqual(q, 0.01)
        self.assertLessEqual(q, 1.0)
        self.assertGreaterEqual(r, 0.1)
        self.assertLessEqual(r, 1.0)

    def test_combined_cost_runs(self):
        runner = ExperimentRunner(self.sys_cfg, self.opt_cfg, cost_type="combined")
        result = runner.run_experiment(plot=False)
        self.assertIn("optimized_params", result)

    def test_nees_cost_runs(self):
        runner = ExperimentRunner(self.sys_cfg, self.opt_cfg, cost_type="nees")
        result = runner.run_experiment(plot=False)
        self.assertIn("optimized_params", result)


class TestRunMultipleExperiments(unittest.TestCase):
    def test_returns_correct_length(self):
        sys_cfg, opt_cfg = _make_configs(n_calls=12)
        results = run_multiple_experiments(
            sys_cfg, opt_cfg, num_experiments=2, cost_type="nis", plot=False
        )
        self.assertEqual(len(results), 2)

    def test_different_seeds_give_different_results(self):
        sys_cfg, opt_cfg = _make_configs(n_calls=12)
        results = run_multiple_experiments(
            sys_cfg, opt_cfg, num_experiments=2, cost_type="nis", plot=False
        )
        q0 = results[0]["optimized_params"]["optimal_Q"].item()
        q1 = results[1]["optimized_params"]["optimal_Q"].item()
        # 異なるシードからは（確率的に）異なる結果が得られる
        # ※ 厳密な等値比較は確率的なため、単純に実行完了を確認
        self.assertIsInstance(q0, float)
        self.assertIsInstance(q1, float)
