"""tests/test_models.py

DefaultModel と AdvancedModel のユニットテスト。
"""

import unittest
from kf_auto_tuning.models import DefaultModel, AdvancedModel


class TestDefaultModel(unittest.TestCase):
    def setUp(self):
        self.model = DefaultModel()

    def test_configs_loaded(self):
        self.assertIsNotNone(self.model.system_config)
        self.assertIsNotNone(self.model.optimization_config)

    def test_run_returns_dict(self):
        # n_calls を最小限に抑えた設定で実行
        self.model.optimization_config.n_calls = 12
        self.model.system_config.steps = 30
        result = self.model.run(plot=False)
        self.assertIsInstance(result, dict)
        self.assertIn("optimized_params", result)

    def test_invalid_config_path_raises(self):
        with self.assertRaises(RuntimeError):
            DefaultModel(config_path="/nonexistent/path.yaml")


class TestAdvancedModel(unittest.TestCase):
    def setUp(self):
        self.model = AdvancedModel()

    def test_configs_loaded(self):
        self.assertIsNotNone(self.model.system_config)
        self.assertIsNotNone(self.model.optimization_config)

    def test_run_returns_dict(self):
        self.model.optimization_config.n_calls = 12
        self.model.system_config.steps = 30
        result = self.model.run(plot=False)
        self.assertIsInstance(result, dict)
        self.assertIn("optimized_params", result)

    def test_invalid_config_path_raises(self):
        with self.assertRaises(RuntimeError):
            AdvancedModel(config_path="/nonexistent/path.yaml")
