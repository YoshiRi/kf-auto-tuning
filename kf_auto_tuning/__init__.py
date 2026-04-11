"""kf_auto_tuning: Bayesian auto-tuning of Kalman filter noise parameters.

主要クラスとユーティリティを公開します。

Examples
--------
>>> from kf_auto_tuning import DefaultModel
>>> model = DefaultModel()
>>> result = model.run(plot=False)

>>> from kf_auto_tuning import ExperimentRunner
>>> from kf_auto_tuning.models import DefaultModel
>>> m = DefaultModel()
>>> runner = ExperimentRunner(m.system_config, m.optimization_config, cost_type="combined")
>>> result = runner.run_experiment()
"""

from kf_auto_tuning.config import SystemConfig, OptimizationConfig
from kf_auto_tuning.data_generator import DataGenerator
from kf_auto_tuning.kalman_filter import KalmanFilter
from kf_auto_tuning.optimizer import ParameterOptimizer
from kf_auto_tuning.visualization import ResultVisualizer
from kf_auto_tuning.models import DefaultModel, AdvancedModel
from kf_auto_tuning.experiment_runner import (
    ExperimentRunner,
    analyze_results,
    run_multiple_experiments,
)

__all__ = [
    "SystemConfig",
    "OptimizationConfig",
    "DataGenerator",
    "KalmanFilter",
    "ParameterOptimizer",
    "ResultVisualizer",
    "DefaultModel",
    "AdvancedModel",
    "ExperimentRunner",
    "analyze_results",
    "run_multiple_experiments",
]
