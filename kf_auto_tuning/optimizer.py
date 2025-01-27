from skopt import gp_minimize
from skopt.space import Real
from typing import Callable, Dict
import numpy as np
from .config import OptimizationConfig

class ParameterOptimizer:
    """ベイズ最適化によるパラメータチューニングクラス"""
    
    def __init__(self, cost_function: Callable, config: OptimizationConfig):
        self.cost_function = cost_function
        self.config = config

    def optimize(self) -> Dict:
        """最適化を実行"""
        space = [
            Real(*self.config.param_bounds['Q'], name="Q"),
            Real(*self.config.param_bounds['R'], name="R")
        ]

        result = gp_minimize(
            self.cost_function,
            space,
            n_calls=self.config.n_calls,
            random_state=self.config.opt_seed
        )

        return {
            'optimal_Q': np.diag([result.x[0]]),
            'optimal_R': result.x[1],
            'optimization_result': result
        }