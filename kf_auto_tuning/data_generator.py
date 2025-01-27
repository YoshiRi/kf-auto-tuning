import numpy as np
from typing import Tuple
from .config import SystemConfig

class DataGenerator:
    """状態空間モデルに基づくシミュレーションデータ生成クラス"""
    
    def __init__(self, config: SystemConfig):
        self.config = config

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """シミュレーションデータを生成"""
        if self.config.data_seed is not None:
            np.random.seed(self.config.data_seed)
        
        x = self.config.x0.copy()
        true_states = []
        observations = []

        for _ in range(self.config.steps):
            w = np.random.multivariate_normal(
                np.zeros(self.config.Q_true.shape[0]), 
                self.config.Q_true
            )
            v = np.random.normal(0, np.sqrt(self.config.R_true))

            x = self.config.F @ x + self.config.G.flatten() * self.config.u + w
            z = self.config.H @ x + v

            true_states.append(x)
            observations.append(z)

        return np.array(true_states), np.array(observations)