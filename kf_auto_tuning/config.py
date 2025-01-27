from dataclasses import dataclass
import numpy as np

@dataclass
class SystemConfig:
    F: np.ndarray
    G: np.ndarray
    H: np.ndarray
    Q_true: np.ndarray
    R_true: np.ndarray
    x0: np.ndarray
    P0: np.ndarray
    u: float
    steps: int
    nx: int
    nz: int

@dataclass
class OptimizationConfig:
    param_bounds: dict
    n_calls: int
    random_state: int