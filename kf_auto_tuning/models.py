import yaml
import numpy as np
from kf_auto_tuning.config import SystemConfig, OptimizationConfig

class DefaultModel:
    def __init__(self, config_path: str = None):
        """
        デフォルトモデルの初期化。設定ファイル (YAML) が指定された場合、その内容でシステム設定と最適化設定を上書きします。
        設定ファイルは以下のような形式を想定しています:

        system_config:
          F: [[1, 0.1], [0, 1]]
          G: [[0], [1]]
          H: [[1, 0]]
          Q_true: [0.0]            # 対角成分として扱われます
          R_true: 0.5
          x0: [0, 1]
          P0: [[1, 0], [0, 1]]
          u: 0
          steps: 100
          nx: 2
          nz: 1
          data_seed: 42

        optimization_config:
          param_bounds:
            Q: [0.01, 1.0]
            R: [0.1, 1.0]
          n_calls: 30
          opt_seed: 42
        """
        if config_path:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            system_cfg = cfg.get("system_config", {})
            opt_cfg = cfg.get("optimization_config", {})
            # Convert list representations to numpy arrays if necessary
            if "F" in system_cfg:
                system_cfg["F"] = np.array(system_cfg["F"])
            if "G" in system_cfg:
                system_cfg["G"] = np.array(system_cfg["G"])
            if "H" in system_cfg:
                system_cfg["H"] = np.array(system_cfg["H"])
            if "Q_true" in system_cfg:
                # Assume Q_true is given as a list for diagonal elements
                system_cfg["Q_true"] = np.diag(system_cfg["Q_true"])
            if "x0" in system_cfg:
                system_cfg["x0"] = np.array(system_cfg["x0"])
            if "P0" in system_cfg:
                system_cfg["P0"] = np.array(system_cfg["P0"])
        else:
            # デフォルト設定
            system_cfg = {
                "F": np.array([[1, 0.1], [0, 1]]),
                "G": np.array([[0], [1]]),
                "H": np.array([[1, 0]]),
                "Q_true": np.diag([0.0]),
                "R_true": 0.5,
                "x0": np.array([0, 1]),
                "P0": np.eye(2),
                "u": 0,
                "steps": 100,
                "nx": 2,
                "nz": 1,
                "data_seed": 42,
            }
            opt_cfg = {
                "param_bounds": {'Q': (0.01, 1.0), 'R': (0.1, 1.0)},
                "n_calls": 30,
                "opt_seed": 42,
            }
        self.system_config = SystemConfig(**system_cfg)
        self.optimization_config = OptimizationConfig(**opt_cfg)

    def run(self):
        # モデルの実行ロジックを実装
        pass

class AdvancedModel:
    def __init__(self):
        # AdvancedModel 専用の初期化処理
        pass

    def run(self):
        # 高度なモデル実行ロジックを実装
        pass
