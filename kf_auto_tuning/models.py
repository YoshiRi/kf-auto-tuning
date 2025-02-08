import yaml
import numpy as np
from kf_auto_tuning.config import SystemConfig, OptimizationConfig

class DefaultModel:
    def __init__(self, config_path: str = None):
        """
        デフォルトモデルの初期化。設定ファイル (YAML) が指定された場合、その内容でシステム設定と最適化設定を上書きします。
        設定ファイルは以下のような形式を想定しています:
        """
        if not config_path:
            from pathlib import Path
            default_file_path = (Path(__file__).resolve().parent.parent / 'config' / 'sample_config.yaml')
            config_path = str(default_file_path)
            print(f"Using default config file: {default_file_path}")
            
        try:        
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
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            
        # デフォルト設定
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
