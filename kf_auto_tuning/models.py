"""kf_auto_tuning/models.py

設定を保持するモデルクラス群。
各クラスは YAML から設定を読み込み、run() で実験を実行します。

クラス
------
DefaultModel
    NIS（観測一貫性）のみをコスト関数に使う標準モデル。

AdvancedModel
    NEES（推定誤差一貫性）と NIS を組み合わせた複合コストを使う高度なモデル。
    真値が存在する場合により厳密なチューニングが可能。
"""

from __future__ import annotations

import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

from kf_auto_tuning.config import SystemConfig, OptimizationConfig


def _load_config(config_path: Optional[str]) -> tuple[SystemConfig, OptimizationConfig]:
    """YAML ファイルを読み込んで SystemConfig と OptimizationConfig を返す共通ローダー。"""
    if not config_path:
        default_file_path = (
            Path(__file__).resolve().parent.parent / "config" / "sample_config.yaml"
        )
        config_path = str(default_file_path)
        print(f"Using default config file: {default_file_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    system_cfg = cfg.get("system_config", {})
    opt_cfg = cfg.get("optimization_config", {})

    # リスト形式を numpy 配列に変換
    for key in ("F", "G", "H", "x0", "P0"):
        if key in system_cfg:
            system_cfg[key] = np.array(system_cfg[key])
    if "Q_true" in system_cfg:
        system_cfg["Q_true"] = np.diag(system_cfg["Q_true"])

    return SystemConfig(**system_cfg), OptimizationConfig(**opt_cfg)


class DefaultModel:
    """NIS コスト関数を用いる標準的なカルマンフィルタ自動チューニングモデル。

    Parameters
    ----------
    config_path:
        YAML 設定ファイルのパス。省略時は ``config/sample_config.yaml`` を使用。
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        try:
            self.system_config, self.optimization_config = _load_config(config_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}") from e

    def run(self, plot: bool = False) -> Dict[str, Any]:
        """NIS コスト関数を使って最適化実験を実行する。

        Parameters
        ----------
        plot:
            True のとき最適化進捗と追跡結果をプロットする。

        Returns
        -------
        dict
            実験結果辞書（``true_params``, ``optimized_params``, ``metrics``,
            ``estimation_results``, ``experiments``）。
        """
        from kf_auto_tuning.experiment_runner import ExperimentRunner

        runner = ExperimentRunner(
            self.system_config,
            self.optimization_config,
            cost_type="nis",
        )
        return runner.run_experiment(plot=plot)


class AdvancedModel:
    """NEES + NIS 複合コスト関数を用いる高度なカルマンフィルタ自動チューニングモデル。

    真値（状態推定誤差）と観測残差の両方を使って一貫性を評価するため、
    DefaultModel より厳密なチューニングが可能です。

    Parameters
    ----------
    config_path:
        YAML 設定ファイルのパス。省略時は ``config/sample_config.yaml`` を使用。
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        try:
            self.system_config, self.optimization_config = _load_config(config_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}") from e

    def run(self, plot: bool = False) -> Dict[str, Any]:
        """NEES + NIS 複合コスト関数を使って最適化実験を実行する。

        Parameters
        ----------
        plot:
            True のとき最適化進捗と追跡結果をプロットする。

        Returns
        -------
        dict
            実験結果辞書（``true_params``, ``optimized_params``, ``metrics``,
            ``estimation_results``, ``experiments``）。
        """
        from kf_auto_tuning.experiment_runner import ExperimentRunner

        runner = ExperimentRunner(
            self.system_config,
            self.optimization_config,
            cost_type="combined",
        )
        return runner.run_experiment(plot=plot)
