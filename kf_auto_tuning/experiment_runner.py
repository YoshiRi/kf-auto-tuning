"""kf_auto_tuning/experiment_runner.py

ExperimentRunner および複数実験ユーティリティ。
main.py と models.py の両方から利用されます。
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Literal

from .config import SystemConfig, OptimizationConfig
from .data_generator import DataGenerator
from .kalman_filter import KalmanFilter
from .optimizer import ParameterOptimizer
from .visualization import ResultVisualizer

CostType = Literal["nis", "nees", "combined"]


class ExperimentRunner:
    """単一実験（データ生成 → 最適化 → 推定 → 可視化）を実行するクラス。

    Parameters
    ----------
    system_config:
        システム行列・真のノイズ共分散などを保持する設定オブジェクト。
    opt_config:
        最適化の探索範囲・反復回数・乱数シードなどを保持する設定オブジェクト。
    cost_type:
        最適化に使うコスト関数の種類。

        - ``"nis"``      : NIS（観測一貫性）のみ。デフォルト。
        - ``"nees"``     : NEES（推定誤差一貫性）のみ。真値が必要。
        - ``"combined"`` : NEES と NIS を加算した複合コスト。
    """

    def __init__(
        self,
        system_config: SystemConfig,
        opt_config: OptimizationConfig,
        cost_type: CostType = "nis",
    ) -> None:
        self.system_config = system_config
        self.opt_config = opt_config
        self.cost_type = cost_type

    def _build_cost_function(self, observations, true_states, iteration_results):
        """コスト関数を生成して返す。"""
        nx = self.system_config.nx
        nz = self.system_config.nz

        def cost_function(params):
            kf = KalmanFilter(self.system_config)
            _, metrics = kf.run(
                Q=np.diag([params[0]]),
                R=params[1],
                observations=observations,
                true_states=true_states,
            )

            c_nis = abs(np.log(metrics["mean_nis"] / nz)) + abs(
                np.log(metrics["var_nis"] / (2.0 * nz))
            )
            c_nees = (
                abs(np.log(metrics["mean_nees"] / nx))
                + abs(np.log(metrics["var_nees"] / (2.0 * nx)))
                if metrics["mean_nees"] > 0
                else 0.0
            )

            if self.cost_type == "nis":
                cost = c_nis
            elif self.cost_type == "nees":
                cost = c_nees
            else:  # "combined"
                cost = c_nis + c_nees

            iteration_results.append(
                (params[0], params[1], cost, metrics["mean_nees"], metrics["mean_nis"])
            )
            return cost

        return cost_function

    def run_experiment(self, plot: bool = False) -> Dict[str, Any]:
        """1 回の実験を実行して結果辞書を返す。

        Parameters
        ----------
        plot:
            True のとき最適化進捗と追跡結果をプロットする。

        Returns
        -------
        dict
            ``true_params``, ``optimized_params``, ``metrics``,
            ``estimation_results``, ``experiments`` キーを持つ辞書。
        """
        # --- データ生成 ---
        data_gen = DataGenerator(self.system_config)
        true_states, observations = data_gen.generate()

        iteration_results: list = []

        # --- コスト関数の構築 ---
        cost_fn = self._build_cost_function(observations, true_states, iteration_results)

        # --- パラメータ最適化 ---
        optimizer = ParameterOptimizer(cost_fn, self.opt_config)
        result = optimizer.optimize()
        print(
            f"[{self.cost_type.upper()}] Optimal Parameters: "
            f"Q={result['optimal_Q'].diagonal()}, R={result['optimal_R']:.4f}"
        )

        # --- 最適パラメータでの推定 ---
        kf = KalmanFilter(self.system_config)
        estimates_opt, metrics_opt = kf.run(
            result["optimal_Q"], result["optimal_R"], observations
        )

        # --- 真のパラメータでの推定（比較用） ---
        estimates_true, metrics_true = kf.run(
            self.system_config.Q_true, self.system_config.R_true, observations
        )

        # --- 可視化 ---
        if plot:
            ResultVisualizer.plot_optimization_progress(iteration_results)
            ResultVisualizer.plot_tracking_results(
                true_states,
                observations,
                {
                    "Optimal Params": estimates_opt,
                    "True Params": estimates_true,
                },
            )

        return {
            "true_params": {
                "Q": self.system_config.Q_true,
                "R": self.system_config.R_true,
            },
            "optimized_params": result,
            "metrics": {
                "optimal": metrics_opt,
                "true_params": metrics_true,
            },
            "estimation_results": {
                "optimal": estimates_opt,
                "true_params": estimates_true,
            },
            "experiments": {
                "true_states": true_states,
                "observations": observations,
            },
        }


def analyze_results(experiments: List[Dict[str, Any]], plot: bool = False) -> None:
    """複数実験の結果をコンソールに出力し、必要に応じてプロットする。

    Parameters
    ----------
    experiments:
        :py:meth:`ExperimentRunner.run_experiment` の結果リスト。
    plot:
        True のとき各実験の追跡結果をプロットする。
    """
    for i, exp in enumerate(experiments):
        print(f"\nExperiment {i + 1}")
        print(
            f"  True Q: {exp['true_params']['Q'].diagonal()}, "
            f"R: {exp['true_params']['R']}"
        )
        print(
            f"  Optimized Q: {exp['optimized_params']['optimal_Q'].diagonal()}, "
            f"R: {exp['optimized_params']['optimal_R']:.4f}"
        )
        print(
            f"  NEES (True / Optimized): "
            f"{exp['metrics']['true_params']['mean_nees']:.2f} / "
            f"{exp['metrics']['optimal']['mean_nees']:.2f}"
        )
        print(
            f"  NIS  (True / Optimized): "
            f"{exp['metrics']['true_params']['mean_nis']:.2f} / "
            f"{exp['metrics']['optimal']['mean_nis']:.2f}"
        )

        if plot:
            ResultVisualizer.plot_tracking_results(
                exp["experiments"]["true_states"],
                exp["experiments"]["observations"],
                {
                    "Optimal Params": exp["estimation_results"]["optimal"],
                    "True Params": exp["estimation_results"]["true_params"],
                },
            )


def run_multiple_experiments(
    base_system_config: SystemConfig,
    base_opt_config: OptimizationConfig,
    num_experiments: int = 3,
    cost_type: CostType = "nis",
    plot: bool = False,
) -> List[Dict[str, Any]]:
    """異なる乱数シードで複数回実験を実行する。

    Parameters
    ----------
    base_system_config:
        ベースとなるシステム設定。各実験で ``data_seed`` が上書きされる。
    base_opt_config:
        ベースとなる最適化設定。各実験で ``opt_seed`` が上書きされる。
    num_experiments:
        実験回数。
    cost_type:
        コスト関数の種類（``"nis"``, ``"nees"``, ``"combined"``）。
    plot:
        True のとき各実験の追跡結果をプロットする。

    Returns
    -------
    list
        各実験の結果辞書のリスト。
    """
    experiments = []
    for seed in range(num_experiments):
        system_config = SystemConfig(
            **{**base_system_config.__dict__, "data_seed": seed}
        )
        opt_config = OptimizationConfig(
            **{**base_opt_config.__dict__, "opt_seed": seed}
        )
        runner = ExperimentRunner(system_config, opt_config, cost_type=cost_type)
        experiments.append(runner.run_experiment(plot=plot))

    analyze_results(experiments, plot=plot)
    return experiments
