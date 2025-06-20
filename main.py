import numpy as np
from typing import Any, Dict, List
from kf_auto_tuning.config import SystemConfig, OptimizationConfig
from kf_auto_tuning.data_generator import DataGenerator
from kf_auto_tuning.kalman_filter import KalmanFilter
from kf_auto_tuning.optimizer import ParameterOptimizer
from kf_auto_tuning.visualization import ResultVisualizer
from kf_auto_tuning.models import DefaultModel


class ExperimentRunner:
    """
    単一実験を実行するためのクラス

    Attributes:
        system_config (SystemConfig): システムの設定情報
        opt_config (OptimizationConfig): 最適化の設定情報
    """
    def __init__(self, system_config: SystemConfig, opt_config: OptimizationConfig) -> None:
        self.system_config = system_config
        self.opt_config = opt_config

    def run_experiment(self, plot: bool = False) -> Dict[str, Any]:
        """
        1回の実験を実行し、最適化結果と推定結果を返す

        Args:
            plot (bool): Trueの場合、最適化の進捗と推定結果を可視化する

        Returns:
            dict: 実験結果をまとめた辞書
        """
        # --- データ生成 ---
        data_gen = DataGenerator(self.system_config)
        true_states, observations = data_gen.generate()

        # 最適化中の各反復の結果を記録するリスト
        iteration_results = []

        # --- コスト関数の定義 ---
        def cost_function(params):
            kf = KalmanFilter(self.system_config)
            _, metrics = kf.run(
                Q=np.diag(np.full(self.system_config.nx, params[0])),
                R=params[1],
                observations=observations,
                true_states=true_states
            )

            # NEES, NISに基づいたコスト計算
            c_nees = abs(np.log(metrics['mean_nees'] / self.system_config.nx)) + \
                     abs(np.log(metrics['var_nees'] / (2.0 * self.system_config.nx)))
            c_nis = abs(np.log(metrics['mean_nis'] / self.system_config.nz)) + \
                    abs(np.log(metrics['var_nis'] / (2.0 * self.system_config.nz)))
            # 今回は c_nis をコストとして採用
            cost = c_nis

            iteration_results.append((params[0], params[1], cost,
                                      metrics['mean_nees'], metrics['mean_nis']))
            return cost

        # --- パラメータ最適化 ---
        optimizer = ParameterOptimizer(cost_function, self.opt_config, self.system_config.nx)
        result = optimizer.optimize()
        print(f"Optimal Parameters: Q={result['optimal_Q']}, R={result['optimal_R']}")

        # --- 最適パラメータでの推定 ---
        kf = KalmanFilter(self.system_config)
        estimates_opt, metrics_opt = kf.run(result['optimal_Q'], result['optimal_R'], observations)

        # --- 真のパラメータでの推定 ---
        estimates_true, metrics_true = kf.run(
            self.system_config.Q_true, self.system_config.R_true, observations
        )

        # --- 結果の可視化 ---
        if plot:
            ResultVisualizer.plot_optimization_progress(iteration_results)
            ResultVisualizer.plot_tracking_results(
                true_states,
                observations,
                {
                    'Optimal Params': estimates_opt,
                    'True Params': estimates_true
                }
            )

        # 結果の返却
        return {
            'true_params': {'Q': self.system_config.Q_true, 'R': self.system_config.R_true},
            'optimized_params': result,
            'metrics': {
                'optimal': metrics_opt,
                'true_params': metrics_true
            },
            'estimation_results': {
                'optimal': estimates_opt,
                'true_params': estimates_true
            },
            'experiments': {
                'true_states': true_states,
                'observations': observations
            }
        }


def analyze_results(experiments: List[Dict[str, Any]], plot: bool = False) -> None:
    """
    複数実験の結果をコンソールに出力し、必要に応じてプロットする

    Args:
        experiments (List[Dict[str, Any]]): 実験結果のリスト
        plot (bool): Trueの場合、各実験の推定結果を可視化する
    """
    for i, exp in enumerate(experiments):
        print(f"\nExperiment {i+1}")
        print(f"True Q: {exp['true_params']['Q'].diagonal()}, R: {exp['true_params']['R']}")
        print(f"Optimized Q: {exp['optimized_params']['optimal_Q'].diagonal()}, R: {exp['optimized_params']['optimal_R']}")
        print(f"NEES (True vs Optimized): {exp['metrics']['true_params']['mean_nees']:.2f} vs {exp['metrics']['optimal']['mean_nees']:.2f}")

        if plot:
            ResultVisualizer.plot_tracking_results(
                exp['experiments']['true_states'],
                exp['experiments']['observations'],
                {
                    'Optimal Params': exp['estimation_results']['optimal'],
                    'True Params': exp['estimation_results']['true_params']
                }
            )


def run_multiple_experiments(
    base_system_config: SystemConfig,
    base_opt_config: OptimizationConfig,
    num_experiments: int = 3,
    plot: bool = False
) -> List[Dict[str, Any]]:
    """
    異なる乱数シードなどを用いて複数回実験を実行する

    Args:
        base_system_config (SystemConfig): 基本となるシステム設定
        base_opt_config (OptimizationConfig): 基本となる最適化設定
        num_experiments (int): 実験回数
        plot (bool): Trueの場合、各実験の推定結果を可視化する

    Returns:
        List[Dict[str, Any]]: 各実験の結果をまとめたリスト
    """
    experiments = []
    for seed in range(num_experiments):
        # 各実験ごとに乱数シードを変更
        system_config = SystemConfig(
            **{**base_system_config.__dict__, 'data_seed': seed}
        )
        opt_config = OptimizationConfig(
            **{**base_opt_config.__dict__, 'opt_seed': seed}
        )

        runner = ExperimentRunner(system_config, opt_config)
        experiment_result = runner.run_experiment(plot=plot)
        experiments.append(experiment_result)

    analyze_results(experiments, plot=plot)
    return experiments


def main():
    """
    エントリーポイント。DefaultModel の設定を用いて単一実験を実行する例。
    複数実験を実行したい場合は run_multiple_experiments() のコメントアウト部分を有効にしてください。
    """
    # モデル（設定）を取得
    model = DefaultModel()

    # 単一実験の実行
    runner = ExperimentRunner(model.system_config, model.optimization_config)
    runner.run_experiment(plot=True)

    # --- 複数実験を実行する場合 ---
    # experiments = run_multiple_experiments(model.system_config, model.base_opt_config, num_experiments=3, plot=True)
    # 取得した experiments をさらに処理するなど、再利用可能です。


if __name__ == "__main__":
    main()
