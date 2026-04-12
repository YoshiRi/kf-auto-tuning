"""main.py

コマンドラインから直接実行するためのエントリーポイント。

使い方:
    python main.py                    # 単一実験（デフォルト設定）
    python main.py --multi 5          # 異なるシードで 5 回実験
    python main.py --cost combined    # 複合コスト関数を使用
    python main.py --config path.yaml # カスタム設定ファイルを指定
    python main.py --no-plot          # プロット無効
"""

import argparse

from kf_auto_tuning.models import DefaultModel
from kf_auto_tuning.experiment_runner import ExperimentRunner, run_multiple_experiments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalman filter auto-tuning via Bayesian optimization."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/sample_config.yaml)",
    )
    parser.add_argument(
        "--cost",
        type=str,
        choices=["nis", "nees", "combined"],
        default="nis",
        help="Cost function type (default: nis)",
    )
    parser.add_argument(
        "--multi",
        type=int,
        default=None,
        metavar="N",
        help="Run N experiments with different random seeds",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable result plots",
    )
    args = parser.parse_args()

    model = DefaultModel(config_path=args.config)
    plot = not args.no_plot

    if args.multi is not None:
        run_multiple_experiments(
            model.system_config,
            model.optimization_config,
            num_experiments=args.multi,
            cost_type=args.cost,
            plot=plot,
        )
    else:
        runner = ExperimentRunner(
            model.system_config,
            model.optimization_config,
            cost_type=args.cost,
        )
        runner.run_experiment(plot=plot)


if __name__ == "__main__":
    main()
