"""kf_auto_tuning/experiment.py

モデルを選択して実験を実行する CLI エントリーポイント。

使い方:
    python -m kf_auto_tuning.experiment
    python -m kf_auto_tuning.experiment --model advanced
    python -m kf_auto_tuning.experiment --model default --plot
    python -m kf_auto_tuning.experiment --model advanced --config my_config.yaml
"""

import argparse

from kf_auto_tuning.models import DefaultModel, AdvancedModel


def create_model(model_name: str, config_path: str = None):
    """モデル名に対応するモデルオブジェクトを生成して返す。"""
    name = model_name.lower()
    if name == "default":
        return DefaultModel(config_path)
    elif name == "advanced":
        return AdvancedModel(config_path)
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. Choose 'default' or 'advanced'."
        )


def run_experiment(model, plot: bool = False):
    """モデルの run() を呼び出して実験を実行する。"""
    result = model.run(plot=plot)
    print("\n--- Experiment Summary ---")
    true_q = result["true_params"]["Q"].diagonal()
    true_r = result["true_params"]["R"]
    opt_q = result["optimized_params"]["optimal_Q"].diagonal()
    opt_r = result["optimized_params"]["optimal_R"]
    print(f"  True      Q={true_q}, R={true_r}")
    print(f"  Optimized Q={opt_q}, R={opt_r:.4f}")
    m_opt = result["metrics"]["optimal"]
    m_true = result["metrics"]["true_params"]
    print(
        f"  NEES (True / Optimized): {m_true['mean_nees']:.2f} / {m_opt['mean_nees']:.2f}"
    )
    print(
        f"  NIS  (True / Optimized): {m_true['mean_nis']:.2f} / {m_opt['mean_nis']:.2f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run a KF auto-tuning experiment with the selected model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        choices=["default", "advanced"],
        help="Model to use: 'default' (NIS cost) or 'advanced' (combined NEES+NIS cost)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/sample_config.yaml)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show result plots",
    )
    args = parser.parse_args()

    model = create_model(args.model, args.config)
    run_experiment(model, plot=args.plot)


if __name__ == "__main__":
    main()
