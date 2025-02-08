"""kf_auto_tuning/experiment.py

実験処理のエントリーポイント。各モデルを用いてOptimizerの動作を確認します。
"""

import argparse
from kf_auto_tuning.optimizer import Optimizer
from kf_auto_tuning.models import DefaultModel, AdvancedModel

def create_model(model_name: str):
    if model_name.lower() == "default":
        return DefaultModel()
    elif model_name.lower() == "advanced":
        return AdvancedModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_experiment(model):
    optimizer = Optimizer(model)
    result = optimizer.optimize()  # Optimizerの処理を実行
    print("Experiment Result:", result)
    return result

def main():
    parser = argparse.ArgumentParser(description="Run experiment with specified model.")
    parser.add_argument("--model", type=str, default="default", help="Choose model: default or advanced")
    args = parser.parse_args()
    model = create_model(args.model)
    run_experiment(model)

if __name__ == "__main__":
    main()
