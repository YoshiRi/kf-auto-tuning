import numpy as np
from kf_auto_tuning.config import SystemConfig, OptimizationConfig
from kf_auto_tuning.data_generator import DataGenerator
from kf_auto_tuning.kalman_filter import KalmanFilter
from kf_auto_tuning.optimizer import ParameterOptimizer
from kf_auto_tuning.visualization import ResultVisualizer

def main():
    # システム設定
    system_config = SystemConfig(
        F=np.array([[1, 0.1], [0, 1]]),
        G=np.array([[0], [1]]),
        H=np.array([[1, 0]]),
        Q_true=np.diag([0.0]),
        R_true=0.5,
        x0=np.array([0, 1]),
        P0=np.eye(2),
        u=0,
        steps=100,
        nx=2,
        nz=1
    )

    # 最適化設定
    opt_config = OptimizationConfig(
        param_bounds={'Q': (0.01, 1.0), 'R': (0.1, 1.0)},
        n_calls=30,
        random_state=42
    )

    # データ生成
    data_gen = DataGenerator(system_config)
    true_states, observations = data_gen.generate()

    # 最適化用コールバックデータ
    iteration_results = []

    def cost_function(params):
        kf = KalmanFilter(system_config)
        _, metrics = kf.run(
            Q=np.diag([params[0]]),
            R=params[1],
            observations=observations,
            true_states=true_states
        )
        
        c_nees = abs(np.log(metrics['mean_nees'] / system_config.nx)) + \
               abs(np.log(metrics['var_nees'] / (2.0 * system_config.nx)))
        c_nis =  abs(np.log(metrics['mean_nis'] / system_config.nz)) + \
               abs(np.log(metrics['var_nis'] / (2.0 * system_config.nz)))
        # One of the easiest solution is using the c_nis
        cost = c_nis
        
        iteration_results.append((params[0], params[1], cost, 
                                metrics['mean_nees'], metrics['mean_nis']))
        return cost

    # 最適化実行
    optimizer = ParameterOptimizer(cost_function, opt_config)
    result = optimizer.optimize()

    # 結果表示
    print(f"Optimal Parameters: Q={result['optimal_Q']}, R={result['optimal_R']}")

    # 可視化
    kf = KalmanFilter(system_config)
    
    # 最適パラメータでの推定
    estimates_opt, _ = kf.run(result['optimal_Q'], result['optimal_R'], observations)
    
    # 真のパラメータでの推定
    estimates_true, _ = kf.run(system_config.Q_true, system_config.R_true, observations)

    ResultVisualizer.plot_optimization_progress(iteration_results)
    ResultVisualizer.plot_tracking_results(
        true_states,
        observations,
        {
            'Optimal Params': estimates_opt,
            'True Params': estimates_true
        }
    )

if __name__ == "__main__":
    main()