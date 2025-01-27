import numpy as np
from kf_auto_tuning.config import SystemConfig, OptimizationConfig
from kf_auto_tuning.data_generator import DataGenerator
from kf_auto_tuning.kalman_filter import KalmanFilter
from kf_auto_tuning.optimizer import ParameterOptimizer
from kf_auto_tuning.visualization import ResultVisualizer

    
def run_experiment(system_config: SystemConfig, opt_config: OptimizationConfig, plot=False):
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
    estimates_opt, metrics_opt = kf.run(result['optimal_Q'], result['optimal_R'], observations)
    
    # 真のパラメータでの推定
    estimates_true, metrics_true = kf.run(system_config.Q_true, system_config.R_true, observations)

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

    return {
        'true_params': {'Q': system_config.Q_true, 'R': system_config.R_true},
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

def analyze_results(experiments, plot=False):
    """実験結果を分析"""
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
                })
            
def main():
    # システム設定
    base_system_config = SystemConfig(
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
        nz=1,
        data_seed=42  # デフォルトシード
    )
    
    # 最適化設定
    base_opt_config = OptimizationConfig(
        param_bounds={'Q': (0.01, 1.0), 'R': (0.1, 1.0)},
        n_calls=30,
        opt_seed=42  # デフォルトシード
    )

    run_experiment(base_system_config, base_opt_config, plot=True)
    # for multiple_experiments_sample
    #multiple_experiments_sample(base_system_config, base_opt_config)
    
def multiple_experiments_sample(base_system_config, base_opt_config):
    # 実験1: 異なる乱数シードで実行
    experiments = []
    for seed in range(3):  # 3つの異なるシード
        system_config = SystemConfig(
            **{**base_system_config.__dict__, 'data_seed': seed}
        )
        opt_config = OptimizationConfig(
            **{**base_opt_config.__dict__, 'opt_seed': seed}
        )
        
        experiments.append(run_experiment(system_config, opt_config))

    # 実験2: 異なる真のパラメータで実行
    # for Q_true_val in [0.1, 0.5, 1.0]:
    #     system_config = SystemConfig(
    #         **{**base_system_config.__dict__, 
    #          'Q_true': np.diag([Q_true_val]),
    #          'R_true': Q_true_val * 0.5}
    #     )
    #     experiments.append(run_experiment(system_config, base_opt_config))

    # 結果分析
    analyze_results(experiments, plot=True)

if __name__ == "__main__":
    main()