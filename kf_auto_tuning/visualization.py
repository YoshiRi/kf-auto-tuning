import matplotlib.pyplot as plt
from typing import List, Tuple

class ResultVisualizer:
    """結果可視化クラス"""
    
    @staticmethod
    def plot_optimization_progress(iteration_results: List[Tuple]):
        """最適化過程を可視化"""
        q_vals, r_vals, costs, nees_vals, nis_vals = zip(*iteration_results)

        # コスト推移
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(costs, marker='o')
        plt.title('Cost Function Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')

        # パラメータ空間
        plt.subplot(1, 2, 2)
        sc = plt.scatter(q_vals, r_vals, c=costs, cmap='viridis', s=50, edgecolor='k')
        plt.colorbar(sc, label='Cost')
        plt.title('Parameter Space Exploration')
        plt.xlabel('Q Value')
        plt.ylabel('R Value')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_tracking_results(true_states, observations, estimates_dict):
        """追跡結果を比較"""
        plt.figure(figsize=(10, 6))
        plt.plot(true_states[:, 0], label='True Position', color='black', linestyle='--')
        plt.plot(observations.squeeze(), 'r.', label='Observations', alpha=0.6)
        
        for label, data in estimates_dict.items():
            plt.plot(data[:, 0], label=label)
            
        plt.title('State Estimation Comparison')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True)
        plt.show()