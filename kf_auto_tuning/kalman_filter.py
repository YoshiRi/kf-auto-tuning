import numpy as np
from typing import Tuple
from .config import SystemConfig

class KalmanFilter:
    """拡張可能なカルマンフィルタ実装"""
    
    def __init__(self, config: SystemConfig):
        self.config = config

    def run(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        observations: np.ndarray,
        true_states: np.ndarray = None
    ) -> Tuple[np.ndarray, dict]:
        """カルマンフィルタを実行"""
        x = self.config.x0.copy()
        P = self.config.P0.copy()
        estimates = []
        metrics = {'nees': [], 'nis': []}

        for i, z in enumerate(observations):
            # 予測ステップ
            x_pred = self.config.F @ x + self.config.G.flatten() * self.config.u
            P_pred = self.config.F @ P @ self.config.F.T + Q

            # 更新ステップ
            S = self.config.H @ P_pred @ self.config.H.T + R
            K = P_pred @ self.config.H.T @ np.linalg.inv(S)
            x = x_pred + K @ (z - self.config.H @ x_pred)
            P = P_pred - K @ self.config.H @ P_pred

            estimates.append(x)

            # メトリクス計算
            self._calculate_metrics(metrics, x_pred, P_pred, S, z, x, true_states[i] if true_states is not None else None)

        return np.array(estimates), self._compute_final_metrics(metrics)

    def _calculate_metrics(self, metrics, x_pred, P_pred, S, z, x_est, true_state=None):
        """各種メトリクスを計算"""
        # NIS計算
        innovation = z - self.config.H @ x_pred
        metrics['nis'].append(innovation.T @ np.linalg.inv(S) @ innovation)

        # NEES計算（真値がある場合）
        if true_state is not None:
            estimation_error = true_state - x_est
            metrics['nees'].append(estimation_error.T @ np.linalg.inv(P_pred) @ estimation_error)

    def _compute_final_metrics(self, metrics: dict) -> dict:
        """最終メトリクスを計算"""
        return {
            'mean_nees': np.mean(metrics['nees']) if metrics['nees'] else 0,
            'var_nees': np.var(metrics['nees']) if metrics['nees'] else 0,
            'mean_nis': np.mean(metrics['nis']),
            'var_nis': np.var(metrics['nis'])
        }