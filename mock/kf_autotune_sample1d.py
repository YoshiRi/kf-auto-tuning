import numpy as np
from numpy.linalg import inv
from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import VerboseCallback
import matplotlib.pyplot as plt

def generate_data(F, G, H, Q_true, R_true, x0, u, steps):
    """
    状態空間モデル (F, G, H, Q, R) に基づいてシミュレーションデータを生成する。
    - F, G, H: システム行列
    - Q_true, R_true: 真のプロセス/観測ノイズ共分散
    - x0: 初期状態
    - u: 入力(ここではスカラー)
    - steps: シミュレーションステップ数

    Returns:
    --------
    true_states : shape=(steps, nx)
    observations: shape=(steps, nz)
    """
    x = x0.copy()
    true_states = []
    observations = []

    for _ in range(steps):
        # プロセスノイズ (多次元の場合は多変量正規乱数を使用)
        w = np.random.multivariate_normal(np.zeros(Q_true.shape[0]), Q_true)
        # 観測ノイズ (今回の観測は1次元のため単変量正規乱数)
        v = np.random.normal(0, np.sqrt(R_true))

        # 状態遷移
        x = F @ x + G.flatten() * u + w
        # 観測
        z = H @ x + v

        true_states.append(x)
        observations.append(z)

    return np.array(true_states), np.array(observations)


def kalman_filter(F, G, H, Q, R, x0, P0, observations, u):
    """
    与えられたパラメータ (F, G, H, Q, R) を使ってカルマンフィルタを実行する。
    - x0: 初期推定状態
    - P0: 初期推定誤差共分散
    - observations: 観測値列 (shape=(steps, nz))
    - u: 入力(スカラー)

    Returns:
    --------
    estimates : shape=(steps, nx)    # 推定状態
    mean_nees : float               # 平均NEES
    var_nees  : float               # NEESの分散
    mean_nis  : float               # 平均NIS
    var_nis   : float               # NISの分散
    """
    x = x0.copy()
    P = P0.copy()
    estimates = []
    nees = []
    nis = []

    for z in observations:
        # 予測ステップ
        x_pred = F @ x + G.flatten() * u
        P_pred = F @ P @ F.T + Q

        # 観測更新ステップ
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ inv(S)
        # z が1次元の場合は K*(z - H@x_pred) でも同じだが、
        # 多次元拡張を見据えて行列積 @ を推奨
        x = x_pred + K @ (z - H @ x_pred)
        P = P_pred - K @ H @ P_pred

        # 推定値の保存
        estimates.append(x)

        # NEES (Normalized Estimation Error Squared)
        e_x = x - x_pred  # 予測との差分(innovationではないが, ここでは簡易的に使用)
        nees.append(e_x.T @ inv(P_pred) @ e_x)

        # NIS (Normalized Innovation Squared)
        e_z = z - H @ x_pred  # 観測残差
        nis.append(e_z.T @ inv(S) @ e_z)

    estimates = np.array(estimates)
    mean_nees = np.mean(nees)
    var_nees = np.var(nees)
    mean_nis = np.mean(nis)
    var_nis = np.var(nis)

    return estimates, mean_nees, var_nees, mean_nis, var_nis


def calc_cost(mean_nees, mean_nis, var_nees, var_nis, nx, nz):
    """
    NEES/NIS の平均・分散をもとにコストを計算する。
    値が理論値 nx, nz から大きく乖離している場合はコストを大きくする。

    mean_nees, mean_nis : NEES/NIS の平均
    var_nees, var_nis   : NEES/NIS の分散
    nx, nz              : 状態ベクトル次元、観測ベクトル次元

    Returns:
    --------
    cost : float
        一貫性(Consistency)が理想的な値(nx, nz)からどれだけ外れているかを示すスカラー。
    """
    # 例として、理想値(nx, nz)からの対数的乖離をコスト化し、NEES/NIS両方を足し合わせたものを返す
    # 分散に関しては理想的には 2*nx, 2*nz 近辺となる(自由度の期待値付近)
    cnees = abs(np.log(mean_nees / nx)) + abs(np.log(var_nees / (2.0 * nx)))
    cnis = abs(np.log(mean_nis / nz)) + abs(np.log(var_nis / (2.0 * nz)))
    return cnis


def main():
    np.random.seed(42)

    # 状態ベクトル次元 (nx=2)、観測ベクトル次元 (nz=1)
    nx, nz = 2, 1

    # システム行列
    F = np.array([[1, 0.1],
                  [0,   1  ]])
    G = np.array([[0],
                  [1]])
    H = np.array([[1, 0]])

    # 真のQ, R（あくまでシミュレーションデータ生成用）
    Q_true = np.diag([0.0])  # 1次元
    R_true = 0.5

    # 初期状態, 初期共分散
    x0 = np.array([0, 1])
    P0 = np.eye(2)

    # 入力(スカラー), シミュレーションステップ数
    u = 0
    steps = 100

    # シミュレーションデータ生成
    true_states, observations = generate_data(
        F, G, H, Q_true, R_true, x0, u, steps
    )

    # 最適化時に途中経過を保存するリスト
    iteration_results = []

    # 最適化に使うコスト関数 (引数 param = [Q, R])
    def cost_function(params):
        """
        gp_minimize から呼ばれるコスト関数。
        params[0] を Q の対角要素、params[1] を R として扱う。
        """
        Q_est = np.diag([params[0]])
        R_est = params[1]

        # 推定パラメータで KF を実行
        _, mean_nees, var_nees, mean_nis, var_nis = kalman_filter(
            F, G, H, Q_est, R_est, x0, P0, observations, u
        )
        # コスト計算
        cost_val = calc_cost(mean_nees, mean_nis, var_nees, var_nis, nx, nz)

        # 途中結果を保存
        iteration_results.append((params[0], params[1], cost_val, mean_nees, mean_nis))
        return cost_val

    # 探索範囲を定義
    search_space = [
        Real(0.01, 1.0, name="Q"),
        Real(0.1, 1.0, name="R")
    ]

    # ガウス過程最適化を実行
    result = gp_minimize(
        cost_function,
        search_space,
        n_calls=30,
        random_state=42,
        # callback=[VerboseCallback(n_total=30)]
    )

    print("最適化されたパラメータ:")
    print(f"Q: {result.x[0]:.4f}, R: {result.x[1]:.4f}")

    # 最適パラメータでKFを再度実行
    optimal_Q = np.diag([result.x[0]])
    optimal_R = result.x[1]
    estimates_opt, mean_nees_opt, var_nees_opt, mean_nis_opt, var_nis_opt = kalman_filter(
        F, G, H, optimal_Q, optimal_R, x0, P0, observations, u
    )

    # 真のパラメータ (Q_true, R_true) でKFを実行 (評価用)
    estimates_true, mean_nees_true, var_nees_true, mean_nis_true, var_nis_true = kalman_filter(
        F, G, H, Q_true, R_true, x0, P0, observations, u
    )

    print("\n一貫性の評価 (最適パラメータ):")
    print(f"NEES (mean/var): {mean_nees_opt:.4f}/{var_nees_opt:.4f}")
    print(f"NIS  (mean/var): {mean_nis_opt:.4f}/{var_nis_opt:.4f}")

    print("\n一貫性の評価 (真のパラメータ):")
    print(f"NEES (mean/var): {mean_nees_true:.4f}/{var_nees_true:.4f}")
    print(f"NIS  (mean/var): {mean_nis_true:.4f}/{var_nis_true:.4f}")

    # 途中経過の可視化
    q_vals, r_vals, costs, nees_vals, nis_vals = zip(*iteration_results)

    # コスト推移
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(costs) + 1), costs, marker='o', label='Cost Function Value')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Optimization Progress')
    plt.grid()
    plt.legend()
    plt.show()

    # (Q, R) 空間におけるコスト値
    plt.figure(figsize=(10, 5))
    sc = plt.scatter(q_vals, r_vals, c=costs, cmap='viridis', s=50, edgecolor='k')
    plt.colorbar(sc, label='Cost Function Value')
    plt.xlabel('Q Value')
    plt.ylabel('R Value')
    plt.title('Cost Function Landscape')
    plt.grid()
    plt.show()

    # NEES と NIS の推移
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(nees_vals) + 1), nees_vals, marker='o', label='Mean NEES', color='blue')
    plt.plot(range(1, len(nis_vals) + 1), nis_vals, marker='x', label='Mean NIS', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('NEES and NIS Progress')
    plt.grid()
    plt.legend()
    plt.show()

    # 最終的な状態推定と真値・観測値の可視化
    plt.figure(figsize=(10, 5))
    plt.plot(true_states[:, 0], label='True Position', color='green')
    plt.plot(observations.squeeze(), label='Observations', color='red', linestyle='dotted')
    plt.plot(estimates_opt[:, 0], label='Estimated Position (Optimal)', color='blue')
    plt.plot(estimates_true[:, 0], label='Estimated Position (True Params)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Tracking Results')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
