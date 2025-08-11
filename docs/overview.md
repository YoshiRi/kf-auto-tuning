# コードベース概要

```
/
├── config/                ... サンプル設定ファイル
│   └── sample_config.yaml
├── docs/                  ... ドキュメント用画像
├── kf_auto_tuning/        ... 主要な Python モジュール
│   ├── config.py
│   ├── data_generator.py
│   ├── kalman_filter.py
│   ├── models.py
│   ├── optimizer.py
│   └── visualization.py
├── mock/                  ... 実験用のサンプルスクリプト
│   └── kf_autotune_sample1d.py
├── tests/                 ... 単体テスト(最小限)
│   └── test_config_loading.py
├── main.py                ... 実験実行例
└── README.md
```

## 主要ファイル・モジュール

### `config/sample_config.yaml`
カルマンフィルタのシステム設定と最適化設定の例です。行列やパラメータ範囲が YAML 形式で定義されています。

### `kf_auto_tuning/data_generator.py`
`SystemConfig` を基に、疑似的な状態空間モデルのデータを生成するクラスです。乱数シードを指定でき、`generate()` で真の状態系列と観測系列を返します。

### `kf_auto_tuning/kalman_filter.py`
基本的なカルマンフィルタの実装です。`run()` で観測値を入力すると推定結果とメトリクス(NEES/NIS)を計算します。

### `kf_auto_tuning/optimizer.py`
`scikit-optimize` を用いたベイズ最適化クラスです。`optimize()` が `cost_function` を最小化し、最適な Q/R を求めます。

### `main.py`
`ExperimentRunner` クラスを中心に、データ生成→パラメータ最適化→フィルタ実行→結果表示を行います。単一実験の実行例が `main()` に記載されています。

## テスト

`tests/test_config_loading.py` は YAML 読み込みを確認する簡易テストです。
ただし、設定ファイルへのパスが `kf_auto_tuning/sample_config.yaml` を参照しており、実際の配置場所(`config/sample_config.yaml`)と食い違っている点に注意してください。

## 実行方法

1. サンプルスクリプトを実行する場合  
   ```bash
   python3 mock/kf_auto_tune_sample1d.py
   ```
   または  
   ```bash
   python3 main.py
   ```

2. テスト実行例  
   ```bash
   python -m unittest discover tests
   ```

## これから学ぶ際の指針

1. **設定ファイル**  
   `config/sample_config.yaml` を編集し、行列の値や最適化範囲 (`param_bounds`) を変化させると挙動を確認しやすいです。

2. **主要クラスの理解**
   - `DataGenerator` でデータ生成
   - `KalmanFilter` でフィルタリング
   - `ParameterOptimizer` で Q/R を探索  
   各モジュールの API を追っていくと全体の流れがつかめます。

3. **可視化**  
   `ResultVisualizer` により、最適化の進行や推定結果をプロットできます。`main.py` の `plot=True` オプションを参考にしてください。

4. **テスト拡充**  
   既存のテストは設定ファイル読み込みのみです。パラメータ最適化やフィルタの振る舞いを検証するテストを追加すると、改修時の信頼性が高まります。

5. **パスの確認**  
   `tests/test_config_loading.py` の設定ファイルのパスが現状のファイル構成と異なるため、修正が必要です。

以上がこのリポジトリの概要と学習のポイントです。コードを読み進めながら、特に `ExperimentRunner` の流れを追うことで、Kalman フィルタのオートチューニング処理全体を把握できます。お試しください。
