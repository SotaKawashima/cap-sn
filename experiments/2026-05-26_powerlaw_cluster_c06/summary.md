# Powerlaw Cluster C0.6 Experiment

## 目的

Facebookの平均クラスタ係数に近い、平均クラスタ係数0.6付近の人工グラフで、情報共有率と利己的行動率の傾向がどう変わるかを確認する。

## 実験条件

- グラフ生成: NetworkX `powerlaw_cluster_graph`
- ノード数: 1000
- `m`: 3
- triad probability: 1.0
- 目標クラスタ係数: 0.6付近
- seed: 1, 2, 3
- `comm.csv`: BA1000の `v2/test_2/network/ba/ba1000/comm.csv` をコピー
- strategy: `balance`, `effective_high`, `certainty_high`

## 注意

`powerlaw_cluster_graph` でクラスタ係数0.6付近を狙うには、`m=3` 程度まで下げる必要がある。そのため、平均次数は約6になり、これまでの平均次数20/40条件とは異なる。

この実験は、平均次数をそろえた比較ではなく、Facebookに近い高クラスタ係数そのものが拡散傾向を変えるかを見るための探索実験として扱う。

## ファイル

- 作成スクリプト: `scripts/prepare_powerlaw_cluster_c06_experiment.py`
- 実行スクリプト: `scripts/run_powerlaw_cluster_c06_strategy.sh`
- ネットワーク定義: `v2/test_2/network/powerlaw_cluster_c06/`
- 実行結果: `experiments/2026-05-26_powerlaw_cluster_c06/strategy_runs/`

