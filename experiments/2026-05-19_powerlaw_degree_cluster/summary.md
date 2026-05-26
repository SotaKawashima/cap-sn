# Powerlaw Degree x Cluster Experiment

## 目的

平均次数をFacebookに近い約40に固定し、クラスタ係数を変えたときに拡散傾向が変わるか確認する。

## 条件

- ノード数: 1000
- 平均次数: 約40
- クラスタ条件: low / high
- seed: 1, 2, 3
- strategy: balance / effective_high / certainty_high
- comm.csv: BA1000の `comm.csv` をコピー

## 保存先

- ネットワーク定義: `experiments/2026-05-19_powerlaw_degree_cluster/network_definitions/`
- 生成サマリ: `v2/test_2/network/powerlaw_degree_cluster/generation_summary.csv`
- 実行結果本体: `experiments/2026-05-19_powerlaw_degree_cluster/strategy_runs/`
- 分析結果: `experiments/2026-05-19_powerlaw_degree_cluster/strategy_runs/analysis/`
- ネットワーク指標: `experiments/2026-05-19_powerlaw_degree_cluster/network_metrics/`
- 互換リンク: `powerlaw_degree_cluster_strategy_runs/`, `network_metrics_powerlaw_degree_cluster/`

## スクリプト

- 作成: `scripts/prepare_powerlaw_degree_cluster_experiment.py`
- 実行: `scripts/run_powerlaw_degree_cluster_strategy.sh`

## 状態

採用。

## メモ

高平均次数条件でも、クラスタ係数を上げると誤情報共有率とpeak利己的行動率は下がった。ただしstrategy間の基本傾向は残った。
