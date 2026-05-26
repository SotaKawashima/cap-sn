# Powerlaw Cluster Experiment

## 目的

平均次数をBA1000程度に保ったままクラスタ係数を変え、strategyごとの拡散傾向が変わるか確認する。

## 条件

- ノード数: 1000
- 平均次数: 約20
- クラスタ条件: low / mid / high
- seed: 1, 2, 3
- strategy: balance / effective_high / certainty_high
- comm.csv: BA1000の `comm.csv` をコピー

## 保存先

- ネットワーク定義: `experiments/2026-05-18_powerlaw_cluster/network_definitions/`
- 生成サマリ: `v2/test_2/network/powerlaw_cluster/generation_summary.csv`
- 実行結果本体: `experiments/2026-05-18_powerlaw_cluster/strategy_runs/`
- 分析結果: `experiments/2026-05-18_powerlaw_cluster/strategy_runs/analysis/`
- 互換リンク: `powerlaw_cluster_strategy_runs/`

## スクリプト

- 作成: `scripts/prepare_powerlaw_cluster_experiment.py`
- 実行: `scripts/run_powerlaw_cluster_strategy.sh`

## 状態

採用。

## メモ

クラスタ係数を上げると誤情報共有とpeak利己的行動率は下がる傾向があった。ただしstrategy間の基本傾向は大きく変わらなかった。
