# Powerlaw Degree Experiment

## 目的

平均次数を変えたときに、誤情報共有率・peak利己的行動率・strategy差がどう変わるか確認する。

## 条件

- ノード数: 1000
- 平均次数条件: deg20 / deg30 / deg40
- seed: 1, 2, 3
- strategy: balance / effective_high / certainty_high
- comm.csv: BA1000の `comm.csv` をコピー

## 保存先

- ネットワーク定義: `experiments/2026-05-19_powerlaw_degree/network_definitions/`
- 生成サマリ: `v2/test_2/network/powerlaw_degree/generation_summary.csv`
- 実行結果本体: `experiments/2026-05-19_powerlaw_degree/strategy_runs/`
- 分析結果: `experiments/2026-05-19_powerlaw_degree/strategy_runs/analysis/`
- ネットワーク指標: `experiments/2026-05-19_powerlaw_degree/network_metrics/`
- 互換リンク: `powerlaw_degree_strategy_runs/`, `network_metrics_powerlaw_degree/`

## スクリプト

- 作成: `scripts/prepare_powerlaw_degree_experiment.py`
- 実行: `scripts/run_powerlaw_degree_strategy.sh`

## 状態

採用。

## メモ

平均次数を上げると誤情報共有率とpeak利己的行動率は上がった。一方で、strategy間の基本傾向は残った。
