# Powerlaw Node Count Experiment, BA1000 comm版

## 目的

`graph_comm` 方式のノード数実験で見られた訂正情報の不自然な拡散が、ノード数の効果なのか、`comm.csv` 再生成の影響なのかを切り分ける。

## 条件

- ノード数: 2000 / 3000 / 4000
- 平均次数: 約40
- クラスタ条件: low / high
- seed: 1, 2, 3
- strategy: balance / effective_high / certainty_high
- comm.csv: BA1000の `comm.csv` から復元抽出

## 保存先

- ネットワーク定義: `experiments/2026-05-26_powerlaw_node_count_ba_comm/network_definitions/`
- 生成サマリ: `v2/test_2/network/powerlaw_node_count_ba_comm/generation_summary.csv`
- 実行結果本体: `experiments/2026-05-26_powerlaw_node_count_ba_comm/strategy_runs/`
- 分析結果: `experiments/2026-05-26_powerlaw_node_count_ba_comm/strategy_runs/analysis/`
- 互換リンク: `powerlaw_node_count_ba_comm_strategy_runs/`

## スクリプト

- 作成: `scripts/prepare_powerlaw_node_count_ba_comm_experiment.py`
- 実行: `scripts/run_powerlaw_node_count_ba_comm_strategy.sh`

## 状態

実行済み・分析前。

## メモ

グラフ生成seedは `powerlaw_node_count_graph_comm` と揃えているため、対応する条件ではグラフは同じ。`comm.csv` だけをBA1000由来分布に戻している。

分析では、訂正情報の共有が前回のように大きく跳ねるかを重点的に確認する。
