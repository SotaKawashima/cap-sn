# Powerlaw Node Count Experiment, graph_comm版

## 目的

ノード数をFacebook規模に近づけたときに、strategy間の傾向が弱まるか確認する。

## 条件

- ノード数: 2000 / 3000 / 4000
- 平均次数: 約40
- クラスタ条件: low / high
- seed: 1, 2, 3
- strategy: balance / effective_high / certainty_high
- comm.csv: `graph_comm.ipynb` と同じ `principled_clustering` 方式で各グラフから生成

## 保存先

- ネットワーク定義: `experiments/2026-05-25_powerlaw_node_count_graph_comm/network_definitions/`
- 生成サマリ: `v2/test_2/network/powerlaw_node_count/generation_summary.csv`
- 実行結果本体: `experiments/2026-05-25_powerlaw_node_count_graph_comm/strategy_runs/`
- 分析結果: `experiments/2026-05-25_powerlaw_node_count_graph_comm/strategy_runs/analysis/`
- 互換リンク: `powerlaw_node_count_strategy_runs/`

## スクリプト

- 作成: `scripts/prepare_powerlaw_node_count_experiment.py`
- 実行: `scripts/run_powerlaw_node_count_strategy.sh`
- comm.csv生成補助: `scripts/generate_comm_from_graph.py`

## 状態

注意 / 不採用候補。

## メモ

ログとArrowファイルは正常だったが、訂正情報の延べ共有回数が不自然に大きくなった。`comm.csv` を各グラフから再生成したことで、発信者配置がこれまでの人工グラフ実験と変わった可能性が高い。

本筋の考察には使わず、`comm.csv` の扱いに関する注意例として残す。
