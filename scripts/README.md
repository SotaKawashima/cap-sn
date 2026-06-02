# scripts

実験セットの作成、実験実行、ネットワーク指標計算に使う補助スクリプトを置く場所。
実験結果の実体は `experiments/` 配下に保存し、旧トップレベル名は互換用リンクとして扱う。

## 実験実行

| スクリプト | 内容 | 出力先 |
| --- | --- | --- |
| `run_ba1000_topology_strategy.sh` | BA1000 の4種類トポロジー比較 | `experiments/2026-05_baseline_ba1000_topology/strategy_runs/` |
| `run_real_network_strategy.sh` | Facebook / Wiki-vote 実ネットワーク比較 | `experiments/2026-05_real_network_strategy/strategy_runs/` |
| `run_powerlaw_cluster_strategy.sh` | クラスタ係数を変えた Powerlaw cluster 実験 | `experiments/2026-05-18_powerlaw_cluster/strategy_runs/` |
| `run_powerlaw_cluster_c06_strategy.sh` | 平均クラスタ係数0.6付近の Powerlaw cluster 実験 | `experiments/2026-05-26_powerlaw_cluster_c06/strategy_runs/` |
| `run_powerlaw_degree_strategy.sh` | 平均次数を変えた Powerlaw cluster 実験 | `experiments/2026-05-19_powerlaw_degree/strategy_runs/` |
| `run_powerlaw_degree_cluster_strategy.sh` | 平均次数とクラスタ係数を同時に変えた実験 | `experiments/2026-05-19_powerlaw_degree_cluster/strategy_runs/` |
| `run_powerlaw_node_count_strategy.sh` | `graph_comm.ipynb` 由来 comm.csv を使ったノード数実験 | `experiments/2026-05-25_powerlaw_node_count_graph_comm/strategy_runs/` |
| `run_powerlaw_node_count_ba_comm_strategy.sh` | BA1000 由来 comm.csv をリサンプリングしたノード数実験 | `experiments/2026-05-26_powerlaw_node_count_ba_comm/strategy_runs/` |
| `run_lfr_community_strategy.sh` | LFRのコミュニティ混合度を変えた実験 | `experiments/2026-06-02_lfr_community/strategy_runs/` |
| `run_optimize.sh` | BA1000 / Facebook / Wiki-vote の最適化実験 | `experiments/optimization_*/optimize_runs/` |
| `run_ba_1000.sh` | `run_optimize.sh ba_1000` のショートカット | `experiments/optimization_ba1000/optimize_runs/` |
| `run_facebook.sh` | `run_optimize.sh facebook` のショートカット | `experiments/optimization_facebook/optimize_runs/` |
| `run_wiki_vote.sh` | `run_optimize.sh wiki-vote` のショートカット | `experiments/optimization_wiki_vote/optimize_runs/` |

## 実験セット作成

| スクリプト | 内容 | 主な生成先 |
| --- | --- | --- |
| `prepare_powerlaw_cluster_experiment.py` | クラスタ係数条件別の Powerlaw cluster グラフ作成 | `v2/test_2/network/powerlaw_cluster/` |
| `prepare_powerlaw_cluster_c06_experiment.py` | 平均クラスタ係数0.6付近の Powerlaw cluster グラフ作成 | `v2/test_2/network/powerlaw_cluster_c06/` |
| `prepare_powerlaw_degree_experiment.py` | 平均次数条件別の Powerlaw cluster グラフ作成 | `v2/test_2/network/powerlaw_degree/` |
| `prepare_powerlaw_degree_cluster_experiment.py` | 平均次数・クラスタ係数条件別グラフ作成 | `v2/test_2/network/powerlaw_degree_cluster/` |
| `prepare_powerlaw_node_count_experiment.py` | ノード数変更グラフと comm.csv 作成 | `v2/test_2/network/powerlaw_node_count/` |
| `prepare_powerlaw_node_count_ba_comm_experiment.py` | BA1000 comm.csv 分布リサンプリング版のノード数変更グラフ作成 | `v2/test_2/network/powerlaw_node_count_ba_comm/` |
| `prepare_lfr_community_experiment.py` | LFRグラフ、support level用comm.csv、LFR正解コミュニティを作成 | `v2/test_2/network/lfr_community/` |
| `generate_comm_from_graph.py` | 既存グラフから comm.csv を作成 | 指定した `--output` |

## 指標計算

| スクリプト | 内容 |
| --- | --- |
| `calc_network_metrics.py` | TOMLで指定されたネットワークのノード数、エッジ数、平均次数、クラスタ係数などを集計する |
