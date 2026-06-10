# Experiments

このファイルは、実験ファイル群の台帳として使う。

実験結果の実データは `experiments/` 配下へ移動した。既存スクリプトやnotebookの参照を壊さないように、元のトップレベルのディレクトリ名には互換用シンボリックリンクを置いている。

ネットワーク定義は `v2/test_2/network/` から参照される設定が多いため、現時点では移動せず、各実験ディレクトリからリンクで参照する。

## 状態の見方

| 状態 | 意味 |
| :--- | :--- |
| 採用 | 現時点の考察に使う |
| 集計済み・考察前 | 実行・集計・可視化は終わったが、研究上の解釈は未整理 |
| 実行済み・分析前 | 実行は終わったが、ログ確認・集計・考察が未完了 |
| 実行前 | 実験セット・設定は作成済みだが、Rustシミュレーションは未実行 |
| 注意 | 結果はあるが、実験条件に注意が必要 |
| 不採用候補 | 本筋の考察には使いにくい |
| 未作成 | 今後作る予定 |

## 実験一覧

| 実験ID | 目的 | ネットワーク | comm.csv | 実行結果 | 分析 | 状態 | 備考 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ba1000_topology` | BA1000のトポロジー違いによるstrategy差の確認 | BA1000 4種 | 各BA1000設定 | `experiments/2026-05_baseline_ba1000_topology/strategy_runs/` | `experiments/2026-05_baseline_ba1000_topology/strategy_runs/analysis/` | 採用 | 既存比較の基準 |
| `real_network` | Facebook / Wiki-voteでの固定パラメータ比較 | Facebook, Wiki-vote | 各実ネットワーク由来 | `experiments/2026-05_real_network_strategy/strategy_runs/` | `experiments/2026-05_real_network_strategy/strategy_runs/analysis/` | 採用 | Facebookでstrategy差が弱いことを確認 |
| `powerlaw_cluster` | クラスタ係数だけを変えた影響を見る | n=1000, 平均次数約20, low/mid/high | BA1000コピー | `experiments/2026-05-18_powerlaw_cluster/strategy_runs/` | `experiments/2026-05-18_powerlaw_cluster/strategy_runs/analysis/` | 採用 | 5/18の実験 |
| `powerlaw_degree` | 平均次数だけを変えた影響を見る | n=1000, deg20/30/40 | BA1000コピー | `experiments/2026-05-19_powerlaw_degree/strategy_runs/` | `experiments/2026-05-19_powerlaw_degree/strategy_runs/analysis/` | 採用 | 5/19案A |
| `powerlaw_degree_cluster` | 平均次数40でクラスタ係数を変えた影響を見る | n=1000, deg40 low/high | BA1000コピー | `experiments/2026-05-19_powerlaw_degree_cluster/strategy_runs/` | `experiments/2026-05-19_powerlaw_degree_cluster/strategy_runs/analysis/` | 採用 | 5/19案B |
| `powerlaw_node_count_graph_comm` | ノード数をFacebook規模へ近づける | n=2000/3000/4000, deg40 low/high | graph_comm方式で各グラフから生成 | `experiments/2026-05-25_powerlaw_node_count_graph_comm/strategy_runs/` | `experiments/2026-05-25_powerlaw_node_count_graph_comm/strategy_runs/analysis/` | 注意 / 不採用候補 | 訂正情報が不自然に大きく拡散。comm.csv再生成で発信者配置が変わった可能性 |
| `powerlaw_node_count_ba_comm` | ノード数実験のやり直し | n=2000/3000/4000, deg40 low/high | BA1000由来分布を復元抽出 | `experiments/2026-05-26_powerlaw_node_count_ba_comm/strategy_runs/` | `experiments/2026-05-26_powerlaw_node_count_ba_comm/strategy_runs/analysis/` | 実行済み・分析前 | graphは前回と同じ、comm.csvだけBA1000分布へ戻す |
| `powerlaw_cluster_c06` | クラスタ係数0.6付近の人工グラフで確認 | n=1000, 平均次数約6, clustering約0.6 | BA1000コピー | `experiments/2026-05-26_powerlaw_cluster_c06/strategy_runs/` | `experiments/2026-05-26_powerlaw_cluster_c06/strategy_runs/analysis/` | 注意 | 平均次数は既存条件と揃っていないため探索実験として扱う |
| `optimization_ba1000` | BA1000での最適化実験 | BA1000 | BA1000設定 | `experiments/optimization_ba1000/optimize_runs/` | `experiments/optimization_ba1000/behavior_compare/` | 採用 | 最適化結果と挙動比較 |
| `optimization_facebook` | Facebookでの最適化実験 | Facebook | Facebook由来 | `experiments/optimization_facebook/optimize_runs/` | `experiments/optimization_facebook/behavior_compare/` | 採用 | 最適化結果と挙動比較 |
| `optimization_wiki_vote` | Wiki-voteでの最適化実験 | Wiki-vote | Wiki-vote由来 | `experiments/optimization_wiki_vote/optimize_runs/` | `experiments/optimization_wiki_vote/behavior_compare/` | 採用 | 最適化結果と挙動比較 |
| `lfr_community` | Facebook的なコミュニティ閉じ込めを制御して確認 | LFR n=1000, mu=0.05/0.20/0.40, 平均次数約38-40 | `principled_clustering(G, 2)`由来 | `experiments/2026-06-02_lfr_community/strategy_runs/` | `experiments/2026-06-02_lfr_community/strategy_runs/analysis/` | 集計済み・考察前 | 9ネットワーク x 3 strategy実行済み。集計・可視化は `notebooks/network_strategy_analysis.ipynb` に追加済み |
| `lfr_facebook_pool` | LFR strongでsupport level候補プールの偏りだけを変える | LFR strong seed1-3固定 | original/random/half_facebook/facebook_like | `experiments/2026-06-02_lfr_facebook_pool/strategy_runs/` | `experiments/2026-06-02_lfr_facebook_pool/strategy_runs/analysis/` | 集計済み・考察前 | 12 network x 3 strategy実行済み。ログ確認・集計・可視化は `notebooks/network_strategy_analysis.ipynb` に追加済み |
| `lfr_rust_target_pool` | Facebook Rust実順序のsupport pool集中度・外部次数比に合わせる | LFR mu=0.02 seed3/4/5 | original/random/rust_target | `experiments/2026-06-03_lfr_rust_target_pool/strategy_runs/` | `experiments/2026-06-03_lfr_rust_target_pool/strategy_runs/analysis/` | 集計済み・考察前 | 9 network x 3 strategy実行済み。ログ確認・集計・可視化は `notebooks/network_strategy_analysis.ipynb` に追加済み |

## スクリプト対応表

| 用途 | 作成スクリプト | 実行スクリプト |
| :--- | :--- | :--- |
| クラスタ係数実験 | `scripts/prepare_powerlaw_cluster_experiment.py` | `scripts/run_powerlaw_cluster_strategy.sh` |
| クラスタ係数0.6実験 | `scripts/prepare_powerlaw_cluster_c06_experiment.py` | `scripts/run_powerlaw_cluster_c06_strategy.sh` |
| 平均次数実験 | `scripts/prepare_powerlaw_degree_experiment.py` | `scripts/run_powerlaw_degree_strategy.sh` |
| 平均次数 x クラスタ係数実験 | `scripts/prepare_powerlaw_degree_cluster_experiment.py` | `scripts/run_powerlaw_degree_cluster_strategy.sh` |
| ノード数実験 graph_comm版 | `scripts/prepare_powerlaw_node_count_experiment.py` | `scripts/run_powerlaw_node_count_strategy.sh` |
| ノード数実験 BA1000 comm版 | `scripts/prepare_powerlaw_node_count_ba_comm_experiment.py` | `scripts/run_powerlaw_node_count_ba_comm_strategy.sh` |
| LFRコミュニティ構造実験 | `scripts/prepare_lfr_community_experiment.py` | `scripts/run_lfr_community_strategy.sh` |
| LFR support pool偏り実験 | `scripts/prepare_lfr_facebook_pool_experiment.py` | `scripts/run_lfr_facebook_pool_strategy.sh` |
| LFR Rust実順序target pool実験 | `scripts/prepare_lfr_rust_target_pool_experiment.py` | `scripts/run_lfr_rust_target_pool_strategy.sh` |
| graph_comm方式のcomm.csv生成 | `scripts/generate_comm_from_graph.py` | なし |

## 分析notebook

| notebook | 役割 |
| :--- | :--- |
| `notebooks/network_strategy_analysis.ipynb` | ネットワーク構造制御実験の集計・可視化 |

## 注意点

- `comm.csv` はコミュニティ所属ではなく、Rust側ではsupport levelとして扱われる。
- 誤情報・訂正情報・行動誘導情報の初期発信者は、`comm.csv` のlevel順に依存する。
- そのため、`comm.csv` の生成方法を変えると、ネットワーク構造だけでなく発信者配置も変わる。
- `powerlaw_node_count_graph_comm` はこの影響で訂正情報が不自然に広がった可能性が高いため、本筋の考察には使わない方針。
- LFR実験では、LFR正解コミュニティは `lfr_communities.csv` に保存し、Rustが読む `comm.csv` とは分ける。
- LFR support pool偏り実験では、グラフ構造を固定し、`comm.csv` のsupport level順位だけを変える。
- LFR Rust実順序target pool実験では、support level値を一意にし、Rustの同値タイブレークに依存しない配置にする。
