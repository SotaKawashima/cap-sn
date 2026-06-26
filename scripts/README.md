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
| `run_lfr_facebook_pool_strategy.sh` | LFR strongでsupport level候補プールの偏りを変えた実験 | `experiments/2026-06-02_lfr_facebook_pool/strategy_runs/` |
| `run_lfr_rust_target_pool_strategy.sh` | LFR mu=0.02でFacebook Rust実順序target pool配置を使う実験 | `experiments/2026-06-03_lfr_rust_target_pool/strategy_runs/` |
| `run_all_optimize.sh` | BA1000 / Facebook / Wiki-vote の最適化実験をまとめて実行 | `experiments/optimization_*/optimize_runs_auc/` |
| `run_optimize.sh` | BA1000 / Facebook / Wiki-vote の最適化実験。デフォルトはAUC基準 | `experiments/optimization_*/optimize_runs_auc/` |
| `run_ba_1000.sh` | `run_optimize.sh ba_1000` のショートカット | `experiments/optimization_ba1000/optimize_runs_auc/` |
| `run_facebook.sh` | `run_optimize.sh facebook` のショートカット | `experiments/optimization_facebook/optimize_runs_auc/` |
| `run_wiki_vote.sh` | `run_optimize.sh wiki-vote` のショートカット | `experiments/optimization_wiki_vote/optimize_runs_auc/` |

全グラフ、全手法、3 Optuna seedをまとめて回す場合は、以下を使う。

```bash
./scripts/run_all_optimize.sh 100 auc 20260617 3
```

これは、`3グラフ × 4手法 × 3 Optuna seed = 36実験`を順番に実行する。

全trialのraw Arrowを保存し、optseed4-6で追加実験する場合は、以下を使う。

```bash
KEEP_TRIAL_RAW=all ./scripts/run_all_optimize.sh 100 auc raw_20260626 3 4
```

この場合、各trialの `info.arrow`、`pop.arrow`、`agent.arrow` は各runの `result/trials/` に保存される。

旧指標の最終時刻スコアで再実行する場合は、第3引数に`final`を指定する。

```bash
./scripts/run_optimize.sh facebook 100 final
```

同じAUC実験を別ディレクトリに保存したい場合は、第4引数にrun labelを指定する。

```bash
./scripts/run_optimize.sh facebook 100 auc 20260617
```

この場合、出力先は`experiments/optimization_facebook/optimize_runs_auc_20260617/`になる。
デフォルトでは、各手法を3つのOptuna seedで実行する。出力先は同じrunディレクトリ内で、
`gpr_optseed1/`、`gpr_optseed2/`、`gpr_optseed3/`のように分ける。

試験的に1 seedだけ実行したい場合は、第5引数に`1`を指定する。

```bash
./scripts/run_optimize.sh facebook 10 auc test 1
```

optseedの開始番号を変える場合は、第6引数に開始番号を指定する。

```bash
./scripts/run_optimize.sh facebook 100 auc raw_20260626 3 4
```

これは `optseed4`、`optseed5`、`optseed6` を実行する。

最適化samplerのseedは、手法間・Optuna seed間で探索点が被りにくいように別の値を使い、`summary_*.json`と`timing_*.csv`に記録する。
現在の設定は、`optseed1`が`4201`から`4204`、`optseed2`が`5201`から`5204`、`optseed3`が`6201`から`6204`で、末尾の`1..4`をそれぞれ`GPR / CMAES / RANDOM / GA`に対応させる。`optseed4`以降も同じ規則で、`optseed4`は`7201`から`7204`になる。

raw Arrow保存モードは環境変数`KEEP_TRIAL_RAW`で指定する。

| 値 | 保存内容 |
| --- | --- |
| `none` | 従来通り、各trialのArrowを削除 |
| `info-pop` | 各trialの`info.arrow`と`pop.arrow`を保存 |
| `all` | 各trialの`info.arrow`、`pop.arrow`、`agent.arrow`を保存 |

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
| `prepare_lfr_facebook_pool_experiment.py` | LFR strong固定でsupport level候補プールの偏りを変えたcomm.csvを作成 | `v2/test_2/network/lfr_facebook_pool/` |
| `prepare_lfr_rust_target_pool_experiment.py` | 事前探索した選定ノードからFacebook Rust実順序target pool配置のcomm.csvを作成 | `v2/test_2/network/lfr_rust_target_pool/` |
| `generate_comm_from_graph.py` | 既存グラフから comm.csv を作成 | 指定した `--output` |

## 指標計算

| スクリプト | 内容 |
| --- | --- |
| `calc_network_metrics.py` | TOMLで指定されたネットワークのノード数、エッジ数、平均次数、クラスタ係数などを集計する |
| `analyze_facebook_rust_candidate_pools.py` | Facebookのsupport level候補プールをRust実順序で再定義し、集中度・外部接続性を集計する |
| `check_lfr_rust_pool_target_feasibility.py` | LFR上でFacebook Rust実順序の候補プール目標に近いsupport level配置を組めるか探索する |
