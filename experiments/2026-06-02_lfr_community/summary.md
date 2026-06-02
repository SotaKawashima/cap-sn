# LFRコミュニティ構造実験

## 状態

実験前準備まで完了。Rustシミュレーションはまだ実行していない。

## 目的

Facebookでstrategy差が弱い要因として、平均次数やクラスタ係数ではなく、コミュニティ内に強く閉じた構造が効いているかを確認する。

## 準備内容

- LFRグラフ生成スクリプト: `scripts/prepare_lfr_community_experiment.py`
- 実行スクリプト: `scripts/run_lfr_community_strategy.sh`
- 生成先: `v2/test_2/network/lfr_community/`
- ネットワーク設定: `v2/test_2/network/network-lfr_*.toml`
- 生成summary: `v2/test_2/network/lfr_community/generation_summary.csv`

LFR正解コミュニティは `lfr_communities.csv` に保存した。Rustが読む `comm.csv` はコミュニティ所属ではなくsupport levelなので、従来と同じく `principled_clustering(G, 2)` から作成した。

## 再現用引数

生成時は以下を実行した。

```bash
.venv/bin/python scripts/prepare_lfr_community_experiment.py
```

デフォルト引数は以下。

```text
num_nodes = 1000
tau1 = 2.5
tau2 = 1.5
average_degree = 30.0
max_degree = 120
min_community = 40
max_community = 150
levels = strong:0.05, middle:0.20, weak:0.40
seeds = 1, 2, 3
seed_base = 20260602
max_iters = 2000
max_retries = 30
output_subdir = lfr_community
comm_communities = 2
comm_index = 0
```

実際のLFR生成seedは `seed_base + level_index * 100000 + seed_index * 1000`。`level_index` は strong=0, middle=1, weak=2。

## 生成条件

| level | mu | seed数 | 平均次数 | 平均クラスタ係数 | modularity | 内部エッジ比 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| strong | 0.05 | 3 | 37.554 | 0.474 | 0.817 | 0.923 |
| middle | 0.20 | 3 | 39.741 | 0.244 | 0.602 | 0.700 |
| weak | 0.40 | 3 | 40.069 | 0.097 | 0.329 | 0.425 |

## 次に実行すること

`scripts/run_lfr_community_strategy.sh` で、9ネットワーク x 3 strategyの27実験を走らせる。その後、`notebooks/network_strategy_analysis.ipynb` で集計する。
