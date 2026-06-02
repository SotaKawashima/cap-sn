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
- 分析結果: `experiments/2026-05-26_powerlaw_cluster_c06/strategy_runs/analysis/`

## 生成されたネットワーク

| network | 平均次数 | 平均クラスタ係数 | transitivity | 最大次数 |
| --- | ---: | ---: | ---: | ---: |
| `plc06_seed1` | 5.982 | 0.593 | 0.162 | 119 |
| `plc06_seed2` | 5.982 | 0.597 | 0.108 | 275 |
| `plc06_seed3` | 5.982 | 0.604 | 0.145 | 175 |

平均クラスタ係数は0.6付近に到達した。一方で、平均次数は約6である。

## 実験結果

3 seed平均の主要指標は以下。

| strategy | 誤情報共有率 | 訂正情報共有率 | 観測情報共有率 | 行動誘導情報共有率 | peak利己的行動率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `balance` | 0.0280 | 0.1009 | 0.0128 | 0.0259 | 0.0103 |
| `effective_high` | 0.0282 | 0.1041 | 0.0064 | 0.0232 | 0.0104 |
| `certainty_high` | 0.0285 | 0.0969 | 0.0051 | 0.0313 | 0.0102 |

可視化は以下に保存した。

- `strategy_average_shared_ratio.png`
- `cluster_level_shared_ratio_with_c06.png`
- `plc06_seed*_strategy_comparison.png`
- `*_cluster_c06_topology_comparison.png`

## メモ

- 平均クラスタ係数は0.6付近に到達した。
- ただし平均次数は約6であり、既存のPowerlaw cluster実験の平均次数約20とはそろっていない。
- 共有率とpeak利己的行動率は既存のlow/mid/high条件より大きく低下した。
- この結果は、クラスタ係数0.6の効果だけでなく、平均次数低下による接触機会の減少も含む探索結果として扱う。
- 詳細な考察は `notes/2026-5-26.md` に記録する。
