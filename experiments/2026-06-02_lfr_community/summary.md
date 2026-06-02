# LFRコミュニティ構造実験

## 状態

Rustシミュレーション実行済み。集計・可視化まで完了。研究上の考察はこれから整理する。

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

## 実行後確認

- master log: `experiments/2026-06-02_lfr_community/strategy_runs/logs/run_all_20260602_135438.log`
- 実行数: 9ネットワーク x 3 strategy = 27
- 個別run log: 27本
- Arrow出力: 81ファイル
- 欠損ファイル: 0
- 空ファイル: 0
- master log上のStart/Finished: 27/27
- `=== All runs finished ===`を確認
- `error`, `panic`, `failed`, `traceback`, `abort` のログヒットなし
- `.venv` のpyarrowで全81 Arrowファイルを読み込み確認済み

## 集計・可視化

`notebooks/network_strategy_analysis.ipynb` にLFR節を追加し、既存実験と同じ流れで集計・可視化した。

主な出力は以下。

- `comparison_summary.csv`
- `info_compare_long.csv`
- `strategy_summary.csv`
- `shared_ratio_pivot.csv`
- `certainty_minus_effective_shared_ratio.csv`
- `community_level_shared_ratio_by_strategy.png`
- `community_level_peak_selfish_ratio.png`
- `shared_ratio_by_strategy.png`
- `*_strategy_comparison.png`
- `*_lfr_community_comparison.png`

初見では、コミュニティ分離が弱いほど誤情報共有率と利己的行動ピークが大きい。`effective_high` は訂正情報共有率が高く、`certainty_high` は行動誘導情報共有率が高い傾向が、strong/middle/weakで共通して見える。

## 分析・考察

LFR実験では、従来のBA1000やPowerlaw系よりもFacebookの結果に寄る傾向が出た。ただし、寄っているのは主に「strategy間の差が弱まる」という点であり、共有率の絶対水準までFacebookを再現できたわけではない。

### strategy間ギャップ

`max(strategy) - min(strategy)`で見ると、LFR strongは従来の人工グラフよりギャップが小さい。

| 条件 | peak selfish spread | misinformation spread | corrective spread | behavior-guiding spread |
| :--- | ---: | ---: | ---: | ---: |
| Facebook | 0.000262 | 0.002201 | 0.001750 | 0.018341 |
| BA1000 | 0.002697 | 0.001190 | 0.202050 | 0.077287 |
| Powerlaw cluster high | 0.002253 | 0.001293 | 0.165613 | 0.076373 |
| Powerlaw deg40 high cluster | 0.004740 | 0.002850 | 0.302673 | 0.063960 |
| LFR strong | 0.001820 | 0.000740 | 0.105887 | 0.045390 |
| LFR middle | 0.002280 | 0.002300 | 0.160830 | 0.043637 |
| LFR weak | 0.002087 | 0.000790 | 0.157967 | 0.044960 |

LFR strongの訂正情報ギャップはBA1000やPowerlawより小さいが、Facebookのほぼ無差別な状態には届いていない。行動誘導情報のギャップも、従来人工グラフより小さく、Facebook側に近づいている。

### 値の水準

LFR strongは構造的にはFacebookに近いが、共有率の水準はまだ異なる。

| 条件 | peak selfish mean | misinformation mean | corrective mean | behavior-guiding mean |
| :--- | ---: | ---: | ---: | ---: |
| Facebook | 0.067734 | 0.346797 | 0.336296 | 0.240827 |
| LFR strong | 0.113503 | 0.430588 | 0.612770 | 0.120717 |

つまり、LFR strongはFacebookより誤情報・訂正情報が広がりやすく、行動誘導情報は広がりにくい。したがって、「Facebook的な閉じたコミュニティ構造だけ」でFacebook結果を完全に説明するのはまだ難しい。

### 構造上の違い

LFR strongはmodularityと内部エッジ比ではFacebookに近い。

| 条件 | avg clustering | modularity | internal edge ratio |
| :--- | ---: | ---: | ---: |
| Facebook | 0.606 | 0.835 | 0.962 |
| LFR strong | 0.474 | 0.817 | 0.923 |

一方で、support level候補プールのコミュニティ集中度はFacebookと違う。

| 条件 | top20 largest comm ratio | middle20 largest comm ratio | bottom20 largest comm ratio |
| :--- | ---: | ---: | ---: |
| Facebook | 0.645 | 0.381 | 0.526 |
| LFR strong | 0.278 | 0.213 | 0.293 |

Facebookでは誤情報候補・訂正情報候補が少数コミュニティに強く偏っているが、LFRでは候補がより分散している。この違いにより、LFRでは訂正情報が複数コミュニティから出やすくなり、Facebookより訂正情報共有率が高くなった可能性がある。

### 現時点の解釈

- コミュニティ分離を強めると、strategy差は弱まり、Facebookの傾向に近づく。
- ただし、Facebookの結果は単なる高modularityだけではなく、support level候補プールがどのコミュニティに偏っているかにも依存している可能性が高い。
- 次の検証では、LFR構造を保ったまま、support level上位・中央値付近・下位の候補プールをFacebookに近い集中度で配置する条件を作るとよい。

## 次に実行すること

図と集計表を見ながら、Facebookでstrategy差が弱い理由としてコミュニティ閉じ込めがどこまで説明できるかを考察する。
