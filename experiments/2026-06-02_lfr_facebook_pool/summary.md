# LFR support pool偏り実験

## 状態

実験前準備まで完了。Rustシミュレーションはまだ実行していない。

## 目的

前回のLFR実験では、LFR strongがFacebookに近いコミュニティ構造を持つ一方で、support level候補プールの配置はFacebookと異なっていた。  
この実験では、LFR strongのグラフ構造を固定したまま `comm.csv` だけを変え、初期発信者候補プールの偏りが拡散結果に与える影響を切り分ける。

## 固定するもの

- 元グラフ: `lfr_strong_seed1-3`
- エッジリスト: 既存LFR strongの `edgelist.txt`
- LFR正解コミュニティ: 既存LFR strongの `lfr_communities.csv`
- strategy: `balance`, `effective_high`, `certainty_high`
- runtime: `v2/test_2/runtime.toml`
- agent: `v2/test_2/agent/agent-type6.toml`

## 変えるもの

`comm.csv` のsupport level順位だけを変える。Rust側では、誤情報はtop、訂正情報はbottom、行動誘導情報はmiddleから初期発信者候補を選ぶため、support level順位が初期発信者配置を決める。

## 作成したスクリプト

- 生成: `scripts/prepare_lfr_facebook_pool_experiment.py`
- 実行: `scripts/run_lfr_facebook_pool_strategy.sh`

## 生成先

- ネットワーク実体: `v2/test_2/network/lfr_facebook_pool/`
- network config: `v2/test_2/network/network-lfrpool_*.toml`
- 生成summary: `v2/test_2/network/lfr_facebook_pool/generation_summary.csv`
- 実行結果予定: `experiments/2026-06-02_lfr_facebook_pool/strategy_runs/`

## 生成条件

対象はLFR strongの3 seedのみ。

| 項目 | 値 |
| :--- | :--- |
| source_subdir | `lfr_community` |
| source_level | `strong` |
| output_subdir | `lfr_facebook_pool` |
| seeds | `1, 2, 3` |
| seed_base | `20260602` |
| methods | `original`, `random`, `half_facebook`, `facebook_like` |

生成される実験条件は、4 comm method x 3 seed = 12 network。3 strategyで実行するため、実行数は36 run。

## comm method

| method | 意味 |
| :--- | :--- |
| `original` | 既存LFR strongの `principled_clustering(G, 2)` 由来 `comm.csv` をそのまま使う |
| `random` | support level順位をランダム化し、候補プールをコミュニティ構造から切り離す |
| `half_facebook` | Facebookとoriginalの中間程度に候補プールを偏らせる |
| `facebook_like` | Facebookで観測した候補プール集中度に近づける |

人工的に生成する `random`, `half_facebook`, `facebook_like` では、support level値を一意にし、順位が曖昧にならないようにした。  
順位配置は、top20が先頭200、middle20が中央値付近200、bottom20が末尾200になるように作った。

## target pool metrics

1000ノードなので、top20/middle20/bottom20はいずれも200ノード。

| method | top20最大比 | middle20最大比 | bottom20最大比 | top20 touched | middle20 touched | bottom20 touched |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random` | 目標なし | 目標なし | 目標なし | 目標なし | 目標なし | 目標なし |
| `original` | 既存値 | 既存値 | 既存値 | 既存値 | 既存値 | 既存値 |
| `half_facebook` | 0.450 | 0.290 | 0.410 | 5 | 9 | 6 |
| `facebook_like` | 0.645 | 0.381 | 0.526 | 3 | 9 | 5 |

`facebook_like` の比率は、Facebookで観測した top20=0.645, middle20=0.381, bottom20=0.526 を1000ノード用に移したもの。200ノードの候補プールでは、おおよそ top20=129, middle20=76, bottom20=105ノードを最大コミュニティに置く。

## 生成後確認

生成summary上の平均は以下。

| method | top20最大比 | middle20最大比 | bottom20最大比 | top20 touched | middle20 touched | bottom20 touched |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `facebook_like` | 0.645 | 0.380 | 0.525 | 3.000 | 9.000 | 5.000 |
| `half_facebook` | 0.450 | 0.290 | 0.410 | 5.000 | 9.000 | 6.000 |
| `original` | 0.278 | 0.213 | 0.293 | 7.000 | 8.000 | 6.333 |
| `random` | 0.150 | 0.148 | 0.147 | 13.333 | 13.333 | 13.333 |

## 確認済み

- `python3 -m py_compile scripts/prepare_lfr_facebook_pool_experiment.py` 成功。
- `bash -n scripts/run_lfr_facebook_pool_strategy.sh` 成功。
- 12個の `network-lfrpool_*.toml` を生成。
- 各configが参照する `edgelist.txt`, `comm.csv`, `lfr_communities.csv` は存在。

## 次に実行すること

```bash
./scripts/run_lfr_facebook_pool_strategy.sh
```

実行後、`notebooks/network_strategy_analysis.ipynb` にこの実験セットの集計・可視化を追加する。
