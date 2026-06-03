# LFR support pool偏り実験

## 状態

Rustシミュレーション実行、ログ確認、集計・可視化まで完了。研究上の考察はこれから整理する。

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
- 実行結果: `experiments/2026-06-02_lfr_facebook_pool/strategy_runs/`
- 分析出力: `experiments/2026-06-02_lfr_facebook_pool/strategy_runs/analysis/`

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

## 実行後確認

- master logは`experiments/2026-06-02_lfr_facebook_pool/strategy_runs/logs/run_all_20260602_152041.log`。
- 12 network x 3 strategy = 36実行分の出力ディレクトリがある。
- 個別run logは36本、master logは1本。
- Arrow出力は108ファイルで、各runに`agent`, `info`, `pop`の3ファイルが揃っている。
- 欠損ファイル、空ファイルは0。
- master log上の`Start`と`Finished`はいずれも36件。
- master log末尾に`=== All runs finished ===`がある。
- `error`, `panic`, `failed`のログヒットはなかった。
- `.venv`のpyarrowで全108 Arrowファイルを読み込み確認できた。

## 集計・可視化

`notebooks/network_strategy_analysis.ipynb`にこの実験セットの節を追加した。既存実験と同じ形で、読み込み、summary保存、strategy比較図、network比較図、method別の最終共有率図、method別の利己的行動ピーク図を作る。

### 出力

- 出力先は`experiments/2026-06-02_lfr_facebook_pool/strategy_runs/analysis/`。
- 集計CSVは`comparison_summary.csv`, `info_compare_long.csv`, `strategy_summary.csv`, `shared_ratio_pivot.csv`。
- 補助表として`method_minus_original_shared_ratio.csv`, `certainty_minus_effective_by_method.csv`を追加した。
- 図は18ファイル、CSVは6ファイル、合計24ファイルを生成した。

### 初見

- support poolを`facebook_like`へ寄せると、誤情報の平均共有率は`original`の0.431から0.493へ上がった。
- 行動誘導情報の平均共有率も`original`の0.121から`facebook_like`の0.151へ上がった。
- 平均ピーク利己的行動率は`original`の0.114から`facebook_like`の0.098へ下がった。
- `certainty_high - effective_high`は、訂正情報では`original=-0.106`, `facebook_like=-0.125`、行動誘導情報では`original=+0.045`, `facebook_like=+0.043`だった。

## 次に見ること

support poolの偏りは共有率の絶対水準をかなり動かしている。一方で、strategy間の差は情報種別によって残り方が違うため、Facebook結果に近づいた部分と、まだ差が残る部分を切り分けて考察する。
