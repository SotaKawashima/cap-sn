# LFR Rust実順序target pool実験

## 状態

Rustシミュレーション実行、ログ確認、集計・可視化まで完了。研究上の考察はこれから整理する。

## 目的

前回の `lfr_facebook_pool` 実験では、Facebook候補プールの定義がRustで実際に使われる `indexes_by_level` の同値順序とずれていた。  
この実験では、Facebookの候補プール指標をRust実順序で再定義した目標に合わせ、LFR上でsupport level配置を作った場合に、Facebook寄りの拡散結果になるかを確認する。

## 固定するもの

- strategy: `balance`, `effective_high`, `certainty_high`
- runtime: `v2/test_2/runtime.toml`
- agent: `v2/test_2/agent/agent-type6.toml`
- 各seed内のエッジリストとLFR正解コミュニティ

## 使うLFRグラフ

既存LFR strong (`mu=0.05`) では外部次数比が高すぎ、Facebook Rust実順序のtop20/bottom20目標に合わせにくかった。  
そのため、低い混合度で生成した `mu=0.02` の候補から、目標に近い配置を作れた3 seedを使う。

| source network | source dir | 採用理由 |
| :--- | :--- | :--- |
| `lfr_mu02_seed3` | `v2/test_2/network/lfr_low_external_candidate/` | 既存候補内で総合的に良い |
| `lfr_mu02_seed4` | `v2/test_2/network/lfr_low_external_mu02_extra/` | 追加候補で最も良い |
| `lfr_mu02_seed5` | `v2/test_2/network/lfr_low_external_mu02_extra/` | 追加候補で良い |

## comm method

3 method x 3 seed = 9 networkを作る。3 strategyで実行するため、実行数は27 run。

| method | 意味 |
| :--- | :--- |
| `original` | LFR生成時の `comm.csv` をそのまま使う |
| `random` | support level順位をランダム化する対照条件 |
| `rust_target` | Facebook Rust実順序で再定義したtop20/middle20/bottom20目標へ近づけた配置 |

`rust_target` では、事前探索で選んだtop20/middle20/bottom20各200ノードを使う。support level値は一意にし、Rustの同値タイブレークに依存しないようにした。順位は `top20 -> filler -> middle20 -> filler -> bottom20` の順に置く。

## 目標値

Facebook `comm.csv` をRustの `SupportLevelTable::indexes_by_level` 実順序で読み直した目標値。

| pool | largest comm ratio | external ratio | degree mean | participation |
| :--- | ---: | ---: | ---: | ---: |
| top20 | 0.421 | 0.009 | 36.829 | 0.016 |
| middle20 | 0.610 | 0.033 | 30.859 | 0.050 |
| bottom20 | 0.526 | 0.019 | 33.952 | 0.034 |

## 作成したスクリプト

- 生成: `scripts/prepare_lfr_rust_target_pool_experiment.py`
- 実行: `scripts/run_lfr_rust_target_pool_strategy.sh`

## 生成先

- ネットワーク実体: `v2/test_2/network/lfr_rust_target_pool/`
- network config: `v2/test_2/network/network-lfrrustpool_*.toml`
- 生成summary: `v2/test_2/network/lfr_rust_target_pool/generation_summary.csv`
- 実行結果: `experiments/2026-06-03_lfr_rust_target_pool/strategy_runs/`
- 分析出力: `experiments/2026-06-03_lfr_rust_target_pool/strategy_runs/analysis/`

## 生成コマンド

```bash
.venv/bin/python scripts/prepare_lfr_rust_target_pool_experiment.py
```

## 生成後確認

- 9個の `network-lfrrustpool_*.toml` を生成。
- `generation_summary.csv` は9行。
- 各configが参照する `edgelist.txt`, `comm.csv`, `lfr_communities.csv` は存在。
- `rust_target` のtop20/middle20/bottom20は、事前探索で選んだノード集合と完全一致。
- `python3 -m py_compile scripts/prepare_lfr_rust_target_pool_experiment.py` 成功。
- `bash -n scripts/run_lfr_rust_target_pool_strategy.sh` 成功。

`rust_target` 条件の目標値との比較。

| network | pool | largest target | largest actual | external target | external actual | degree target | degree actual | participation target | participation actual |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lfrrustpool_mu02_rust_target_seed3` | top20 | 0.421 | 0.420 | 0.009 | 0.010 | 36.829 | 37.530 | 0.016 | 0.020 |
| `lfrrustpool_mu02_rust_target_seed3` | middle20 | 0.610 | 0.610 | 0.033 | 0.028 | 30.859 | 37.550 | 0.050 | 0.053 |
| `lfrrustpool_mu02_rust_target_seed3` | bottom20 | 0.526 | 0.525 | 0.019 | 0.017 | 33.952 | 34.755 | 0.034 | 0.034 |
| `lfrrustpool_mu02_rust_target_seed4` | top20 | 0.421 | 0.420 | 0.009 | 0.008 | 36.829 | 32.135 | 0.016 | 0.016 |
| `lfrrustpool_mu02_rust_target_seed4` | middle20 | 0.610 | 0.610 | 0.033 | 0.029 | 30.859 | 33.615 | 0.050 | 0.056 |
| `lfrrustpool_mu02_rust_target_seed4` | bottom20 | 0.526 | 0.525 | 0.019 | 0.019 | 33.952 | 33.990 | 0.034 | 0.037 |
| `lfrrustpool_mu02_rust_target_seed5` | top20 | 0.421 | 0.420 | 0.009 | 0.011 | 36.829 | 38.415 | 0.016 | 0.020 |
| `lfrrustpool_mu02_rust_target_seed5` | middle20 | 0.610 | 0.610 | 0.033 | 0.027 | 30.859 | 35.890 | 0.050 | 0.052 |
| `lfrrustpool_mu02_rust_target_seed5` | bottom20 | 0.526 | 0.525 | 0.019 | 0.018 | 33.952 | 35.350 | 0.034 | 0.034 |

## 実行コマンド

Rustシミュレーションは以下で実行した。

```bash
scripts/run_lfr_rust_target_pool_strategy.sh
```

## 実行後確認

- master logは `experiments/2026-06-03_lfr_rust_target_pool/strategy_runs/logs/run_all_20260603_132630.log`。
- 9 network x 3 strategy = 27実行分の出力ディレクトリがある。
- 個別run logは27本、master logは1本。
- Arrow出力は81ファイルで、各runに`agent`, `info`, `pop`の3ファイルが揃っている。
- 空のArrowファイルは0。
- `.venv`のpyarrowで全81 Arrowファイルを読み込み確認できた。
- master log上の`Start`と`Finished`はいずれも27件。
- master log末尾に`=== All runs finished ===`がある。
- `error`, `panic`, `failed`のログヒットはなかった。

## 集計・可視化

`notebooks/network_strategy_analysis.ipynb`にこの実験セットの節を追加した。既存のLFR support pool偏り実験と同じ形で、読み込み、summary保存、strategy比較図、network比較図、method別の最終共有率図、method別の利己的行動ピーク図、method差の補助表を作る。

### 出力

- 出力先は `experiments/2026-06-03_lfr_rust_target_pool/strategy_runs/analysis/`。
- 集計CSVは `comparison_summary.csv`, `info_compare_long.csv`, `strategy_summary.csv`, `shared_ratio_pivot.csv`。
- 補助表として `method_minus_original_shared_ratio.csv`, `certainty_minus_effective_by_method.csv` を追加した。
- 図は15ファイル、CSVは6ファイル、合計21ファイルを生成した。
