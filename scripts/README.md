# scripts

実験セットの作成、実験実行、ネットワーク指標計算に使う補助スクリプトを置く場所。
実験結果の実体は `experiments/` 配下に保存し、旧トップレベル名は互換用リンクとして扱う。

## 実験実行

### 夏休み研究プロトコルの新実行系

夏休み以降の固定条件実験と単目的最適化には、リポジトリ直下の次の実行ファイルを用いる。

| 実行ファイル | 用途 |
| --- | --- |
| `run_fixed_condition.py` | 固定した確実性・有効性、または無介入条件の実行 |
| `optimize_single_objective.py` | `bo_gp`、`cma_es`、`random_search`による単目的最適化 |

両者は`analysis/optimization_metrics.py`の共通関数を使用し、累積利己的行動実行者割合`cumulative_selfish_fraction`を計算する。春学期の処理を保存するため、`optimize_test.py`と従来のshellスクリプトは変更せず残す。

無介入条件の実行例：

```bash
.venv/bin/python run_fixed_condition.py \
  --stage stage2_existing_reanalysis \
  --experiment-id 20260810_130000_none_check_v01 \
  --network ba1000 \
  --no-intervention \
  --simulator-seed 101 \
  --iterations 100 \
  --raw-level all
```

固定介入条件の実行例：

```bash
.venv/bin/python run_fixed_condition.py \
  --stage stage2_existing_reanalysis \
  --experiment-id 20260810_131000_fixed_check_v01 \
  --network ba1000 \
  --condition-id fixed_c0800_e0800 \
  --certainty 0.8 \
  --effectiveness 0.8 \
  --simulator-seed 101 \
  --iterations 100 \
  --raw-level all
```

単目的最適化の実行例：

```bash
.venv/bin/python optimize_single_objective.py \
  --stage stage6_optimization \
  --experiment-id 20260810_140000_ba_random_v01 \
  --network ba1000 \
  --method random_search \
  --optimizer-replicate 1 \
  --optimizer-seed 4203 \
  --simulator-seed 101 \
  --iterations 100 \
  --trials 100 \
  --raw-level all
```

`--network`には`ba1000`、`facebook`、`wiki_vote`を指定する。実行単位のディレクトリは上書きされないため、再実行時は新しい実験IDまたはversionを使用する。標準の出力先は次のとおりである。

```text
experiments/summer_2026/<stage>/<experiment_id>/
```

各実行ではruntime、strategy、設定ハッシュ、seed、目的値、処理時間、rawと集計ファイルの場所を`manifest.json`へ保存する。無介入条件は設計変数の下限値とは区別され、Rust実行時に行動誘導情報を有効化する`-e`を付けない。失敗した最適化試行は目的値`1.0`へ置換せず、Optuna上の`FAIL`として記録する。

#### 第3段階のpilot実験

第3段階では、[`run_stage3_pilot.py`](../run_stage3_pilot.py)を使用する。Phase A実行時の旧規則は[`stage3_pilot_v1.json`](../experiment_protocols/stage3_pilot_v1.json)、Phase Aの正式判定は[`stage3_pilot_v2.json`](../experiment_protocols/stage3_pilot_v2.json)に保存している。Phase Bの実行条件と判断規則は[`stage3_pilot_v3.json`](../experiment_protocols/stage3_pilot_v3.json)、Phase A・Bの採用結果と第4段階以降のseed分割は[`stage3_pilot_v4.json`](../experiment_protocols/stage3_pilot_v4.json)に固定している。MacBookでは`--dry-run`とテストだけを行い、実際のpilotはGitHubへ同期したcleanなcommitを共用PCで取得してから実行する。

反復数pilotの実行例：

```bash
.venv/bin/python run_stage3_pilot.py \
  --phase precision \
  --experiment-id 20260812_120000_pilot_precision_v01
```

Phase Aでは、leave-one-seed-block-out selection regretが全ネットワーク・全seedブロックで`0.005`以下となる最小の反復数を採用する。保存済みanalysisの完全な反復表を入力にする場合は、`--source-analysis-root`を指定する。入力表はprotocolの90 runおよび18,000反復と照合される。

```bash
.venv/bin/python analyze_stage3_pilot.py \
  --phase precision \
  --experiment-root experiments/summer_2026/stage3_pilot/<precision_experiment_id> \
  --source-analysis-root experiments/summer_2026/stage3_pilot/<precision_experiment_id>/<source_analysis_id> \
  --protocol experiment_protocols/stage3_pilot_v2.json \
  --analysis-id precision_selection_regret_v03
```

Phase Aで決定した`M=100`を使って、BA1000、Facebook、Wiki-voteの3ネットワークで、3手法・各3 runの評価回数pilotを実行する。実行単位は`3ネットワーク × 3手法 × 3 run = 27 run`である。

```bash
.venv/bin/python run_stage3_pilot.py \
  --phase optimization_budget \
  --iterations 100 \
  --protocol experiment_protocols/stage3_pilot_v3.json \
  --experiment-id <optimization_experiment_id>

.venv/bin/python analyze_stage3_pilot.py \
  --phase optimization_budget \
  --experiment-root experiments/summer_2026/stage3_pilot/<optimization_experiment_id> \
  --protocol experiment_protocols/stage3_pilot_v3.json \
  --analysis-id optimization_budget_v03 \
  --iterations 100
```

評価回数は50、75、100回時点のbest-so-farを比較する。各非最終時点から100回時点までの改善量について、各ネットワーク・各手法の3 run中2 run以上が`0.005`以下となる最小時点を採用する。3 runではこれは中央値が`0.005`以下であることと等価であり、中央値も診断値として出力する。9つのネットワーク×手法の組のうち1つでも75回時点を通過しない場合は100回で十分とはせず、protocolを改訂して100回を超えるpilotへ延長する。この判定は探索予算を決めるためだけに使用し、候補の最終性能は独立したシミュレータseedブロックで評価する。

正式結果では$M=100$、$K=50$を採用した。Phase Aの保存済み反復表から第4段階と再最適化の資源量を再計算する場合は、最終protocol v4を指定する。`--optimization-evaluations`を省略すると、v4に固定した50評価が使われる。

```bash
.venv/bin/python analyze_stage3_pilot.py \
  --phase precision \
  --experiment-root experiments/summer_2026/stage3_pilot/20260812_134236_pilot_precision_v01 \
  --source-analysis-root experiments/summer_2026/stage3_pilot/20260812_134236_pilot_precision_v01/precision_without_delta_v01 \
  --protocol experiment_protocols/stage3_pilot_v4.json \
  --analysis-id final_resource_projection_v04
```

この投影は、第4段階を14条件×5 simulator seed、再最適化を3手法×6 optimizer seed×50評価として計算する。時間はPhase Aのネットワーク別1反復当たり中央値とp90、容量は保存Arrowの中央値に基づく。候補検証と最終テストは候補数が未確定のため、この合計には含めない。

実験出力は`.gitignore`の対象であり、GitHubには同期されない。分析と可視化には[`第3段階_pilot実験の反復数と評価予算.ipynb`](../notebooks/第3段階_pilot実験の反復数と評価予算.ipynb)を用いる。

#### 第4段階の固定確認実験

第4段階では、[`stage4_fixed_confirmation_v1.json`](../experiment_protocols/stage4_fixed_confirmation_v1.json)に固定した3ネットワーク、14条件、5 simulator seedを、[`run_stage4_fixed_confirmation.py`](../run_stage4_fixed_confirmation.py)で順次実行する。1 runは1ネットワーク、1条件、1 simulator seedからなり、各runで100反復を行う。全体は210 run、21,000反復である。

実行前のdry-runは次のように行う。

```bash
.venv/bin/python run_stage4_fixed_confirmation.py \
  --experiment-id <fixed_confirmation_experiment_id> \
  --dry-run
```

`full_run_count`と`selected_run_count`がともに210であることを確認してから、cleanなGit worktreeとrelease版Rust binaryがある共用PCで実行する。

```bash
.venv/bin/python -u run_stage4_fixed_confirmation.py \
  --experiment-id <fixed_confirmation_experiment_id> \
  2>&1 | tee "$HOME/<fixed_confirmation_experiment_id>.log"
```

中断した場合は、同じGit commit、protocol、experiment IDで再開する。

```bash
.venv/bin/python -u run_stage4_fixed_confirmation.py \
  --experiment-id <fixed_confirmation_experiment_id> \
  --resume \
  2>&1 | tee -a "$HOME/<fixed_confirmation_experiment_id>.log"
```

実行計画と進捗は`experiments/summer_2026/stage4_fixed_confirmation/<experiment_id>/fixed_confirmation_execution_plan.json`に記録される。条件、ネットワーク、seedを限定するオプションは確認や分割実行用であり、正式実験の構成自体はprotocolから変更しない。

実行結果をMacBookへ転送した後は、[`analyze_stage4_fixed_confirmation.py`](../analyze_stage4_fixed_confirmation.py)で全210 runをraw Arrowから再計算し、階層的paired bootstrapによる正式分析を行う。

```bash
.venv/bin/python analyze_stage4_fixed_confirmation.py \
  --experiment-root experiments/summer_2026/stage4_fixed_confirmation/<experiment_id> \
  --protocol experiment_protocols/stage4_fixed_confirmation_v1.json \
  --analysis-id fixed_confirmation_analysis_v03
```

分析処理は、manifestとprotocolの照合、`metrics.csv`の再現、`pop.arrow`と`agent.arrow`の整合、情報データの検査を行う。主要比較、全介入条件と無介入の記述比較、固定格子の事前対比、角点交互作用を別表へ保存する。無介入時に情報ラベル3が存在しないことは、行動誘導情報を投入しない正常な状態としてゼロ件に補完する。可視化には[`第4段階_無介入・代表条件・固定設計の確認実験.ipynb`](../notebooks/第4段階_無介入・代表条件・固定設計の確認実験.ipynb)を用いる。

#### `prior_high`の補正実験

先行研究の高説得力条件`prior_high`は、[`98_98.csv`](../v2/test_2/strategy/inhibition_opinion/98_98.csv)を再生成せず、その全列をそのまま使用する。固定条件実行で既存CSVを指定する場合は`--intervention-opinion-csv`を用いる。この引数は`--certainty`および`--effectiveness`と同時には指定できない。実行時にはCSVをrunディレクトリへバイト単位で複製し、入力元と複製先のSHA-256をmanifestへ保存する。

補正対象は、第3段階Phase Aの15 runと第4段階の15 runだけである。その他のPhase A 75 run、第4段階195 run、およびPhase B 27 runは再利用する。補正用protocolは次の2つである。

- [`stage3_pilot_v5_prior_high_correction.json`](../experiment_protocols/stage3_pilot_v5_prior_high_correction.json)
- [`stage4_fixed_confirmation_v2_prior_high_correction.json`](../experiment_protocols/stage4_fixed_confirmation_v2_prior_high_correction.json)

補正runは条件フィルタを付けて実行する。

```bash
.venv/bin/python -u run_stage3_pilot.py \
  --phase precision \
  --protocol experiment_protocols/stage3_pilot_v5_prior_high_correction.json \
  --experiment-id <prior_high_correction_experiment_id> \
  --conditions prior_high

.venv/bin/python -u run_stage4_fixed_confirmation.py \
  --protocol experiment_protocols/stage4_fixed_confirmation_v2_prior_high_correction.json \
  --experiment-id <prior_high_correction_experiment_id> \
  --conditions prior_high
```

転送後の再集計では元データを上書きせず、同じrun keyの`prior_high`だけを補正runへ差し替え、新しいanalysis IDへ保存する。

補正後のPhase Aではselection regret基準を満たす最小prefixは20反復である。一方、連続空間の最適化、`M=100`で実施済みのPhase Bとの整合性、対外的に保守的な評価を行う目的から、運用値は`M=100`に固定する。この区別と採用理由は[`stage3_pilot_v6_prior_high_corrected_decision.json`](../experiment_protocols/stage3_pilot_v6_prior_high_corrected_decision.json)に記録する。補正run自体はv5で実行済みであるため、分析時に`--condition-override-protocol`でv5を明示する。

```bash
.venv/bin/python analyze_stage3_pilot.py \
  --phase precision \
  --experiment-root experiments/summer_2026/stage3_pilot/20260812_134236_pilot_precision_v01 \
  --source-analysis-root experiments/summer_2026/stage3_pilot/20260812_134236_pilot_precision_v01/precision_without_delta_v01 \
  --protocol experiment_protocols/stage3_pilot_v6_prior_high_corrected_decision.json \
  --condition-override-id prior_high \
  --condition-override-root experiments/summer_2026/stage3_pilot/<prior_high_correction_experiment_id> \
  --condition-override-protocol experiment_protocols/stage3_pilot_v5_prior_high_correction.json \
  --analysis-id precision_prior_high_corrected_v06

.venv/bin/python analyze_stage4_fixed_confirmation.py \
  --experiment-root experiments/summer_2026/stage4_fixed_confirmation/20260817_112313_fixed_confirmation_v01 \
  --protocol experiment_protocols/stage4_fixed_confirmation_v1.json \
  --condition-override-id prior_high \
  --condition-override-root experiments/summer_2026/stage4_fixed_confirmation/<prior_high_correction_experiment_id> \
  --condition-override-protocol experiment_protocols/stage4_fixed_confirmation_v2_prior_high_correction.json \
  --analysis-id fixed_confirmation_prior_high_corrected_v05
```

### 第6段階：新目的関数による再最適化

第6段階では、累積利己的行動率`cumulative_selfish_fraction`を目的関数として、BA1000、Facebook、Wiki-voteを個別に再最適化する。正式な実験仕様は[`stage6_reoptimization_v1.json`](../experiment_protocols/stage6_reoptimization_v1.json)へ固定している。

- 1評価当たりのシミュレーション反復数：`M=100`
- 1実行当たりの評価回数：`K=50`
- 手法：GPベースのベイズ最適化、CMA-ES、ランダムサーチ
- optimizer replicate：各手法6個
- 探索用simulator seed：`30001`
- 保存するraw：各試行の`pop.arrow`
- 合計：`3ネットワーク × 3手法 × 6反復 = 54実行、2700試行`

実行には[`run_stage6_reoptimization.py`](../run_stage6_reoptimization.py)を用いる。まずdry-runを実行する。

```bash
.venv/bin/python run_stage6_reoptimization.py \
  --experiment-id <stage6_experiment_id> \
  --dry-run
```

`full_run_count`と`selected_run_count`がともに54であることを確認後、cleanなGit worktreeとrelease版Rust binaryがある共用PCで実行する。

```bash
.venv/bin/python -u run_stage6_reoptimization.py \
  --experiment-id <stage6_experiment_id> \
  2>&1 | tee "$HOME/<stage6_experiment_id>.log"
```

中断後、すでに完了した実行を飛ばして続ける場合は、同じGit commit、protocol、experiment IDで`--resume`を指定する。

```bash
.venv/bin/python -u run_stage6_reoptimization.py \
  --experiment-id <stage6_experiment_id> \
  --resume \
  2>&1 | tee -a "$HOME/<stage6_experiment_id>.log"
```

`--resume`が自動的に飛ばすのは、50試行すべてが正常完了したrunだけである。中断中だったrunや失敗試行を含むrunは上書きせず停止するため、その場合は保存済みデータを確認してから再実行方法を決める。

実行結果をMacBookへ転送した後、[`analyze_stage6_reoptimization.py`](../analyze_stage6_reoptimization.py)で正式監査と集計を行う。

```bash
.venv/bin/python analyze_stage6_reoptimization.py \
  --experiment-root experiments/summer_2026/stage6_reoptimization/<stage6_experiment_id> \
  --protocol experiment_protocols/stage6_reoptimization_v1.json \
  --analysis-id reoptimization_analysis_v01
```

分析処理は、全54 run、全2700試行、全270,000反復について、manifestとprotocolの一致、試行完遂、適用パラメータ、raw保存方針を検査する。各試行の`pop.arrow`から目的関数を再計算し、保存値と一致した場合だけ収束表と候補プールを出力する。候補プールには各runの最良試行を1件ずつ残す。探索用seedだけで最終候補や抑制効果を確定せず、未使用seedによる候補検証は第7段階で行う。

以下は春学期までの旧実行系である。

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
