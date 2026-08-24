# 第8段階：未使用テストseedによる最終評価

## 1. この段階の目的

候補選択後の性能を、選択に一切使用していないデータで評価し、最適化中に偶然良く見えた候補が選ばれる楽観バイアス（optimizer's curse）と、検証データへの過適合を抑える。

## 2. 実験条件の概要

- 第7段階で固定した最終候補のみを評価する。
- 無介入条件と代表条件を同時に評価する。
- 探索・検証で使っていないテストseedブロックを使う。
- 候補を追加したり、テスト結果を見て候補を変更したりしない。

## 3. 最終的に報告する値

- $J_{\mathrm{cum}}$ の平均と95%信頼区間。
- 無介入条件に対する $\Delta_G$、$\eta_G$。
- 代表条件に対する改善量。
- seedブロックごとの効果方向。
- 確実性・有効性の最終候補値または良好領域。
- ピーク、発生時刻等の補助指標。

## 4. 最終結論の強さ

- $\Delta_G$ と $\eta_G$ が正で、信頼区間も0より上なら、無介入条件より抑制したと比較的強く主張する。
- 推定値は正だが信頼区間が0を含む場合は、減少傾向とする。
- 代表条件との差が小さい場合、介入効果と最適化による追加効果を分けて記述する。
- 候補間差が小さい場合、厳密な単一最適値ではなく良好領域として示す。

## 5. 完了条件

- 未使用データによる最終性能が確定している。
- 最適化による介入効果と追加改善効果を分けて説明できる。
- 最終候補をテスト結果に応じて再選択していない。

## 6. 用語

| 用語 | この段階での意味 |
| --- | --- |
| 最終テストseed | 探索と候補選択のどちらにも使っていない`50001`から`50005` |
| seedブロック | 同一シミュレータseedで行う $M=100$ 回の反復 |
| 主候補 | 第7段階の選択順1位。最終テスト前に各ネットワーク1点を固定し、主結論に用いる |
| 副候補 | 選択順2位・3位。別領域の再現性を記述するが、テスト後に主候補と差し替えない |
| 探索的フォールバック | 第7段階の適格条件を満たす候補がなかったFacebookで、確認のために残した候補 |
| out-of-sample評価 | 候補生成・候補選択に使用していないデータによる評価 |
| paired比較 | 同一ネットワーク、同一seed、同一反復番号の候補と比較条件を対応させる比較 |
| 95% CI | seed間とseed内反復の不確実性を含めた95%信頼区間 |

第7段階のデータは候補選択に使用済みである。その値を最終性能とすると、良く見えた候補を選んだことによる楽観バイアスが残る。第8段階では未使用データで固定候補を評価し、**テスト結果を見た後に候補を追加、削除、順位変更、差し替え、再調整しない。**

## 7. 第7段階から固定した候補

追跡用の候補一覧を[stage8_final_candidate_pool_v1.csv](../../experiment_protocols/stage8_final_candidate_pool_v1.csv)へ固定した。

- 候補数：9件
- ネットワークごとの候補数：3件
- SHA-256：`13b5b4511b643e63f50a3fa687cfed8dc31f4728d3a91974c7d9ab4e7d16f703`
- 元データ：[selected_candidates.csv](../../experiments/summer_2026/stage7_candidate_validation/20260824_101150_candidate_validation_v01/candidate_validation_analysis_v01/tables/selected_candidates.csv)
- 元データのSHA-256：`b6c43b80c3dc1e44898afe053608376f0cae1724256706b270225ed29a930bf6`

| ネットワーク | 順位 | 候補ID | 確実性 | 有効性 | 領域 | 事前の位置付け |
| --- | ---: | --- | ---: | ---: | --- | --- |
| BA1000 | 1 | `cand_cma_es_r03` | 0.7947 | 0.8157 | `region_07` | 主候補 |
| BA1000 | 2 | `cand_bo_gp_r04` | 0.6969 | 0.9928 | `region_05` | 副候補 |
| BA1000 | 3 | `cand_bo_gp_r02` | 0.5034 | 0.9927 | `region_01` | 副候補 |
| Facebook | 1 | `cand_bo_gp_r06` | 0.7109 | 0.8575 | `region_05` | 主探索候補 |
| Facebook | 2 | `cand_bo_gp_r01` | 0.6614 | 0.8815 | `region_04` | 副探索候補 |
| Facebook | 3 | `cand_bo_gp_r03` | 0.6211 | 0.5428 | `region_03` | 副探索候補 |
| Wiki-vote | 1 | `cand_cma_es_r04` | 0.7442 | 0.7482 | `region_04` | 主候補 |
| Wiki-vote | 2 | `cand_random_search_r04` | 0.7764 | 0.5068 | `region_05` | 副候補 |
| Wiki-vote | 3 | `cand_random_search_r03` | 0.8448 | 0.5016 | `region_08` | 副候補 |

BA1000とWiki-voteでは、第7段階の適格条件を満たした候補から選択した。Facebookは適格候補が0件だったため、3件とも`no_qualified_candidate_exploratory_fallback`である。Facebookの最終結果が良くても、第7段階の判定を事後的に「適格だった」と変更しない。

## 8. 比較条件

各ネットワークで3候補と次の3条件を同じ最終テストseedで評価する。

| 条件ID | 内容 | 最終評価での役割 |
| --- | --- | --- |
| `none` | 行動誘導情報を投入しない | 介入そのものによる抑制効果の基準 |
| `legacy_balance` | 確実性0.8、有効性0.8 | 春学期代表条件に対する最適化の追加改善の基準 |
| `prior_high` | 修正済みの先行研究高説得力CSVをそのまま使用 | 先行研究パッケージとの外部比較 |

`prior_high`には[98_98.csv](../../v2/test_2/strategy/inhibition_opinion/98_98.csv)を用いる。SHA-256は`f850c9b911d125eb7e0da1debac3f8d5cf157d651359bd8ec7f34dd4a4ab3745`である。

`prior_high`は確実性・有効性以外の意見成分も含む先行研究の完全な設定である。候補が`prior_high`を上回るかは報告するが、本研究の2設計変数による最適化の合否条件にはしない。

## 9. 事前固定した実験設計

正式仕様は[stage8_final_evaluation_v1.json](../../experiment_protocols/stage8_final_evaluation_v1.json)に固定した。

| 項目 | 設定 |
| --- | --- |
| ネットワーク | `ba1000`、`facebook`、`wiki_vote` |
| 候補数 | ネットワークごとに3、合計9 |
| 比較条件 | ネットワークごとに3 |
| 条件数 | ネットワークごとに6 |
| 最終テストseed | `50001`、`50002`、`50003`、`50004`、`50005` |
| 反復数 | 1 seedブロック当たり $M=100$ |
| 保存raw | `pop`のみ |
| 実行数 | $3 \times 6 \times 5 = 90$ run |
| 総反復数 | $90 \times 100 = 9{,}000$ 回 |
| bootstrap反復 | 10,000回 |
| bootstrap seed | `92001` |

seed群は次のように分離する。

| 用途 | seed |
| --- | --- |
| 開発・固定条件確認 | `20001`から`20005` |
| 第6段階の探索 | `30001` |
| 第7段階の候補選択 | `40001`から`40003` |
| 第8段階の最終評価 | `50001`から`50005` |

全条件で、最適化対象である確実性・有効性と明示した比較介入以外は、先行研究に合わせた現在の固定条件を維持する。ネットワーク内では同じseed、同じ反復数、同じエージェント設定、同じ情報設定、同じ投入時刻、同じ初期発信者を使用する。

## 10. 最終評価指標

主指標は累積利己的行動率 $J_{\mathrm{cum}}$ である。比較条件 $r$ に対する候補 $c$ の絶対抑制量を、

$$
\Delta_{G,c}^{(r)} = J_{\mathrm{cum},G}^{(r)} - J_{\mathrm{cum},G}^{(c)}
$$

とする。$\Delta > 0$ なら候補の方が利己的行動を抑制し、$\Delta < 0$ なら候補の方が悪化している。

相対抑制率は、

$$
\eta_{G,c}^{(r)} = \Delta_{G,c}^{(r)} / J_{\mathrm{cum},G}^{(r)}
$$

とする。ネットワーク間で基準値が異なるため、効果の大きさは主に $\eta$ で示す。加えて、$N\Delta$ を平均的な利己的行動者数換算として出力する。

補助指標として、ピーク新規利己的行動率、初回・最終発生時刻、50%・90%到達時刻、時間重心、発生期間、活動ステップ数、利己的行動0回の反復割合を集計する。これらは主目的関数を置き換えず、効果がどのような時系列過程で生じたかを説明するために用いる。

## 11. 信頼区間と最終解釈

候補と比較条件は、同一ネットワーク、同一seed、同一反復番号で対応させる。95% CIには階層paired bootstrapを用いる。

1. 5個のseedブロックを復元抽出する。
2. 選ばれた各seedブロック内で100反復を復元抽出する。
3. 候補と比較条件は同じ抽出位置を使用する。
4. 10,000回の再標本化から95%区間を得る。

| 判定 | 条件 | 記述 |
| --- | --- | --- |
| `reduction` | $\Delta$ の95% CI下限が0より大きい | 無介入より抑制したことが支持された |
| `reduction_tendency_with_uncertainty` | 推定値は正だがCIが0を含む | 減少傾向だが不確実性が残る |
| `increase` | $\Delta$ の95% CI上限が0より小さい | 無介入より増加したことが支持された |
| `no_clear_reduction` | 上記以外 | 明確な減少は確認できない |

実質的最小効果量は事前設定しない。効果量とCIを連続値として提示し、結果を見た後に有利な閾値を作らない。

## 12. 候補を再選択しないための境界

1. 選択順1位を各ネットワークの主候補とする。
2. 選択順2位・3位は、別領域の副候補として全件報告する。
3. 主結論は主候補の未使用seed評価に基づく。
4. 副候補は、別領域でも方向が再現するかを記述するために用いる。
5. 副候補の結果が主候補より良くても、主候補と差し替えない。
6. 第8段階の結果を候補調整へ戻さない。
7. 全9候補の結果を削除せず出力する。

最終`decision.json`には、`candidate_selection_frozen_before_test=true`、`candidates_reselected=false`、`test_results_used_for_candidate_selection=false`を保存する。

## 13. 実装したファイル

| ファイル | 役割 |
| --- | --- |
| [stage8_final_candidate_pool_v1.csv](../../experiment_protocols/stage8_final_candidate_pool_v1.csv) | 第7段階から固定した9候補と事前報告role |
| [stage8_final_evaluation_v1.json](../../experiment_protocols/stage8_final_evaluation_v1.json) | seed、比較条件、推論規則、再選択禁止を固定した正式protocol |
| [run_stage8_final_evaluation.py](../../run_stage8_final_evaluation.py) | 90 runのdry-run、本実行、再開、execution plan更新 |
| [final_evaluation_analysis.py](../../analysis/final_evaluation_analysis.py) | 最終効果、補助指標、検証値との差、ネットワーク別結論の生成 |
| [analyze_stage8_final_evaluation.py](../../analyze_stage8_final_evaluation.py) | 品質監査から正式な最終判定までを実行するCLI |
| [candidate_validation_analysis.py](../../analysis/candidate_validation_analysis.py) | Stage 7とStage 8でraw監査・階層bootstrapを共用できるよう後方互換で一般化 |
| [test_stage8_final_evaluation.py](../../tests/test_stage8_final_evaluation.py) | 候補固定、seed境界、実行モード、raw監査、CI、再選択禁止のテスト |
| [第8段階_未使用テストseedによる最終評価.ipynb](../../notebooks/第8段階_未使用テストseedによる最終評価.ipynb) | 正式分析後の表確認と可視化 |
| [scripts/README.md](../../scripts/README.md) | 第8段階実行系の簡易説明 |

## 14. データ品質条件

- 90 runがすべて`completed`である。
- 各runに100反復分の`pop.arrow`がある。
- `info.arrow`と`agent.arrow`を保存していない。
- ネットワーク、条件、seed、確実性、有効性がprotocolと一致する。
- 9候補の値、順位、role、候補CSVハッシュが事前固定値と一致する。
- `prior_high`の既存CSVハッシュが固定値と一致する。
- `pop.arrow`から再計算した指標が保存済みCSV、summary、manifestと許容誤差`1e-12`以内で一致する。
- 候補と比較条件のpairedキーに欠損や重複がない。
- 各条件に5 seed、各seedに100反復が揃っている。

1件でも不一致があれば、最終推論を行わず分析を失敗として停止する。

## 15. 正式分析の出力

正式分析は`<experiment_root>/final_evaluation_analysis_v01/`へ次を出力する。

| 出力 | 内容 |
| --- | --- |
| `analysis_manifest.json` | 分析コード、入力、ハッシュ、成否 |
| `analysis_summary.json` | run数、反復数、候補数、再選択の有無、判定概要 |
| `decision.json` | 主候補のネットワーク別結論と固定候補全件 |
| `tables/data_audit.csv` | runごとの品質検査結果 |
| `tables/run_inventory.csv` | 90 runの一覧 |
| `tables/iteration_metrics.parquet` | 9,000反復の再計算済み指標 |
| `tables/condition_seed_summary.csv` | 条件・seed別の平均指標 |
| `tables/condition_summary.csv` | 条件別の $J_{\mathrm{cum}}$、CI、時系列補助指標 |
| `tables/candidate_seed_performance.csv` | 候補の最終seed別効果方向 |
| `tables/candidate_effects_vs_none.csv` | 無介入に対する最終介入効果 |
| `tables/candidate_effects_vs_legacy_balance.csv` | 春学期代表条件に対する追加改善 |
| `tables/candidate_effects_vs_prior_high.csv` | 先行研究パッケージとの外部比較 |
| `tables/validation_test_comparison.csv` | 第7段階検証値とテスト値の差。再選択には使わない |
| `tables/final_candidate_results.csv` | 9候補の最終結果を統合した表 |
| `tables/network_conclusions.csv` | 事前指定主候補によるネットワーク別結論 |

## 16. 可視化

[第8段階_未使用テストseedによる最終評価.ipynb](../../notebooks/第8段階_未使用テストseedによる最終評価.ipynb)は、完了済みの最新分析を読み込み、次を確認する。

1. ネットワーク別の主候補と最終結論。
2. 全9候補の無介入に対する相対抑制率と95% CI。
3. `legacy_balance`に対する追加改善。
4. 第7段階の検証値と第8段階のテスト値の差。
5. 主候補の5 seedブロック別効果方向。
6. 確実性・有効性空間における固定候補の位置。

丸印を主候補、四角印を副候補として表示する。候補間を補間せず、未評価点まで良好であるとは解釈しない。

## 17. 実装時の検証結果

2026年8月24日時点で、実験前の実装確認を行った。

- 第8段階専用テスト9件：成功。
- 第7段階の既存テスト10件：成功。
- リポジトリ全体の単体テスト105件：成功。
- dry-run：`full_run_count=90`、`selected_run_count=90`。
- 各ネットワーク30 run、各最終テストseed18 run。
- 無介入15 run、既存CSVの`prior_high` 15 run、生成介入60 run。
- 候補45 run、比較条件45 run。
- 合成データで9候補全件、3ネットワーク結論、95% CI、`candidates_reselected=false`まで確認。
- 合成データによる90 run、9,000反復、正式分析12表のend-to-end処理：成功。
- protocol、候補CSV、実行・分析Python、notebook JSONの構文を確認。

## 18. MacBookでのcommit・push手順

第7段階の未commit分と第8段階の実装を、対象ファイルを明示して同じcommitに含める。`notes/`と`notebooks/`は`.gitignore`対象であるため、新規作成した第8段階の記録とnotebookだけは`git add -f`を使う。

### 18.1 対象ファイルをステージする

```bash
cd /Users/sota/projects/cap-sn

git status --short

git add \
  analysis/candidate_validation_analysis.py \
  analysis/final_evaluation_analysis.py \
  analyze_stage8_final_evaluation.py \
  experiment_protocols/stage8_final_candidate_pool_v1.csv \
  experiment_protocols/stage8_final_evaluation_v1.json \
  run_stage8_final_evaluation.py \
  scripts/README.md \
  tests/test_stage8_final_evaluation.py \
  'notes/notes_summer/第7段階：検証seedによる候補選択.md' \
  'notebooks/第7段階_検証seedによる候補選択.ipynb'

git add -f \
  'notes/notes_summer/第8段階：未使用テストseedによる最終評価.md' \
  'notebooks/第8段階_未使用テストseedによる最終評価.ipynb'

git -c core.quotePath=false --no-pager diff --cached --name-status
git --no-pager diff --cached --check
```

`git -c core.quotePath=false --no-pager diff --cached --name-status`に上記12ファイルだけが表示されることを確認する。意図しないファイルがあれば、その時点でcommitせず確認する。

### 18.2 テスト後にcommit・pushする

```bash
cd /Users/sota/projects/cap-sn

.venv/bin/python -m unittest discover -s tests -q

git commit -m "Add Stage 8 final evaluation pipeline"
git push origin main

git log -1 --oneline
git rev-parse HEAD
git status --short
```

最後の`git status --short`が空であれば、MacBook側のcommit・pushは完了である。表示されたcommit hashは、共用PCで`git pull`した後の`git rev-parse HEAD`と照合する。

## 19. 共用PCでの実行手順

### 19.1 同期と事前確認

```bash
cd ~/cap-sn
git pull --ff-only origin main
git rev-parse HEAD
git status --short
cargo build --release --locked -p v2
.venv/bin/python -m unittest discover -s tests -q
git status --short
```

最後の`git status --short`が空であることを確認する。

### 19.2 experiment IDを固定してdry-runする

```bash
cd ~/cap-sn

STAGE8_ID="$(date '+%Y%m%d_%H%M%S')_final_evaluation_v01"
printf '%s\n' "$STAGE8_ID" > "$HOME/cap_sn_stage8_id.txt"
echo "STAGE8_ID=$STAGE8_ID"

DRY_RUN="$HOME/${STAGE8_ID}_dry_run.json"

.venv/bin/python run_stage8_final_evaluation.py \
  --experiment-id="$STAGE8_ID" \
  --dry-run > "$DRY_RUN"

.venv/bin/python - "$DRY_RUN" <<'PY'
import json
import sys
from collections import Counter

data = json.load(open(sys.argv[1]))
commands = data["commands"]
networks = Counter(c[c.index("--network") + 1] for c in commands)
seeds = Counter(int(c[c.index("--simulator-seed") + 1]) for c in commands)

print("full_run_count =", data["full_run_count"])
print("selected_run_count =", data["selected_run_count"])
print("networks =", dict(networks))
print("seeds =", dict(seeds))

assert data["full_run_count"] == 90
assert data["selected_run_count"] == 90
assert networks == Counter({"ba1000": 30, "facebook": 30, "wiki_vote": 30})
assert seeds == Counter({50001: 18, 50002: 18, 50003: 18, 50004: 18, 50005: 18})
assert sum("--no-intervention" in c for c in commands) == 15
assert sum("--intervention-opinion-csv" in c for c in commands) == 15
assert sum("--certainty" in c for c in commands) == 60
print("Stage 8 dry-run check: OK")
PY
```

### 19.3 本実行する

```bash
cd ~/cap-sn

STAGE8_ID="$(cat "$HOME/cap_sn_stage8_id.txt")"
LOG_FILE="$HOME/${STAGE8_ID}.log"

git status --short

.venv/bin/python -u run_stage8_final_evaluation.py \
  --experiment-id="$STAGE8_ID" \
  2>&1 | tee "$LOG_FILE"
```

中断後は同じID、Git commit、protocol、候補CSVで再開する。

```bash
.venv/bin/python -u run_stage8_final_evaluation.py \
  --experiment-id="$STAGE8_ID" \
  --resume \
  2>&1 | tee -a "$LOG_FILE"
```

## 20. 実験完了後の確認と正式分析

### 20.1 実行数を確認する

```bash
cd ~/cap-sn

STAGE8_ID="$(cat "$HOME/cap_sn_stage8_id.txt")"
PLAN="experiments/summer_2026/stage8_final_evaluation/${STAGE8_ID}/final_evaluation_execution_plan.json"

.venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("experiment_id =",d["experiment_id"]); print("stage =",d["stage"]); print("counts =",d["counts"]); assert d["stage"]=="stage8_final_evaluation"; assert d["counts"]=={"total":90,"completed":90,"pending":0,"other":0}; print("Stage 8 execution check: OK")' "$PLAN"
```

### 20.2 正式分析を実行する

```bash
cd ~/cap-sn

STAGE8_ID="$(cat "$HOME/cap_sn_stage8_id.txt")"
EXPERIMENT_ROOT="experiments/summer_2026/stage8_final_evaluation/${STAGE8_ID}"

.venv/bin/python analyze_stage8_final_evaluation.py \
  --experiment-root "$EXPERIMENT_ROOT" \
  --protocol experiment_protocols/stage8_final_evaluation_v1.json \
  --analysis-id final_evaluation_analysis_v01
```

### 20.3 分析判定を確認する

```bash
cd ~/cap-sn

STAGE8_ID="$(cat "$HOME/cap_sn_stage8_id.txt")"
ANALYSIS_ROOT="experiments/summer_2026/stage8_final_evaluation/${STAGE8_ID}/final_evaluation_analysis_v01"

.venv/bin/python - \
  "$ANALYSIS_ROOT/analysis_manifest.json" \
  "$ANALYSIS_ROOT/analysis_summary.json" \
  "$ANALYSIS_ROOT/decision.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
summary = json.load(open(sys.argv[2]))
decision = json.load(open(sys.argv[3]))

print("analysis_status =", manifest["status"])
print("run_count =", summary["run_count"])
print("valid_run_count =", summary["valid_run_count"])
print("iteration_metric_count =", summary["iteration_metric_count"])
print("candidate_count =", summary["candidate_count"])
print("decision_status =", decision["status"])
print("final_test_seeds =", decision["final_test_seeds"])
print("candidates_reselected =", decision["candidates_reselected"])

assert manifest["status"] == "completed"
assert summary["run_count"] == 90
assert summary["valid_run_count"] == 90
assert summary["iteration_metric_count"] == 9000
assert summary["candidate_count"] == 9
assert decision["status"] == "final_evaluation_complete"
assert decision["final_test_seeds"] == [50001, 50002, 50003, 50004, 50005]
assert decision["candidate_selection_frozen_before_test"] is True
assert decision["candidates_reselected"] is False
assert decision["test_results_used_for_candidate_selection"] is False

for row in decision["networks"]:
    print()
    print(row["network"], row["evidence_scope"], row["conclusion_code"])
    print(" primary =", row["primary_condition_id"])
    print(" eta vs none =", row["primary_none_effect"])
    assert len(row["frozen_candidates"]) == 3

print("Stage 8 analysis check: OK")
PY
```

## 21. 現在の進捗

2026年8月24日時点では、次まで完了している。

- 第7段階の正式出力から9候補を追跡用CSVへ固定。
- 主候補、副候補、Facebookの探索的フォールバックを実行前に固定。
- 最終seed、比較条件、CI、解釈規則、再選択禁止をprotocolへ固定。
- 新規実行スクリプト、正式分析、テスト、README、可視化notebookを実装。
- dry-runと合成分析を検証。

残作業は、ユーザ側でのcommit・push、共用PCでの90 run実行、正式分析、MacBookへの転送、可視化、結果の研究上の解釈である。
