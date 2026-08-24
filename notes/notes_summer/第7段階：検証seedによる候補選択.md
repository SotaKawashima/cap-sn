# 第7段階：検証seedによる候補選択

## 1. この段階の目的

第6段階では、新目的関数 $J_{\mathrm{cum}}$ を用いて3ネットワークを再最適化した。ただし、各最適化で得られた最良点は、探索に使用したシミュレータseed `30001`に偶然適合している可能性がある。

第7段階では、探索に使用していない検証seedを用いて第6段階の候補を再評価し、次の第8段階の最終評価へ送る候補または候補領域を固定する。

この段階の目的は、検証seed上の最小値を研究成果として報告することではない。**探索時の偶然性を減らして最終テストへ送る候補を選ぶこと**である。したがって、第7段階の数値は候補選択にのみ使用し、最終性能や最終的な抑制効果は第8段階の未使用seedで評価する。

## 2. 用語

| 用語 | この段階での意味 |
| --- | --- |
| 候補 | 第6段階の各ネットワーク・各最適化手法・各最適化seedで得られた最良の確実性・有効性の組 |
| 検証seedブロック | 同一シミュレータseedで行う $M=100$ 回の反復。今回は `40001`、`40002`、`40003` の3ブロック |
| optimizer's curse | 多数の探索結果から最小値を選ぶことで、偶然良かった点を実力以上に評価してしまう現象 |
| ブロック順位 | 同一ネットワーク・同一検証seed内で、18候補を平均 $J_{\mathrm{cum}}$ の小さい順に並べた順位 |
| paired比較 | 候補と比較条件を、同一ネットワーク・同一seed・同一反復番号で対応させて比較する方法 |
| 候補領域 | 確実性・有効性が近い実測候補をまとめた集合。領域内部の未実測点まで良いと仮定するものではない |

## 3. 入力候補

第6段階では、3ネットワーク、3手法、6最適化seedから、次の54候補が得られた。

\[
3\ \text{ネットワーク}
\times 3\ \text{手法}
\times 6\ \text{最適化seed}
=54\ \text{候補}
\]

各ネットワークには18候補がある。候補はネットワーク固有とし、例えばBA1000で得た候補をFacebookへ適用しない。

検証結果を見る前に候補を減らすと、候補除外の判断に恣意性が入る。そのため、**54候補を事前に削除・統合せず、すべて検証する**。近い候補を領域にまとめる処理は、全候補の検証完了後に行う。

- 固定した候補一覧：[stage7_candidate_pool_v1.csv](../../experiment_protocols/stage7_candidate_pool_v1.csv)
- 候補一覧のSHA-256：`f66b2f551f461bdaf4fdc2cd1b423cf716d3ef2539170e4ec3d33d077604fb64`
- 第6段階の正式候補一覧：[candidate_pool.csv](../../experiments/summer_2026/stage6_reoptimization/20260819_162842_stage6_reoptimization_v01/reoptimization_analysis_v01/tables/candidate_pool.csv)
- 第6段階の正式候補一覧のSHA-256：`98f37150c3a9a2a69908d740d6d76893bf19a13f7d8b1cba98f181e64cc358d3`

固定した候補一覧は、第6段階の正式候補54件と全列を照合済みである。シミュレータへ渡す確実性・有効性には、第6段階で実際に適用した小数第4位の値を使用する。

## 4. 比較条件

各ネットワークで18候補に次の3条件を加え、合計21条件を同じ検証seedで評価する。

| 条件ID | 内容 | 選択での役割 |
| --- | --- | --- |
| `none` | 行動誘導情報を投入しない | 行動誘導情報による抑制効果の必須基準 |
| `legacy_balance` | 確実性 $0.8$、有効性 $0.8$ | 春学期の代表条件に対する最適化の追加改善を測る必須基準 |
| `prior_high` | 修正済みの先行研究高説得力CSVをそのまま使用 | 先行研究パッケージとの外部比較。候補の合否条件にはしない |

`prior_high`には、[98_98.csv](../../v2/test_2/strategy/inhibition_opinion/98_98.csv)を使用する。SHA-256は`f850c9b911d125eb7e0da1debac3f8d5cf157d651359bd8ec7f34dd4a4ab3745`である。

`prior_high`は、確実性・有効性以外の意見成分も含む先行研究の完全な設定である。本研究の2設計変数だけでは同じ条件を必ず再現できないため、`prior_high`を上回ることは候補適格性の必須条件にしない。比較結果は別途報告する。

## 5. 事前固定した実験設計

正式な仕様は[stage7_candidate_validation_v1.json](../../experiment_protocols/stage7_candidate_validation_v1.json)に固定した。

| 項目 | 設定 |
| --- | --- |
| ネットワーク | `ba1000`、`facebook`、`wiki_vote` |
| 候補数 | ネットワークごとに18、合計54 |
| 比較条件 | ネットワークごとに3 |
| 条件数 | ネットワークごとに21 |
| 検証seed | `40001`、`40002`、`40003` |
| 反復数 | 1 seedブロック当たり $M=100$ |
| 保存raw | `pop`のみ |
| 実行数 | $3\times21\times3=189$ run |
| 総反復数 | $189\times100=18{,}900$ 回 |
| 最終テストseed | `50001`から`50005`。第7段階では未使用 |

全候補と比較条件について、ネットワーク内では同じseed系列、同じ反復数、同じエージェント設定、同じ情報設定、同じ投入時刻、同じ初期発信者を使用する。最適化対象である確実性・有効性と、比較条件として明示した介入設定以外は変更しない。

## 6. 評価指標

ネットワーク $G$、条件 $c$、検証seed $s$における100反復の平均目的値を、次のように表す。

\[
\overline{J}_{G,c,s}
=
\frac{1}{100}\sum_{m=1}^{100}J_{\mathrm{cum},G,c,s,m}
\]

比較条件 $r$ に対する候補 $c$ の絶対抑制量は、

\[
\Delta_{G,c}^{(r)}
=
J_{\mathrm{cum},G}^{(r)}-J_{\mathrm{cum},G}^{(c)}
\]

とする。$\Delta>0$なら候補の方が利己的行動を抑制し、$\Delta<0$なら候補の方が悪化している。

比較条件に対する相対抑制率は、

\[
\eta_{G,c}^{(r)}
=
\frac{J_{\mathrm{cum},G}^{(r)}-J_{\mathrm{cum},G}^{(c)}}
{J_{\mathrm{cum},G}^{(r)}}
\]

とする。ネットワーク規模が異なるため、主に$\eta$を用いてネットワーク間で効果の大きさを読み比べる。あわせて、$N\Delta$を時間平均の利己的行動者数換算として出力する。

## 7. 候補選択規則

候補選択規則は検証結果を見る前に固定した。各ネットワークで独立に次の処理を行う。

### 7.1 適格条件

候補を適格とするには、次の両方を満たす必要がある。

1. `none`に対する3 seed平均の絶対抑制量が正で、3 seed中2 seed以上でも正である。
2. `legacy_balance`に対する3 seed平均の絶対改善量が正で、3 seed中2 seed以上でも正である。

これは、平均だけが良く特定の1 seedで大きく悪化する候補を避けつつ、ランダム性の大きいモデルに完全な3 seed一致を要求しない規則である。

95%信頼区間が0を超えることは、候補適格性の必須条件にしない。検証seedが3ブロックであることと、実質的最小効果量を事前に設定していないことから、信頼区間を機械的な合否判定にすると必要以上に厳しくなるためである。効果量と信頼区間は連続量として必ず報告する。

### 7.2 適格候補の順位

適格候補は、次の優先順で並べる。

1. 3 seedにおけるブロック順位の中央値が小さい。
2. 3 seedにおける最悪順位が小さい。
3. 3 seed全体の平均 $J_{\mathrm{cum}}$ が小さい。
4. 同順位なら候補IDの辞書順とする。

この規則では、1 seedだけで極端に良い候補よりも、複数seedで安定して上位にある候補を優先する。

### 7.3 候補領域

適格候補を、確実性・有効性の2次元空間で完全連結法によりまとめる。距離は生の設計変数単位でのユークリッド距離とし、同一領域内の全候補間距離が`0.05`以下になるようにする。

各領域から順位最上位の候補を1点選び、ネットワークごとに最大3領域を第8段階へ送る。手法別の枠は設けず、検証性能だけで選ぶ。

領域は「近い実測候補が複数あった」という要約であり、領域内部の未実測点すべての有効性を保証しない。

### 7.4 適格候補がない場合

適格候補が0件のネットワークでは、順位上位の異なる領域から最大3候補を探索的候補として残す。この場合、判定を`no_qualified_candidate_exploratory_fallback`とし、第6段階の最適化成功を主張しない。第8段階では、候補が本当に改善するかを未使用seedで確認する。

## 8. 統計処理

候補と比較条件は、同一ネットワーク、同一シミュレータseed、同一反復番号で対応付ける。

95%信頼区間には、階層paired bootstrapを用いる。

1. 検証seedブロックを復元抽出する。
2. 選ばれた各seedブロック内で100反復を復元抽出する。
3. 候補と比較条件は同じ抽出位置を使用する。
4. これを10,000回繰り返す。

bootstrap seedは`91001`に固定する。この信頼区間は不確実性の把握に用い、候補の適格・不適格を決める二値閾値にはしない。

## 9. 実装したファイル

| ファイル | 役割 |
| --- | --- |
| [stage7_candidate_pool_v1.csv](../../experiment_protocols/stage7_candidate_pool_v1.csv) | 第6段階から固定した54候補 |
| [stage7_candidate_validation_v1.json](../../experiment_protocols/stage7_candidate_validation_v1.json) | seed、比較条件、選択規則、出力を固定した正式protocol |
| [run_stage7_candidate_validation.py](../../run_stage7_candidate_validation.py) | 189 runのdry-run、本実行、再開、manifest更新 |
| [candidate_validation_analysis.py](../../analysis/candidate_validation_analysis.py) | raw再集計、paired効果量、順位、領域、候補選択 |
| [analyze_stage7_candidate_validation.py](../../analyze_stage7_candidate_validation.py) | 品質監査から候補固定までを実行する分析CLI |
| [test_stage7_candidate_validation.py](../../tests/test_stage7_candidate_validation.py) | protocol、実行コマンド、raw監査、領域形成、選択規則のテスト |
| [第7段階_検証seedによる候補選択.ipynb](../../notebooks/第7段階_検証seedによる候補選択.ipynb) | 正式集計後の確認と可視化 |
| [scripts/README.md](../../scripts/README.md) | Stage 7実行系の簡易説明 |

`run_stage7_candidate_validation.py`は、protocolと候補CSVのハッシュ、実行開始時のGit commit、完了run数を記録する。同じ実験IDを再開する場合も、protocol、候補CSV、Git commitが開始時と一致しなければ停止する。

## 10. データ品質条件

正式分析へ進むには、次をすべて満たす必要がある。

- 189 runがすべて`completed`である。
- 各runに100反復分の`pop.arrow`がある。
- `info.arrow`と`agent.arrow`を保存していない。
- 各runのネットワーク、条件、seed、確実性、有効性がprotocolと一致する。
- `prior_high`のCSVハッシュが固定値と一致する。
- `pop.arrow`から再計算した $J_{\mathrm{cum}}$ が`metrics.csv`、summary、manifestと許容誤差`1e-12`以内で一致する。
- 候補と比較条件のpairedキーに欠損や重複がない。
- 最終テストseed `50001`から`50005`が使われていない。

1件でも不一致があれば、候補選択を行わず分析を失敗として停止する。

## 11. 正式分析の出力

正式分析は`<experiment_root>/candidate_validation_analysis_v01/`へ次を出力する。

| 出力 | 内容 |
| --- | --- |
| `analysis_manifest.json` | 分析コード、入力、ハッシュ、成否 |
| `analysis_summary.json` | run数、候補数、選択数、判定概要 |
| `decision.json` | ネットワーク別の適格数、選択候補、判定 |
| `tables/data_audit.csv` | runごとの品質検査結果 |
| `tables/run_inventory.csv` | 189 runの一覧 |
| `tables/iteration_metrics.parquet` | 18,900反復の共通指標 |
| `tables/condition_seed_summary.csv` | 条件・seed別の平均指標 |
| `tables/condition_summary.csv` | 条件別の平均と区間推定 |
| `tables/candidate_block_performance.csv` | 候補のseedブロック別順位と比較効果 |
| `tables/candidate_effects_vs_none.csv` | 無介入に対する候補効果 |
| `tables/candidate_effects_vs_legacy_balance.csv` | 春学期代表条件に対する追加改善 |
| `tables/candidate_effects_vs_prior_high.csv` | 先行研究パッケージとの外部比較 |
| `tables/exploration_validation_comparison.csv` | 探索時最良値と検証値の差 |
| `tables/candidate_clusters.csv` | 候補領域の構成 |
| `tables/candidate_ranking.csv` | 適格判定と固定順位 |
| `tables/selected_candidates.csv` | 第8段階へ送る候補 |

## 12. 可視化

[第7段階_検証seedによる候補選択.ipynb](../../notebooks/第7段階_検証seedによる候補選択.ipynb)は、最新の`candidate_validation_analysis_v01`を自動的に読み込み、次を表示・保存する。

1. seedブロック別候補順位。
2. 探索時目的値と検証時目的値の比較。
3. 候補別の無介入に対する相対抑制率と95%信頼区間。
4. 確実性・有効性空間における候補領域と選択結果。

正式分析後にノートブックを全セル実行し、日本語フォント、ラベル、注記位置を含めて4図を視覚確認した。

- [図1：seedブロック別候補順位](../../experiments/summer_2026/stage7_candidate_validation/20260824_101150_candidate_validation_v01/candidate_validation_analysis_v01/figures/01_seedブロック別候補順位.png)
- [図2：探索値と検証値の比較](../../experiments/summer_2026/stage7_candidate_validation/20260824_101150_candidate_validation_v01/candidate_validation_analysis_v01/figures/02_探索値と検証値の比較.png)
- [図3：候補別相対抑制率](../../experiments/summer_2026/stage7_candidate_validation/20260824_101150_candidate_validation_v01/candidate_validation_analysis_v01/figures/03_候補別相対抑制率.png)
- [図4：候補領域と選択結果](../../experiments/summer_2026/stage7_candidate_validation/20260824_101150_candidate_validation_v01/candidate_validation_analysis_v01/figures/04_候補領域と選択結果.png)

## 13. 実装時の検証結果

実装後に次を確認した。

- Stage 7専用テスト10件：成功。
- リポジトリ全体の単体テスト96件：成功。
- dry-run：189 run、各ネットワーク63 run、各検証seed63 runを確認。
- 候補run：162、各比較条件：9 runを確認。
- 54候補が第6段階の正式候補一覧と一致することを確認。
- protocol、候補CSV、ノートブックJSON、全Pythonセルの構文を確認。
- 54候補を使った合成データで、集計、順位、領域形成、候補選択、decision出力までの分析契約を確認。

既存テストが意図的に発生させるシミュレーション失敗のログは表示されるが、テスト全体の最終結果は`OK`である。

## 14. 実行資源の概算

第4段階の実測時間中央値から単純換算した逐次実行時間の目安は次のとおりである。

| ネットワーク | 63 runの概算 |
| --- | ---: |
| BA1000 | 約6分 |
| Facebook | 約39分 |
| Wiki-vote | 約23分 |
| 合計 | 約68分 |

環境負荷やI/Oにより変動する。`pop.arrow`単体の概算は約9.5 MiBで、manifest、CSV等を含む実験ディレクトリ全体はこれより大きくなる。

## 15. 実行手順

### 15.1 MacBookでcommit・pushする

この記録用MDとノートブックはGit管理除外対象のため、`git add -f`を使用する。

```bash
cd /Users/sota/projects/cap-sn

git status --short
git diff --check

git add \
  analysis/candidate_validation_analysis.py \
  analyze_stage7_candidate_validation.py \
  experiment_protocols/stage7_candidate_pool_v1.csv \
  experiment_protocols/stage7_candidate_validation_v1.json \
  run_stage7_candidate_validation.py \
  scripts/README.md \
  tests/test_stage7_candidate_validation.py

git add -f \
  'notes/notes_summer/第7段階：検証seedによる候補選択.md' \
  'notebooks/第7段階_検証seedによる候補選択.ipynb'

git diff --cached --name-status
.venv/bin/python -m unittest discover -s tests -q

git commit -m "Add Stage 7 candidate validation pipeline"
git push origin main

git log -1 --oneline
git status --short
```

### 15.2 共用PCを同期して事前確認する

```bash
cd ~/cap-sn

git pull --ff-only origin main
git rev-parse HEAD
git branch --show-current
git status --short

cargo build --release --locked -p v2
.venv/bin/python -m unittest discover -s tests -q

git status --short
```

最後の`git status --short`に何も表示されないことを確認する。

### 15.3 dry-runする

```bash
cd ~/cap-sn

STAGE7_ID="$(date +%Y%m%d_%H%M%S)_candidate_validation_v01"
printf '%s\n' "$STAGE7_ID" > "$HOME/cap_sn_stage7_id.txt"
DRY_RUN="$HOME/${STAGE7_ID}_dry_run.json"

.venv/bin/python run_stage7_candidate_validation.py \
  --experiment-id="$STAGE7_ID" \
  --dry-run > "$DRY_RUN"

.venv/bin/python - "$DRY_RUN" <<'PY'
import json
import sys
from collections import Counter

data = json.load(open(sys.argv[1]))
commands = data["commands"]
networks = Counter(c[c.index("--network") + 1] for c in commands)
seeds = Counter(int(c[c.index("--simulator-seed") + 1]) for c in commands)
conditions = Counter(c[c.index("--condition-id") + 1] for c in commands)

print("experiment_id =", data["experiment_id"])
print("full_run_count =", data["full_run_count"])
print("selected_run_count =", data["selected_run_count"])
print("networks =", dict(networks))
print("seeds =", dict(seeds))

assert data["full_run_count"] == 189
assert data["selected_run_count"] == 189
assert networks == Counter({
    "ba1000": 63,
    "facebook": 63,
    "wiki_vote": 63,
})
assert seeds == Counter({40001: 63, 40002: 63, 40003: 63})
assert conditions["none"] == 9
assert conditions["legacy_balance"] == 9
assert conditions["prior_high"] == 9
print("Stage 7 dry-run check: OK")
PY
```

### 15.4 本実行する

```bash
cd ~/cap-sn

STAGE7_ID="$(cat "$HOME/cap_sn_stage7_id.txt")"
LOG_FILE="$HOME/${STAGE7_ID}.log"

.venv/bin/python -u run_stage7_candidate_validation.py \
  --experiment-id="$STAGE7_ID" \
  2>&1 | tee "$LOG_FILE"
```

中断後に同じGit commit、protocol、候補CSVで再開する場合は次を使う。

```bash
cd ~/cap-sn

STAGE7_ID="$(cat "$HOME/cap_sn_stage7_id.txt")"
LOG_FILE="$HOME/${STAGE7_ID}.log"

.venv/bin/python -u run_stage7_candidate_validation.py \
  --experiment-id="$STAGE7_ID" \
  --resume \
  2>&1 | tee -a "$LOG_FILE"
```

### 15.5 完了数を確認する

```bash
cd ~/cap-sn

STAGE7_ID="$(cat "$HOME/cap_sn_stage7_id.txt")"
PLAN="experiments/summer_2026/stage7_candidate_validation/${STAGE7_ID}/candidate_validation_execution_plan.json"

.venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("experiment_id =",d["experiment_id"]); print("stage =",d["stage"]); print("counts =",d["counts"]); assert d["stage"]=="stage7_candidate_validation"; assert d["counts"]=={"total":189,"completed":189,"pending":0,"other":0}; print("Stage 7 execution check: OK")' "$PLAN"
```

### 15.6 正式分析を実行する

```bash
cd ~/cap-sn

STAGE7_ID="$(cat "$HOME/cap_sn_stage7_id.txt")"
EXPERIMENT_ROOT="experiments/summer_2026/stage7_candidate_validation/${STAGE7_ID}"

.venv/bin/python analyze_stage7_candidate_validation.py \
  --experiment-root "$EXPERIMENT_ROOT" \
  --analysis-id candidate_validation_analysis_v01
```

分析後は`decision.json`と`tables/selected_candidates.csv`を確認し、第8段階へ送る候補を確定する。ただし、検証値を最終性能として解釈しない。

## 16. 正式実験と品質監査

### 16.1 実行情報

| 項目 | 内容 |
| --- | --- |
| 実験ID | `20260824_101150_candidate_validation_v01` |
| 実行Git commit | `a4443b989b349d289acbfe7efe9fe1a84f87254d` |
| 実行環境 | 共用PC |
| run数 | 189 |
| 総反復数 | 18,900 |
| ローカル展開後の容量 | 約18 MiB |
| 正式分析ID | `candidate_validation_analysis_v01` |

正式結果は次から参照できる。

- [analysis_summary.json](../../experiments/summer_2026/stage7_candidate_validation/20260824_101150_candidate_validation_v01/candidate_validation_analysis_v01/analysis_summary.json)
- [decision.json](../../experiments/summer_2026/stage7_candidate_validation/20260824_101150_candidate_validation_v01/candidate_validation_analysis_v01/decision.json)
- [selected_candidates.csv](../../experiments/summer_2026/stage7_candidate_validation/20260824_101150_candidate_validation_v01/candidate_validation_analysis_v01/tables/selected_candidates.csv)
- [candidate_ranking.csv](../../experiments/summer_2026/stage7_candidate_validation/20260824_101150_candidate_validation_v01/candidate_validation_analysis_v01/tables/candidate_ranking.csv)

### 16.2 品質監査結果

| 検査項目 | 結果 |
| --- | ---: |
| 完了run | 189 / 189 |
| 有効run | 189 / 189 |
| iteration metric | 18,900 |
| 候補数 | 54 |
| 候補・seedブロック | 162 |
| 適格候補 | 17 |
| 選択候補 | 9 |
| `pop.arrow` | 189 |
| `info.arrow` | 0 |
| `agent.arrow` | 0 |
| 最終テストseed使用 | なし |

全runについて、`pop.arrow`から再計算した指標と保存済み指標が許容誤差`1e-12`以内で一致した。品質条件をすべて満たしたため、候補選択を正式結果として採用する。

## 17. 比較条件の結果

表のCIは3 seedブロックと各ブロック内100反復を考慮した95%信頼区間である。これらは検証データ上の値であり、最終性能ではない。

| ネットワーク | 条件 | 平均 $J_{\mathrm{cum}}$ | 95% CI | 平均peak new selfish ratio |
| --- | --- | ---: | ---: | ---: |
| BA1000 | `none` | 0.281923 | [0.274437, 0.289377] | 0.098357 |
| BA1000 | `legacy_balance` | 0.277103 | [0.269847, 0.284767] | 0.094547 |
| BA1000 | `prior_high` | 0.152897 | [0.148707, 0.156787] | 0.062603 |
| Facebook | `none` | 0.233221 | [0.227592, 0.238866] | 0.059336 |
| Facebook | `legacy_balance` | 0.251931 | [0.246321, 0.258182] | 0.070239 |
| Facebook | `prior_high` | 0.191581 | [0.189302, 0.193818] | 0.063405 |
| Wiki-vote | `none` | 0.097512 | [0.094285, 0.100872] | 0.031670 |
| Wiki-vote | `legacy_balance` | 0.092744 | [0.089179, 0.096284] | 0.030700 |
| Wiki-vote | `prior_high` | 0.090688 | [0.089641, 0.091660] | 0.026105 |

無介入に対する`legacy_balance`の相対抑制率は、BA1000で$1.71\%$、Wiki-voteで$4.89\%$であった。一方、Facebookでは$-8.02\%$であり、春学期代表条件が無介入より利己的行動を増加させた。したがって、Facebookで候補が`legacy_balance`を上回ることだけでは、利己的行動を抑制したとは言えない。

`prior_high`の無介入に対する相対抑制率は、BA1000で$45.77\%$、Facebookで$17.85\%$、Wiki-voteで$7.00\%$であった。ただし、`prior_high`は確実性・有効性以外も含む先行研究の意見CSV全体であり、2設計変数だけを変えた候補との単純な優劣を設計変数の効果として解釈しない。

## 18. 適格判定と選択候補

### 18.1 ネットワーク別の適格判定

| ネットワーク | 無介入基準を通過 | `legacy_balance`基準を通過 | 両方を通過 | 適格領域 | 選択方法 |
| --- | ---: | ---: | ---: | ---: | --- |
| BA1000 | 18 / 18 | 12 / 18 | 12 / 18 | 8 | 適格領域から3候補 |
| Facebook | 0 / 18 | 16 / 18 | 0 / 18 | 0 | 探索的fallbackを3候補 |
| Wiki-vote | 18 / 18 | 5 / 18 | 5 / 18 | 4 | 適格領域から3候補 |

BA1000とWiki-voteでは全18候補が無介入より良い方向を示したが、代表条件に対する追加改善まで満たす候補は一部であった。Facebookでは16候補が代表条件を上回ったものの、無介入基準を通過した候補は1件もなかった。このため、Facebookの3候補は適格候補ではなく、第8段階で独立確認するための探索的fallbackである。

### 18.2 第8段階へ送る9候補

各相対抑制率は「比較条件から候補を引いた値」であり、正なら候補の方が良い。角括弧内は95%階層paired bootstrap CIである。順位欄は「3ブロック順位の中央値 / 最悪順位」を示す。

| ネットワーク | 判定 | 選択順 | 候補 | 確実性 | 有効性 | 順位 | $\eta$ vs `none` | $\eta$ vs `legacy` | $\eta$ vs `prior_high` |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BA1000 | 適格 | 1 | `cand_cma_es_r03` | 0.7947 | 0.8157 | 4 / 13 | 2.97% [-0.61, 6.60] | 1.28% [-1.27, 3.68] | -78.91% [-84.01, -74.10] |
| BA1000 | 適格 | 2 | `cand_bo_gp_r04` | 0.6969 | 0.9928 | 5 / 18 | 2.50% [-3.73, 7.63] | 0.80% [-3.23, 4.51] | -79.78% [-87.33, -72.28] |
| BA1000 | 適格 | 3 | `cand_bo_gp_r02` | 0.5034 | 0.9927 | 6 / 15 | 2.65% [-3.20, 7.47] | 0.95% [-2.90, 4.46] | -79.51% [-86.54, -72.58] |
| Facebook | fallback | 1 | `cand_bo_gp_r06` | 0.7109 | 0.8575 | 2 / 3 | -6.25% [-9.99, -2.86] | 1.64% [-0.49, 3.66] | -29.34% [-32.50, -26.21] |
| Facebook | fallback | 2 | `cand_bo_gp_r01` | 0.6614 | 0.8815 | 2 / 8 | -6.18% [-10.14, -2.51] | 1.71% [-0.34, 3.68] | -29.25% [-32.60, -26.03] |
| Facebook | fallback | 3 | `cand_bo_gp_r03` | 0.6211 | 0.5428 | 5 / 18 | -7.82% [-11.27, -4.57] | 0.19% [-2.61, 2.76] | -31.25% [-34.18, -28.33] |
| Wiki-vote | 適格 | 1 | `cand_cma_es_r04` | 0.7442 | 0.7482 | 4 / 12 | 6.53% [3.04, 9.80] | 1.73% [-1.50, 4.82] | -0.50% [-3.57, 2.34] |
| Wiki-vote | 適格 | 2 | `cand_random_search_r04` | 0.7764 | 0.5068 | 8 / 17 | 5.23% [0.63, 9.37] | 0.36% [-4.23, 4.78] | -1.90% [-6.37, 1.97] |
| Wiki-vote | 適格 | 3 | `cand_random_search_r03` | 0.8448 | 0.5016 | 9 / 11 | 6.06% [2.27, 9.64] | 1.23% [-2.59, 5.00] | -1.01% [-4.90, 2.79] |

BA1000の選択候補は無介入に対して$2.50\%$から$2.97\%$の抑制方向を示したが、3候補とも95% CIは0を含んだ。Wiki-voteの3候補は無介入に対して$5.23\%$から$6.53\%$の抑制方向を示し、95% CIの下限も0を上回った。Facebookの3 fallback候補は、無介入に対して$6.18\%$から$7.82\%$の悪化を示し、CIも全て負であった。

`legacy_balance`に対する追加改善は9候補すべてで小さく、全候補のCIが0を含んだ。第8段階では、行動誘導情報そのものの効果と、最適化による代表条件からの追加改善を分けて評価する必要がある。

## 19. 探索時評価と検証時評価の差

| ネットワーク | 検証時に悪化した候補 | 平均差「検証－探索」 | 中央値差 | 探索値と検証値のSpearman $\rho$ |
| --- | ---: | ---: | ---: | ---: |
| BA1000 | 18 / 18 | 0.007917 | 0.008578 | -0.143 |
| Facebook | 18 / 18 | 0.006599 | 0.007156 | -0.585 |
| Wiki-vote | 18 / 18 | 0.005378 | 0.005499 | 0.602 |

54候補すべてで検証時 $J_{\mathrm{cum}}$ が探索時最良値より高くなった。第6段階の最良値には、探索seedへの適合と、多数の評価点から最小値を選んだことによる楽観性が含まれていたと考えられる。

探索時の候補順序が検証時にも維持された程度はネットワークで異なり、Wiki-voteでは中程度の正の関係が見られた一方、BA1000ではほぼ関係がなく、Facebookでは逆方向であった。したがって、探索時最良値だけで最終候補を決めず、独立した検証seedで再順位付けした判断は必要であった。

## 20. 結果の解釈

### 20.1 単一の万能パラメータは得られていない

BA1000では、有効性が約0.99の2領域に加え、確実性0.7947、有効性0.8157の領域も選ばれた。Wiki-voteでは、有効性約0.50の2領域と、有効性0.7482の領域が選ばれた。第7段階の結果は、全ネットワーク共通で有効性を最大化すればよいという単純な方針を支持していない。

ただし、これは54候補の検証結果であり、連続空間全体の最適領域を証明したものではない。候補領域は観測済み点の整理に限って解釈する。

### 20.2 Facebookでは最適化成功を主張しない

Facebookでは、候補の多くが`legacy_balance`より良かったが、`legacy_balance`自体が無介入より悪かった。18候補すべてが無介入基準を通過しなかったため、第6段階の探索範囲と候補集合から「利己的行動を抑制する有効候補が得られた」とは結論しない。

一方、外部ベンチマークの`prior_high`は無介入より良かった。したがって、Facebookで行動誘導情報が常に逆効果という結論にもならない。`prior_high`と生成候補では確実性・有効性以外の意見成分も異なるため、この差は第9段階以降のモデル解釈上の課題として分けて扱う。

### 20.3 手法優位性は判断しない

選択候補の由来は、BA1000がCMA-ES 1件とGPR 2件、Wiki-voteがCMA-ES 1件とランダムサーチ2件、Facebook fallbackがGPR 3件であった。しかし、候補は手法比較用に独立再現した標本ではなく、各手法が生成した点の出自にすぎない。この構成から最適化手法の優劣は主張しない。

### 20.4 第7段階で主張できる範囲

第7段階で正式に言えるのは、事前規則に従ってBA1000とWiki-voteの適格候補、Facebookの探索的fallbackを選び、第8段階へ送る候補を固定したことである。表18.2の効果量は選択にも使用したため、最終性能や一般化性能として報告しない。

## 21. 完了判定と次段階

- [x] 第6段階の正式候補54件を固定した。
- [x] 比較条件と検証seedを固定した。
- [x] 適格条件、順位規則、候補領域、fallback規則を事前固定した。
- [x] MacBook側でcommit・pushした。
- [x] 共用PCで189 runを完了した。
- [x] 189 runと18,900反復の品質監査に合格した。
- [x] 正式分析を実行した。
- [x] 第8段階へ送る9候補を固定した。
- [x] 可視化ノートブックを全セル実行した。
- [x] 最終テストseed `50001`から`50005`が未使用であることを確認した。

以上より、第7段階を**完了**とする。次の第8段階では、ここで固定した9候補と比較条件だけを、未使用seed `50001`から`50005`で評価する。第8段階の結果を見て候補を追加・削除・入れ替えない。
