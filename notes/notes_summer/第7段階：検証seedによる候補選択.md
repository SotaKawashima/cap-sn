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

分析結果がまだ存在しないため、ノートブックは実装と構文確認まで行い、未実行で保存している。正式分析後に上から順に実行する。

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

## 16. 現在の進捗

- [x] 第6段階の正式候補54件を固定した。
- [x] 比較条件と検証seedを固定した。
- [x] 適格条件、順位規則、候補領域、fallback規則を固定した。
- [x] 実行・分析・テスト・可視化コードを実装した。
- [x] 単体テストとdry-runで実装を検証した。
- [ ] MacBook側でcommit・pushする。
- [ ] 共用PCで189 runを完了する。
- [ ] 正式分析を実行する。
- [ ] 第8段階へ送る候補または候補領域を固定する。
- [ ] 最終テストseedが未使用であることを最終確認する。

現時点では、第7段階の**実験前準備と実装が完了**している。第7段階そのものの完了は、189 runの品質監査と正式候補選択が終了した時点とする。
