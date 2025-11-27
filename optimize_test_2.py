import subprocess
import sys
import os
import toml
import pandas as pd
import optuna
import shutil
from optuna.samplers import TPESampler

# ==========================================
# 1. パス・環境設定 (test_2 用)
# ==========================================

# Rust実行ファイル
RUST_BIN = "./target/release/v2.exe" 
# RUST_BIN = "./target/release/v2" # Mac/Linux

# 環境のルートディレクトリ
ENV_ROOT_DIR = "./v2/test_2"

# 各種設定ファイル (CLIで指定するもの)
RUNTIME_CONF = f"{ENV_ROOT_DIR}/runtime.toml"
NETWORK_CONF = f"{ENV_ROOT_DIR}/network/network-ba100.toml"
AGENT_CONF   = f"{ENV_ROOT_DIR}/agent/agent-type2.toml"
STRATEGY_TEMPLATE = f"{ENV_ROOT_DIR}/strategy/strategy-config.toml"

# 実験データの一時保存先
TEMP_DIR = "./temp_test_2_experiment"
RESULT_DIR = f"{TEMP_DIR}/result"
CSV_DIR = f"{TEMP_DIR}/csv"

# エージェント総数 (ba100 なので 100)
TOTAL_AGENTS = 100

# CSVヘッダー (17列)
CSV_HEADER = [
    "phi_b0", "phi_b1", "phi_u", "phi_a0", "phi_a1",
    "psi0_b0", "psi0_b1", "psi0_u",
    "psi1_b0", "psi1_b1", "psi1_u",
    "b0_b0", "b0_b1", "b0_u",
    "b1_b0", "b1_b1", "b1_u"
]

# ==========================================
# 2. ヘルパー関数
# ==========================================

def setup_environment():
    for d in [TEMP_DIR, RESULT_DIR, CSV_DIR]:
        os.makedirs(d, exist_ok=True)

def resolve_path(rel_path, base_file_path):
    """
    設定ファイル内の相対パスを絶対パスに変換する
    base_file_path: 相対パスが記述されているファイルのパス (例: strategy-config.toml)
    """
    # 記述されているファイルのディレクトリを基準にする
    base_dir = os.path.dirname(os.path.abspath(base_file_path))
    
    # パス結合して正規化 (../ などを解決)
    joined = os.path.normpath(os.path.join(base_dir, rel_path))
    return joined.replace("\\", "/") # Windowsパス対策

def create_inhibition_csv(x1, x2, trial_id):
    """x1, x2 をCSVに保存 (小数点以下4桁に丸める)"""
    x1 = round(x1, 4)
    x2 = round(x2, 4)
    
    # --- パラメータ計算 (前回定義通り) ---
    # 1. phi (対策実施)
    phi_b0, phi_b1 = 0.0, x1
    phi_u = 1.0 - x1
    phi_a0, phi_a1 = 0.5, 0.5

    # 2. psi0 (原因なし & 対策あり)
    psi0_b0, psi0_b1 = 0.5, 0.0
    psi0_u = 0.5

    # 3. psi1 (原因あり & 対策あり)
    psi1_b0, psi1_b1 = x2, 0.0
    psi1_u = 1.0 - x2

    # 4. b0 (観察なし & 対策あり)
    b0_b0, b0_b1 = 0.5, 0.0
    b0_u = 0.5

    # 5. b1 (観察あり & 対策あり)
    b1_b0, b1_b1 = x2, 0.0
    b1_u = 1.0 - x2

    row_data = [
        phi_b0, phi_b1, round(phi_u, 4), phi_a0, phi_a1,
        psi0_b0, psi0_b1, psi0_u,
        psi1_b0, psi1_b1, round(psi1_u, 4),
        b0_b0, b0_b1, b0_u,
        b1_b0, b1_b1, round(b1_u, 4)
    ]

    df = pd.DataFrame([row_data], columns=CSV_HEADER)
    
    csv_filename = f"inhibition_{trial_id}.csv"
    csv_abs_path = os.path.abspath(os.path.join(CSV_DIR, csv_filename)).replace("\\", "/")
    
    # ヘッダーありで保存
    df.to_csv(csv_abs_path, index=False, header=True)
    return csv_abs_path

def create_strategy_toml(csv_abs_path, trial_id):
    """Strategy TOMLを生成 (相対パス ../ を適切に処理)"""
    
    # テンプレート読み込み
    with open(STRATEGY_TEMPLATE, "r") as f:
        config = toml.load(f)
    
    # 1. 'informing' のパス解決
    # strategy-config.toml から見た相対パス (../informing/...) を解決
    if "informing" in config:
        config["informing"] = resolve_path(config["informing"], STRATEGY_TEMPLATE)
        
    # 2. '[information]' セクションのパス解決 & 差し替え
    if "information" in config:
        info_sec = config["information"]
        for key in info_sec:
            if key == "inhibition":
                # ここだけ新しいCSVのパスを使う
                info_sec[key] = csv_abs_path
            else:
                # 他(misinfo, correction等)は既存ファイルを指すように絶対パス化
                # 例: "../information_opinion/n95_95.csv"
                info_sec[key] = resolve_path(info_sec[key], STRATEGY_TEMPLATE)

    # 保存
    new_toml_path = f"{TEMP_DIR}/strategy_{trial_id}.toml"
    with open(new_toml_path, "w") as f:
        toml.dump(config, f)
        
    return os.path.abspath(new_toml_path)

def calc_score(trial_id):
    """
    Arrowファイルからスコア計算
    test_2環境は複数回の試行(iteration)を含むため、
    全イテレーションの最終ステップにおける利己的行動者数の「平均」をスコアとする。
    """
    # test_ba100 という識別子で出力される (CLI引数で指定するため)
    identifier = f"trial_{trial_id}"
    arrow_path = f"{RESULT_DIR}/{identifier}_pop.arrow"
    
    if not os.path.exists(arrow_path):
        return 1.0

    try:
        df = pd.read_feather(arrow_path)
        
        # num_selfish カラムの存在確認
        if "num_selfish" not in df.columns:
            print("  [Error] 'num_selfish' column not found.")
            return 1.0

        # num_iter (試行番号) ごとにグループ化
        if "num_iter" in df.columns:
            # 各イテレーションの最終行を取得
            # (tでソートされている前提で、groupごとの最後の要素を取得)
            last_rows = df.groupby("num_iter").last()
            
            # 各イテレーションの最終ステップの利己的行動者数
            final_selfish_counts = last_rows["num_selfish"]
            
            # 平均値を計算
            avg_selfish = final_selfish_counts.mean()
            
            # 割合に変換
            score = avg_selfish / TOTAL_AGENTS
            return score
            
        else:
            # num_iterがない場合（単回実行）
            last_row = df.iloc[-1]
            num_selfish = last_row["num_selfish"]
            return num_selfish / TOTAL_AGENTS
        
    except Exception as e:
        print(f"  [Error] Arrow read failed: {e}")
        return 1.0

def cleanup_trial_files(trial_id):
    """試行ごとの一時ファイルを削除"""
    identifier = f"trial_{trial_id}"
    targets = [
        f"{CSV_DIR}/inhibition_{trial_id}.csv",
        f"{TEMP_DIR}/strategy_{trial_id}.toml",
        f"{RESULT_DIR}/{identifier}_pop.arrow",
        f"{RESULT_DIR}/{identifier}_info.arrow",
        f"{RESULT_DIR}/{identifier}_agent.arrow"
    ]
    for path in targets:
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

# ==========================================
# 3. 目的関数
# ==========================================
def objective(trial):
    trial_id = trial.number
    
    # パラメータ決定
    x1 = trial.suggest_float("certainty", 0.0, 1.0)
    x2 = trial.suggest_float("effectiveness", 0.0, 1.0)
    
    # ファイル生成
    csv_path = create_inhibition_csv(x1, x2, trial_id)
    strategy_path = create_strategy_toml(csv_path, trial_id)
    
    # Rust実行
    identifier = f"trial_{trial_id}"
    abs_result_dir = os.path.abspath(RESULT_DIR)

    # test_2用のコマンドライン引数
    cmd = [
        RUST_BIN,
        identifier,      # identifier (例: trial_0)
        abs_result_dir,  # output_dir
        "--runtime", RUNTIME_CONF,
        "--network", NETWORK_CONF,
        "--agent", AGENT_CONF,
        "--strategy", strategy_path,
        "-e",            # enable_inhibition
        "-d", "0"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        score = calc_score(trial_id)
        return score

    except subprocess.CalledProcessError:
        print(f"  [Error] Simulation crashed at trial {trial_id}")
        return 1.0
    
    finally:
        cleanup_trial_files(trial_id)

# ==========================================
# 4. メイン実行
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(RUST_BIN):
        print(f"Error: Rust binary not found.")
        sys.exit(1)
        
    setup_environment()
    
    print(f"=== Start Optimization (Test_2 Env: {TOTAL_AGENTS} Agents) ===")
    
    study = optuna.create_study(
        study_name="test_2_opt",
        direction="minimize",
        sampler=TPESampler(seed=42),
        storage="sqlite:///test_2_opt.db",
        load_if_exists=True
    )
    
    # まずは10回で様子見
    study.optimize(objective, n_trials=5)
    
    print("\n=== Optimization Finished ===")
    print(f"Best Params: {study.best_params}")
    print(f"Best Score : {study.best_value}")

    # --- 最良パラメータでの再実行 ---
    print("\n=== Re-running Best Configuration ===")
    
    best_x1 = study.best_params["certainty"]
    best_x2 = study.best_params["effectiveness"]
    
    best_csv = create_inhibition_csv(best_x1, best_x2, "best")
    best_toml = create_strategy_toml(best_csv, "best")
    
    print(f"Saved Best CSV : {best_csv}")
    print(f"Saved Best TOML: {best_toml}")
    
    abs_result_dir = os.path.abspath(RESULT_DIR)
    cmd = [
        RUST_BIN,
        "trial_best",
        abs_result_dir,
        "--runtime", RUNTIME_CONF,
        "--network", NETWORK_CONF,
        "--agent", AGENT_CONF,
        "--strategy", best_toml,
        "-e",
        "-d", "0"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Best simulation run completed.")
        print(f"Arrow Files Saved in: {RESULT_DIR}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error re-running best simulation: {e}")