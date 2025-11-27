import subprocess
import sys
import os
import toml
import pandas as pd
import optuna
import shutil
from optuna.samplers import TPESampler

# ==========================================
# 1. パス・環境設定
# ==========================================

# Rust実行ファイル
RUST_BIN = "./target/release/v2.exe" 
# RUST_BIN = "./target/release/v2" # Mac/Linux

# ベースディレクトリ
BASE_TEST_DIR = "./v2/test"

# 各種設定ファイル
RUNTIME_CONF = f"{BASE_TEST_DIR}/runtime.toml"
NETWORK_CONF = f"{BASE_TEST_DIR}/network_config.toml"
AGENT_CONF   = f"{BASE_TEST_DIR}/agent_config.toml"
STRATEGY_TEMPLATE = f"{BASE_TEST_DIR}/strategy_config.toml"

# 出力先
TEMP_DIR = "./temp_test_experiment"
RESULT_DIR = f"{TEMP_DIR}/result"
CSV_DIR = f"{TEMP_DIR}/csv"

# CSVヘッダー
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

def to_absolute_path(rel_path, base_dir):
    """相対パスを絶対パスに変換"""
    abs_base = os.path.abspath(base_dir)
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    joined = os.path.join(abs_base, rel_path)
    return os.path.abspath(joined).replace("\\", "/")

def create_inhibition_csv(x1, x2, trial_id):
    """x1, x2 をCSVに保存 (小数点以下4桁に丸める)"""
    
    # --- ★変更点1: 数値を丸める ---
    x1 = round(x1, 4)
    x2 = round(x2, 4)
    
    # --- パラメータ計算 ---
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

    # CSV書き込み用リスト
    # 計算された u も念のため丸めておく
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
    
    df.to_csv(csv_abs_path, index=False, header=True)
    return csv_abs_path

def create_strategy_toml(csv_abs_path, trial_id):
    """Strategy TOMLを生成 (パスを絶対パス化)"""
    with open(STRATEGY_TEMPLATE, "r") as f:
        config = toml.load(f)
    
    if "informing" in config:
        config["informing"] = to_absolute_path(config["informing"], BASE_TEST_DIR)
        
    if "information" in config:
        info_sec = config["information"]
        for key in info_sec:
            if key == "inhibition":
                info_sec[key] = csv_abs_path
            else:
                info_sec[key] = to_absolute_path(info_sec[key], BASE_TEST_DIR)

    new_toml_path = f"{TEMP_DIR}/strategy_{trial_id}.toml"
    with open(new_toml_path, "w") as f:
        toml.dump(config, f)
        
    return os.path.abspath(new_toml_path)

def calc_score(trial_id):
    """Arrowファイルからスコア計算"""
    arrow_path = f"{RESULT_DIR}/trial_{trial_id}_pop.arrow"
    
    if not os.path.exists(arrow_path):
        return 1.0

    try:
        df = pd.read_feather(arrow_path)
        last_row = df.iloc[-1] # 最終ステップを取得
        
        # 【修正】正しいカラム名 "num_selfish" を使用
        num_selfish = last_row["num_selfish"]
        
        # 【重要】PopStatには総数(total)が含まれていないため、
        # シミュレーション設定に合わせて固定値(1000)または設定ファイルから読み取る必要があります。
        total = 100
        
        if total == 0: return 1.0
        
        return num_selfish / total
        
    except KeyError as e:
        print(f"  [Error] Column not found: {e}")
        return 1.0
    except Exception as e:
        print(f"  [Error] Arrow read failed: {e}")
        return 1.0
    
def cleanup_trial_files(trial_id):
    """試行ごとの一時ファイルを削除"""
    identifier = f"trial_{trial_id}"
    
    # 削除対象: CSV, TOML, および全てのArrowファイル
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

    cmd = [
        RUST_BIN,
        identifier,
        abs_result_dir,
        "--runtime", RUNTIME_CONF,
        "--network", NETWORK_CONF,
        "--agent", AGENT_CONF,
        "--strategy", strategy_path,
        "--enable_inhibition",
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
        # 都度削除
        cleanup_trial_files(trial_id)

# ==========================================
# 4. メイン実行
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(RUST_BIN):
        print(f"Error: Rust binary not found.")
        sys.exit(1)
        
    setup_environment()
    
    print("=== Start Optimization ===")
    
    study = optuna.create_study(
        study_name="test_env_opt",
        direction="minimize",
        sampler=TPESampler(seed=42),
        storage="sqlite:///test_env.db",
        load_if_exists=True
    )
    
    # 試行回数 (テスト用に5回)
    study.optimize(objective, n_trials=5)
    
    print("\n=== Optimization Finished ===")
    print(f"Best Params: {study.best_params}")
    print(f"Best Score : {study.best_value}")

    # --- ★変更点2: 最良パラメータでの再実行と結果保存 ---
    print("\n=== Re-running Best Configuration ===")
    
    # 最良パラメータの取得
    best_x1 = study.best_params["certainty"]
    best_x2 = study.best_params["effectiveness"]
    
    # ファイル生成 (識別子を 'best' にする)
    best_csv = create_inhibition_csv(best_x1, best_x2, "best")
    best_toml = create_strategy_toml(best_csv, "best")
    
    print(f"Saved Best CSV : {best_csv}")
    print(f"Saved Best TOML: {best_toml}")
    
# Rust再実行
    abs_result_dir = os.path.abspath(RESULT_DIR)
    cmd = [
        RUST_BIN,
        "trial_best",
        abs_result_dir,
        "--runtime", RUNTIME_CONF,
        "--network", NETWORK_CONF,
        "--agent", AGENT_CONF,
        "--strategy", best_toml,
        "-e", # <--- 【修正】 "--enable_inhibition" から変更
        "-d", "0"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Best simulation run completed.")
        print(f"Arrow Files Saved in: {RESULT_DIR}")
        print(f" - trial_best_pop.arrow")
        print(f" - trial_best_info.arrow")
        print(f" - trial_best_agent.arrow")
        
    except subprocess.CalledProcessError as e:
        print(f"Error re-running best simulation: {e}")