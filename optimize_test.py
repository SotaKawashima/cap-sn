import subprocess
import sys
import toml
import pandas as pd
import optuna
import os
from pathlib import Path
from optuna.integration import BoTorchSampler
from tqdm import tqdm  # 【追加】進捗バー表示用

# ==========================================
# 1. 設定 & 定数
# ==========================================
ENV_ROOT = Path("./v2/test_2")
TEMP_DIR = Path("./temp_test_2_experiment")
RESULT_DIR = TEMP_DIR / "result"
CSV_DIR = TEMP_DIR / "csv"

#RUST_BIN = Path("./target/release/v2.exe")
RUST_BIN = Path("./target/release/v2.exe" if os.name == "nt" else "./target/release/v2")
RUNTIME_CONF = ENV_ROOT / "runtime.toml"
NETWORK_CONF = ENV_ROOT / "network/network-ba1000.toml"
AGENT_CONF   = ENV_ROOT / "agent/agent-type6.toml" # ファイル実在確認必須
STRATEGY_TEMPLATE = ENV_ROOT / "strategy/strategy-config.toml"

TOTAL_AGENTS = 1000
N_STARTUP_TRIALS = 10
N_TRIALS = 100
DB_NAME = "test_2_opt_gpr.db"
DB_URL = f"sqlite:///{DB_NAME}"

# ==========================================
# 2. ヘルパー関数
# ==========================================

def setup_directories():
    """必要なディレクトリを一括作成"""
    for d in [TEMP_DIR, RESULT_DIR, CSV_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def resolve_path_in_toml(rel_path: str, base_file: Path) -> str:
    """TOML内の相対パスを絶対パス文字列に変換"""
    return str((base_file.parent / rel_path).resolve()).replace("\\", "/")

def create_inhibition_csv(x1: float, x2: float, identifier: str) -> Path:
    """パラメータをCSVに保存"""
    x1, x2 = round(x1, 4), round(x2, 4)
    u1, u2 = round(1.0 - x1, 4), round(1.0 - x2, 4)

    data = {
        "phi_b0": 0.0, "phi_b1": x1, "phi_u": u1, "phi_a0": 0.5, "phi_a1": 0.5,
        "psi0_b0": 0.5, "psi0_b1": 0.0, "psi0_u": 0.5,
        "psi1_b0": x2, "psi1_b1": 0.0, "psi1_u": u2,
        "b0_b0": 0.5, "b0_b1": 0.0, "b0_u": 0.5,
        "b1_b0": x2, "b1_b1": 0.0, "b1_u": u2
    }

    df = pd.DataFrame([data])
    file_path = CSV_DIR / f"inhibition_{identifier}.csv"
    df.to_csv(file_path, index=False)
    return file_path.resolve()

def create_strategy_toml(csv_path: Path, identifier: str) -> Path:
    """テンプレートを読み込みCSVパスを差し替え"""
    config = toml.load(STRATEGY_TEMPLATE)
    csv_abs_path = str(csv_path).replace("\\", "/")

    if "informing" in config:
        config["informing"] = resolve_path_in_toml(config["informing"], STRATEGY_TEMPLATE)
    
    if "information" in config:
        for key, val in config["information"].items():
            if key == "inhibition":
                config["information"][key] = csv_abs_path
            else:
                config["information"][key] = resolve_path_in_toml(val, STRATEGY_TEMPLATE)

    output_path = TEMP_DIR / f"strategy_{identifier}.toml"
    with open(output_path, "w") as f:
        toml.dump(config, f)
    
    return output_path.resolve()

def run_simulation(identifier: str, strategy_path: Path, show_output=False):
    """Rustシミュレータ実行"""
    cmd = [
        str(RUST_BIN),
        identifier,
        str(RESULT_DIR.resolve()),
        "--runtime", str(RUNTIME_CONF),
        "--network", str(NETWORK_CONF),
        "--agent", str(AGENT_CONF),
        "--strategy", str(strategy_path),
        "-e", "-d", "0"
    ]
    
    if show_output:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def calc_score(identifier: str) -> float:
    """スコア計算 (最終ステップの平均割合)"""
    arrow_path = RESULT_DIR / f"{identifier}_pop.arrow"
    if not arrow_path.exists():
        return 1.0

    try:
        df = pd.read_feather(arrow_path)
        if "num_selfish" not in df.columns:
            return 1.0

        if "num_iter" in df.columns:
            avg_selfish = df.groupby("num_iter")["num_selfish"].last().mean()
        else:
            avg_selfish = df["num_selfish"].iloc[-1]
            
        return avg_selfish / TOTAL_AGENTS

    except Exception as e:
        # プログレスバーの邪魔にならないようエラーは標準エラー出力へ
        print(f"  [Error] Score calculation failed: {e}", file=sys.stderr)
        return 1.0

def cleanup_files(identifier: str):
    """指定された identifier を持つ全一時ファイルを削除"""
    files = [
        CSV_DIR / f"inhibition_{identifier}.csv",
        TEMP_DIR / f"strategy_{identifier}.toml",
        RESULT_DIR / f"{identifier}_pop.arrow",
        RESULT_DIR / f"{identifier}_info.arrow",
        RESULT_DIR / f"{identifier}_agent.arrow"
    ]
    for p in files:
        p.unlink(missing_ok=True)

# ==========================================
# 3. 最適化プロセス
# ==========================================
def objective(trial):
    identifier = f"trial_{trial.number}"
    
    # パラメータ探索範囲: 0.5 ～ 1.0
    x1 = trial.suggest_float("certainty", 0.5, 1.0)
    x2 = trial.suggest_float("effectiveness", 0.5, 1.0)
    
    csv_path = create_inhibition_csv(x1, x2, identifier)
    toml_path = create_strategy_toml(csv_path, identifier)
    
    try:
        run_simulation(identifier, toml_path, show_output=False)
        return calc_score(identifier)
        
    except subprocess.CalledProcessError:
        return 1.0
    finally:
        cleanup_files(identifier)

# ==========================================
# 4. メイン実行
# ==========================================
if __name__ == "__main__":
    if not RUST_BIN.exists():
        sys.exit(f"Error: Binary not found at {RUST_BIN}")
    
    # 【追加】エージェント設定ファイルの存在確認
    if not AGENT_CONF.exists():
        sys.exit(f"Error: Agent config not found at {AGENT_CONF}")

    setup_directories()

    # DB初期化
    if os.path.exists(DB_NAME):
        try:
            os.remove(DB_NAME)
            print(f"Deleted old database: {DB_NAME}")
        except PermissionError:
            print(f"Warning: Could not delete {DB_NAME}. Is it open?")

    print(f"=== Start Optimization (Agents: {TOTAL_AGENTS}, Method: GPR) ===")
    
    # 【追加】Optunaのデフォルトログを抑制（バーと被るため）
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = BoTorchSampler(n_startup_trials=N_STARTUP_TRIALS, seed=42)
    study = optuna.create_study(
        study_name="test_2_opt_gpr",
        direction="minimize",
        sampler=sampler,
        storage=DB_URL,
        load_if_exists=True
    )

    # 【追加】tqdmを使った進捗バーの設定
    print("Running trials...")
    with tqdm(total=N_TRIALS, desc="Optimization Progress", unit="trial") as pbar:
        
        # Optunaのコールバック機能を使ってバーを更新する関数
        def progress_callback(study, trial):
            pbar.update(1)
            # 現在のベストスコアをバーの横に表示
            best_val = study.best_value if study.best_value is not None else float('inf')
            pbar.set_postfix({"Best Score": f"{best_val:.5f}"})

        # 最適化実行
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[progress_callback])

    # 完了後の表示
    print("\n=== Optimization Finished ===")
    print(f"Best Params: {study.best_params}")
    print(f"Best Score : {study.best_value}")

    # --- 最良結果の再現 ---
    print("\n=== Re-running Best Configuration ===")
    best_id = "trial_best"
    
    cleanup_files(best_id)
    
    best_csv = create_inhibition_csv(study.best_params["certainty"], study.best_params["effectiveness"], best_id)
    best_toml = create_strategy_toml(best_csv, best_id)
    
    try:
        run_simulation(best_id, best_toml, show_output=True)
        print(f"\n[Success] Results saved to: {RESULT_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Best simulation failed with exit code {e.returncode}.")