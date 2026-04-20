import subprocess
import sys
import toml
import pandas as pd
import optuna
import os
import time
import json
import warnings
import argparse
from pathlib import Path
from optuna.integration import BoTorchSampler
from tqdm import tqdm
from optuna.exceptions import ExperimentalWarning
from botorch.exceptions.warnings import OptimizationWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)


# ==========================================
# 1. 引数処理
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run optimization experiment.")
    parser.add_argument(
        "--method",
        type=str,
        default="GPR",
        choices=["GPR", "CMAES", "GA", "RANDOM"],
        help="Optimization method"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="ba_1000",
        choices=["ba_1000", "facebook", "wiki-vote"],
        help="Network setting"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of optimization trials"
    )
    return parser.parse_args()


args = parse_args()


# ==========================================
# 2. 設定 & 定数
# ==========================================

METHOD_NAME = args.method
NETWORK_NAME = args.network
N_TRIALS = args.trials

BASE_DIR = Path(__file__).resolve().parent
ENV_ROOT = BASE_DIR / "v2" / "test_2"

NETWORK_CONF_MAP = {
    "ba_1000": ENV_ROOT / "network" / "network-ba1000.toml",
    "facebook": ENV_ROOT / "network" / "network-facebook.toml",
    "wiki-vote": ENV_ROOT / "network" / "network-wiki-vote.toml",
}

if NETWORK_NAME not in NETWORK_CONF_MAP:
    raise ValueError(f"Unsupported NETWORK_NAME: {NETWORK_NAME}")

# ---- 出力先フォルダ構成 ----
# 例:
# optimize_test_ba_1000/gpr/
# optimize_test_facebook/cmaes/
# optimize_test_wiki-vote/random/
NETWORK_OUTPUT_DIR = BASE_DIR / f"optimize_test_{NETWORK_NAME}"
METHOD_DIR = NETWORK_OUTPUT_DIR / METHOD_NAME.lower()

RESULT_DIR = METHOD_DIR / "result"
CSV_DIR = METHOD_DIR / "csv"
LOG_DIR = METHOD_DIR / "logs"

# study DB も各手法フォルダ内に置く
DB_PATH = METHOD_DIR / "study.db"
DB_URL = f"sqlite:///{DB_PATH.resolve().as_posix()}"

RUST_BIN = BASE_DIR / "target" / "release" / ("v2.exe" if os.name == "nt" else "v2")
RUNTIME_CONF = ENV_ROOT / "runtime.toml"
NETWORK_CONF = NETWORK_CONF_MAP[NETWORK_NAME]
AGENT_CONF = ENV_ROOT / "agent" / "agent-type6.toml"
STRATEGY_TEMPLATE = ENV_ROOT / "strategy" / "strategy-config.toml"

TOTAL_AGENTS = 1000
N_STARTUP_TRIALS = 10


# ==========================================
# 3. ヘルパー関数
# ==========================================

def setup_directories():
    """必要なディレクトリを一括作成"""
    for d in [NETWORK_OUTPUT_DIR, METHOD_DIR, RESULT_DIR, CSV_DIR, LOG_DIR]:
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

    output_path = METHOD_DIR / f"strategy_{identifier}.toml"
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
    """
    各 num_iter の最終時刻 t における num_selfish を取り、
    その平均を TOTAL_AGENTS で割った割合を返す
    """
    arrow_path = RESULT_DIR / f"{identifier}_pop.arrow"
    if not arrow_path.exists():
        return 1.0

    try:
        df = pd.read_feather(arrow_path)

        if "num_selfish" not in df.columns:
            return 1.0

        if "num_iter" in df.columns and "t" in df.columns:
            final_rows = (
                df.sort_values(["num_iter", "t"])
                  .groupby("num_iter", as_index=False)
                  .tail(1)
            )
            avg_selfish = final_rows["num_selfish"].mean()

        elif "num_iter" in df.columns:
            avg_selfish = df.groupby("num_iter")["num_selfish"].last().mean()

        else:
            avg_selfish = df["num_selfish"].iloc[-1]

        return float(avg_selfish) / TOTAL_AGENTS

    except Exception as e:
        print(f"  [Error] Score calculation failed: {e}", file=sys.stderr)
        return 1.0

def cleanup_files(identifier: str):
    """指定された identifier を持つ全一時ファイルを削除"""
    files = [
        CSV_DIR / f"inhibition_{identifier}.csv",
        METHOD_DIR / f"strategy_{identifier}.toml",
        RESULT_DIR / f"{identifier}_pop.arrow",
        RESULT_DIR / f"{identifier}_info.arrow",
        RESULT_DIR / f"{identifier}_agent.arrow"
    ]
    for p in files:
        p.unlink(missing_ok=True)

def save_timing_report(study, total_elapsed_sec: float):
    """trialごとの実行時間ログと要約を保存"""
    rows = []
    for t in study.trials:
        rows.append({
            "trial": t.number,
            "state": str(t.state),
            "value": t.value,
            "certainty": t.params.get("certainty"),
            "effectiveness": t.params.get("effectiveness"),
            "trial_elapsed_sec": t.user_attrs.get("trial_elapsed_sec"),
            "simulation_elapsed_sec": t.user_attrs.get("simulation_elapsed_sec"),
            "score_elapsed_sec": t.user_attrs.get("score_elapsed_sec"),
        })

    df = pd.DataFrame(rows)
    csv_path = LOG_DIR / f"timing_{METHOD_NAME.lower()}.csv"
    df.to_csv(csv_path, index=False)

    complete_df = df[df["state"].str.contains("COMPLETE", na=False)].copy()

    summary = {
        "method": METHOD_NAME,
        "network": NETWORK_NAME,
        "n_trials_total": len(df),
        "n_trials_complete": len(complete_df),
        "total_optimization_sec": total_elapsed_sec,
        "mean_trial_sec": None if complete_df.empty else float(complete_df["trial_elapsed_sec"].mean()),
        "mean_simulation_sec": None if complete_df.empty else float(complete_df["simulation_elapsed_sec"].mean()),
        "mean_score_sec": None if complete_df.empty else float(complete_df["score_elapsed_sec"].mean()),
        "best_score": study.best_value if len(study.trials) > 0 else None,
        "best_params": study.best_params if len(study.trials) > 0 else None,
        "timing_csv": str(csv_path),
    }

    json_path = LOG_DIR / f"summary_{METHOD_NAME.lower()}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Timing] Trial log saved to: {csv_path}")
    print(f"[Timing] Summary saved to: {json_path}")

def create_sampler(method_name: str):
    """手法名に応じて Optuna Sampler を返す"""
    name = method_name.upper()

    if name == "GPR":
        return BoTorchSampler(n_startup_trials=N_STARTUP_TRIALS, seed=42)

    elif name == "RANDOM":
        return optuna.samplers.RandomSampler(seed=42)

    elif name == "CMAES":
        return optuna.samplers.CmaEsSampler(seed=42)

    elif name == "GA":
        # 仮置き。必要ならここを正式な GA 実装に差し替える
        return optuna.samplers.NSGAIISampler(seed=42)

    else:
        raise ValueError(f"Unsupported METHOD_NAME: {method_name}")


# ==========================================
# 4. 最適化プロセス
# ==========================================
def objective(trial):
    identifier = f"trial_{trial.number}"
    trial_start = time.perf_counter()
    sim_elapsed = None
    score_elapsed = None

    x1 = trial.suggest_float("certainty", 0.5, 1.0)
    x2 = trial.suggest_float("effectiveness", 0.5, 1.0)

    csv_path = create_inhibition_csv(x1, x2, identifier)
    toml_path = create_strategy_toml(csv_path, identifier)

    try:
        sim_start = time.perf_counter()
        run_simulation(identifier, toml_path, show_output=False)
        sim_elapsed = time.perf_counter() - sim_start

        score_start = time.perf_counter()
        score = calc_score(identifier)
        score_elapsed = time.perf_counter() - score_start

        return score

    except subprocess.CalledProcessError:
        return 1.0

    finally:
        trial_elapsed = time.perf_counter() - trial_start
        trial.set_user_attr("trial_elapsed_sec", trial_elapsed)

        if sim_elapsed is not None:
            trial.set_user_attr("simulation_elapsed_sec", sim_elapsed)

        if score_elapsed is not None:
            trial.set_user_attr("score_elapsed_sec", score_elapsed)

        cleanup_files(identifier)


# ==========================================
# 5. メイン実行
# ==========================================
if __name__ == "__main__":
    if not RUST_BIN.exists():
        sys.exit(f"Error: Binary not found at {RUST_BIN}")

    if not AGENT_CONF.exists():
        sys.exit(f"Error: Agent config not found at {AGENT_CONF}")

    if not NETWORK_CONF.exists():
        sys.exit(f"Error: Network config not found at {NETWORK_CONF}")

    setup_directories()

    # DB初期化
    if DB_PATH.exists():
        try:
            DB_PATH.unlink()
            print(f"Deleted old database: {DB_PATH}")
        except PermissionError:
            print(f"Warning: Could not delete {DB_PATH}. Is it open?")

    print(f"=== Start Optimization (Network: {NETWORK_NAME}, Method: {METHOD_NAME}, Agents: {TOTAL_AGENTS}) ===")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = create_sampler(METHOD_NAME)

    study = optuna.create_study(
        study_name=f"optimize_test_{NETWORK_NAME}_{METHOD_NAME.lower()}",
        direction="minimize",
        sampler=sampler,
        storage=DB_URL,
        load_if_exists=True
    )

    print("Running trials...")

    with tqdm(total=N_TRIALS, desc="Optimization Progress", unit="trial") as pbar:

        def progress_callback(study, trial):
            pbar.update(1)
            best_val = study.best_value if study.best_value is not None else float("inf")
            pbar.set_postfix({"Best Score": f"{best_val:.5f}"})

        opt_start = time.perf_counter()
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[progress_callback])
        opt_elapsed = time.perf_counter() - opt_start

    print("\n=== Optimization Finished ===")
    print(f"Best Params: {study.best_params}")
    print(f"Best Score : {study.best_value}")
    print(f"Total Time : {opt_elapsed:.2f} sec")
    print(f"Avg Time/Trial : {opt_elapsed / N_TRIALS:.2f} sec")

    save_timing_report(study, opt_elapsed)

    print("\n=== Re-running Best Configuration ===")
    best_id = "trial_best"

    cleanup_files(best_id)

    best_csv = create_inhibition_csv(
        study.best_params["certainty"],
        study.best_params["effectiveness"],
        best_id
    )
    best_toml = create_strategy_toml(best_csv, best_id)

    try:
        run_simulation(best_id, best_toml, show_output=True)
        print(f"\n[Success] Results saved to: {RESULT_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Best simulation failed with exit code {e.returncode}.")