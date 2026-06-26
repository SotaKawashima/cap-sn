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
import math
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
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Override total number of agents for score normalization"
    )
    parser.add_argument(
        "--score-metric",
        type=str,
        default="auc",
        choices=["auc", "final", "peak", "final-window"],
        help=(
            "Optimization score metric. "
            "'auc' uses mean selfish ratio over all timesteps, "
            "'final' keeps the original final-timestep score."
        )
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Override optimizer sampler seed. Default uses a different seed for each method."
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Optional label appended to the output run directory, e.g. 20260617."
    )
    parser.add_argument(
        "--replicate-label",
        type=str,
        default="",
        help="Optional label appended to the method directory, e.g. optseed1."
    )
    parser.add_argument(
        "--keep-trial-raw",
        type=str,
        default="none",
        choices=["none", "info-pop", "all"],
        help=(
            "Keep raw Arrow files for every optimization trial. "
            "'none' deletes trial Arrow files after scoring, "
            "'info-pop' keeps info/pop Arrow files, "
            "'all' keeps info/pop/agent Arrow files."
        )
    )
    return parser.parse_args()


args = parse_args()


# ==========================================
# 2. 設定 & 定数
# ==========================================

METHOD_NAME = args.method
NETWORK_NAME = args.network
N_TRIALS = args.trials
SCORE_METRIC = args.score_metric
RUN_LABEL = args.run_label.strip()
REPLICATE_LABEL = args.replicate_label.strip()
KEEP_TRIAL_RAW = args.keep_trial_raw

BASE_DIR = Path(__file__).resolve().parent
ENV_ROOT = BASE_DIR / "v2" / "test_2"

NETWORK_CONF_MAP = {
    "ba_1000": ENV_ROOT / "network" / "network-ba1000.toml",
    "facebook": ENV_ROOT / "network" / "network-facebook.toml",
    "wiki-vote": ENV_ROOT / "network" / "network-wiki-vote.toml",
}

# ネットワークごとの総エージェント数
NETWORK_AGENT_COUNT_MAP = {
    "ba_1000": 1000,
    "facebook": 4039,
    "wiki-vote": 7115,
}

if NETWORK_NAME not in NETWORK_CONF_MAP:
    raise ValueError(f"Unsupported NETWORK_NAME: {NETWORK_NAME}")

if NETWORK_NAME not in NETWORK_AGENT_COUNT_MAP:
    raise ValueError(f"Unsupported NETWORK_NAME for agent count: {NETWORK_NAME}")

# 引数 --agents が与えられたときはそれを優先
TOTAL_AGENTS = args.agents if args.agents is not None else NETWORK_AGENT_COUNT_MAP[NETWORK_NAME]

if TOTAL_AGENTS <= 0:
    raise ValueError(f"TOTAL_AGENTS must be positive, got {TOTAL_AGENTS}")

if "/" in RUN_LABEL or "\\" in RUN_LABEL:
    raise ValueError("--run-label must not contain path separators")

if "/" in REPLICATE_LABEL or "\\" in REPLICATE_LABEL:
    raise ValueError("--replicate-label must not contain path separators")

SAMPLER_SEED_MAP = {
    "GPR": 4201,
    "CMAES": 4202,
    "RANDOM": 4203,
    "GA": 4204,
}
SAMPLER_SEED = args.sampler_seed if args.sampler_seed is not None else SAMPLER_SEED_MAP[METHOD_NAME]

# ---- 出力先フォルダ構成 ----
# 例:
# experiments/optimization_ba1000/optimize_runs/gpr/
# experiments/optimization_facebook/optimize_runs/cmaes/
# experiments/optimization_wiki_vote/optimize_runs/random/
#
# AUCなど新しい指標で再実行するときに既存のfinal指標結果を上書きしないよう、
# final以外は optimize_runs_<metric> に分けて保存する。
RUN_DIR_NAME = "optimize_runs" if SCORE_METRIC == "final" else f"optimize_runs_{SCORE_METRIC.replace('-', '_')}"
if RUN_LABEL:
    RUN_DIR_NAME = f"{RUN_DIR_NAME}_{RUN_LABEL}"

OPTIMIZE_OUTPUT_DIR_MAP = {
    "ba_1000": BASE_DIR / "experiments" / "optimization_ba1000" / RUN_DIR_NAME,
    "facebook": BASE_DIR / "experiments" / "optimization_facebook" / RUN_DIR_NAME,
    "wiki-vote": BASE_DIR / "experiments" / "optimization_wiki_vote" / RUN_DIR_NAME,
}

NETWORK_OUTPUT_DIR = OPTIMIZE_OUTPUT_DIR_MAP[NETWORK_NAME]
METHOD_DIR_NAME = METHOD_NAME.lower()
if REPLICATE_LABEL:
    METHOD_DIR_NAME = f"{METHOD_DIR_NAME}_{REPLICATE_LABEL}"

METHOD_DIR = NETWORK_OUTPUT_DIR / METHOD_DIR_NAME

RESULT_DIR = METHOD_DIR / "result"
TRIAL_RAW_DIR = RESULT_DIR / "trials"
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

N_STARTUP_TRIALS = 10
FINAL_WINDOW_FRACTION = 0.10
MIN_FINAL_WINDOW_STEPS = 3


# ==========================================
# 3. ヘルパー関数
# ==========================================

def setup_directories():
    """必要なディレクトリを一括作成"""
    for d in [NETWORK_OUTPUT_DIR, METHOD_DIR, RESULT_DIR, CSV_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if KEEP_TRIAL_RAW != "none":
        TRIAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

def kept_trial_arrow_kinds() -> tuple[str, ...]:
    """全trial保存対象のArrow種別を返す"""
    if KEEP_TRIAL_RAW == "info-pop":
        return ("info", "pop")
    if KEEP_TRIAL_RAW == "all":
        return ("info", "pop", "agent")
    return ()

def format_run_path(path: Path | None) -> str | None:
    """runディレクトリ内の相対パスに整形する。存在しない場合はNone。"""
    if path is None:
        return None
    try:
        return path.relative_to(METHOD_DIR).as_posix()
    except ValueError:
        return path.as_posix()

def preserve_trial_raw_arrows(identifier: str, trial_number: int) -> dict[str, str | None]:
    """
    指定trialのArrowを result/trials/ に移動して保存する。
    戻り値は timing CSV に保存するための相対パス。
    """
    saved = {
        "trial_info_arrow": None,
        "trial_pop_arrow": None,
        "trial_agent_arrow": None,
    }
    if KEEP_TRIAL_RAW == "none":
        return saved

    TRIAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_stem = f"trial_{trial_number:04d}"

    for kind in kept_trial_arrow_kinds():
        src = RESULT_DIR / f"{identifier}_{kind}.arrow"
        if not src.exists():
            continue
        dst = TRIAL_RAW_DIR / f"{raw_stem}_{kind}.arrow"
        dst.unlink(missing_ok=True)
        src.replace(dst)
        saved[f"trial_{kind}_arrow"] = format_run_path(dst)

    return saved

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

def read_pop_arrow(identifier: str) -> pd.DataFrame | None:
    """pop.arrowを読み込む。読めない場合はNoneを返す。"""
    arrow_path = RESULT_DIR / f"{identifier}_pop.arrow"
    if not arrow_path.exists():
        return None

    try:
        df = pd.read_feather(arrow_path)
    except Exception as e:
        print(f"  [Error] pop.arrow read failed: {e}", file=sys.stderr)
        return None

    if "num_selfish" not in df.columns:
        return None

    return df

def calc_final_selfish_score(identifier: str) -> float:
    """
    旧スコア。

    各 num_iter の最終時刻 t における num_selfish を取り、
    その平均を TOTAL_AGENTS で割った割合を返す
    """
    df = read_pop_arrow(identifier)
    if df is None:
        return 1.0

    try:
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
        print(f"  [Error] Final score calculation failed: {e}", file=sys.stderr)
        return 1.0

def calc_normalized_selfish_auc_score(identifier: str) -> float:
    """
    新しい主スコア。

    各 num_iter について全時刻の num_selfish / TOTAL_AGENTS を平均し、
    さらに num_iter 間で平均する。時系列全体で利己的行動がどれだけ
    存在したかを0-1系の値として評価する。
    """
    df = read_pop_arrow(identifier)
    if df is None:
        return 1.0

    try:
        if "num_iter" in df.columns and "t" in df.columns:
            sorted_df = df.sort_values(["num_iter", "t"]).copy()
            sorted_df["selfish_ratio"] = sorted_df["num_selfish"] / TOTAL_AGENTS
            iter_auc = sorted_df.groupby("num_iter")["selfish_ratio"].mean()
            return float(iter_auc.mean())

        return float(df["num_selfish"].mean()) / TOTAL_AGENTS

    except Exception as e:
        print(f"  [Error] AUC score calculation failed: {e}", file=sys.stderr)
        return 1.0

def calc_peak_selfish_score(identifier: str) -> float:
    """
    補助候補スコア。

    各 num_iter の peak selfish ratio を平均する。
    """
    df = read_pop_arrow(identifier)
    if df is None:
        return 1.0

    try:
        if "num_iter" in df.columns:
            iter_peak = df.groupby("num_iter")["num_selfish"].max()
            return float(iter_peak.mean()) / TOTAL_AGENTS

        return float(df["num_selfish"].max()) / TOTAL_AGENTS

    except Exception as e:
        print(f"  [Error] Peak score calculation failed: {e}", file=sys.stderr)
        return 1.0

def calc_final_window_selfish_score(identifier: str) -> float:
    """
    補助候補スコア。

    各 num_iter の最後10%または最低3ステップの selfish ratio を平均し、
    さらに num_iter 間で平均する。
    """
    df = read_pop_arrow(identifier)
    if df is None:
        return 1.0

    try:
        if "num_iter" in df.columns and "t" in df.columns:
            scores = []
            for _, group in df.sort_values(["num_iter", "t"]).groupby("num_iter"):
                n_steps = len(group)
                final_window_steps = min(
                    n_steps,
                    max(MIN_FINAL_WINDOW_STEPS, math.ceil(n_steps * FINAL_WINDOW_FRACTION)),
                )
                selfish_ratio = group["num_selfish"].tail(final_window_steps) / TOTAL_AGENTS
                scores.append(float(selfish_ratio.mean()))

            return float(sum(scores) / len(scores)) if scores else 1.0

        final_window_steps = min(
            len(df),
            max(MIN_FINAL_WINDOW_STEPS, math.ceil(len(df) * FINAL_WINDOW_FRACTION)),
        )
        return float(df["num_selfish"].tail(final_window_steps).mean()) / TOTAL_AGENTS

    except Exception as e:
        print(f"  [Error] Final-window score calculation failed: {e}", file=sys.stderr)
        return 1.0

def compute_pop_metric_report(df: pd.DataFrame) -> dict:
    """pop.arrowの時系列から、目的関数候補と時刻長の補助指標をまとめて計算する。"""
    report = {
        "final_selfish_ratio": None,
        "normalized_selfish_auc": None,
        "peak_selfish_ratio": None,
        "final_window_selfish_mean": None,
        "n_rows": int(len(df)),
        "n_iter": None,
        "mean_n_steps": None,
        "min_n_steps": None,
        "max_n_steps": None,
        "mean_t_max": None,
        "min_t_max": None,
        "max_t_max": None,
    }

    if len(df) == 0 or "num_selfish" not in df.columns:
        return report

    if "num_iter" in df.columns and "t" in df.columns:
        ordered = df.sort_values(["num_iter", "t"]).copy()
        ordered["selfish_ratio"] = ordered["num_selfish"] / TOTAL_AGENTS
        grouped = ordered.groupby("num_iter", sort=True)

        n_steps = grouped.size()
        t_max = grouped["t"].max()
        final_rows = grouped.tail(1)

        final_window_scores = []
        for _, group in grouped:
            n = len(group)
            final_window_steps = min(
                n,
                max(MIN_FINAL_WINDOW_STEPS, math.ceil(n * FINAL_WINDOW_FRACTION)),
            )
            final_window_scores.append(float(group["selfish_ratio"].tail(final_window_steps).mean()))

        report.update({
            "final_selfish_ratio": float(final_rows["selfish_ratio"].mean()),
            "normalized_selfish_auc": float(grouped["selfish_ratio"].mean().mean()),
            "peak_selfish_ratio": float(grouped["selfish_ratio"].max().mean()),
            "final_window_selfish_mean": float(sum(final_window_scores) / len(final_window_scores)),
            "n_iter": int(grouped.ngroups),
            "mean_n_steps": float(n_steps.mean()),
            "min_n_steps": int(n_steps.min()),
            "max_n_steps": int(n_steps.max()),
            "mean_t_max": float(t_max.mean()),
            "min_t_max": int(t_max.min()),
            "max_t_max": int(t_max.max()),
        })
        return report

    selfish_ratio = df["num_selfish"] / TOTAL_AGENTS
    final_window_steps = min(
        len(df),
        max(MIN_FINAL_WINDOW_STEPS, math.ceil(len(df) * FINAL_WINDOW_FRACTION)),
    )
    report.update({
        "final_selfish_ratio": float(selfish_ratio.iloc[-1]),
        "normalized_selfish_auc": float(selfish_ratio.mean()),
        "peak_selfish_ratio": float(selfish_ratio.max()),
        "final_window_selfish_mean": float(selfish_ratio.tail(final_window_steps).mean()),
        "n_iter": 1,
        "mean_n_steps": float(len(df)),
        "min_n_steps": int(len(df)),
        "max_n_steps": int(len(df)),
    })
    return report

def score_key() -> str:
    if SCORE_METRIC == "auc":
        return "normalized_selfish_auc"
    if SCORE_METRIC == "final":
        return "final_selfish_ratio"
    if SCORE_METRIC == "peak":
        return "peak_selfish_ratio"
    if SCORE_METRIC == "final-window":
        return "final_window_selfish_mean"

    return "score"

def calc_score_report(identifier: str) -> dict:
    """選択スコアと補助指標をまとめて返す。失敗時はscore=1.0にする。"""
    df = read_pop_arrow(identifier)
    if df is None:
        return {"score": 1.0, "score_failed": True}

    try:
        report = compute_pop_metric_report(df)
        selected = report.get(score_key())
        report["score"] = 1.0 if selected is None else float(selected)
        report["score_failed"] = selected is None
        return report
    except Exception as e:
        print(f"  [Error] Score report calculation failed: {e}", file=sys.stderr)
        return {"score": 1.0, "score_failed": True}

def calc_score(identifier: str) -> float:
    """選択されたscore metricで最適化用スコアを計算する。"""
    return float(calc_score_report(identifier)["score"])

def score_definition() -> str:
    if SCORE_METRIC == "auc":
        return "mean over num_iter of mean_t(num_selfish / total_agents)"
    if SCORE_METRIC == "final":
        return "original metric: mean over num_iter of final_t(num_selfish / total_agents)"
    if SCORE_METRIC == "peak":
        return "mean over num_iter of max_t(num_selfish / total_agents)"
    if SCORE_METRIC == "final-window":
        return "mean over num_iter of final-window mean(num_selfish / total_agents)"

    return "unknown"

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
            "score_metric": SCORE_METRIC,
            "replicate_label": REPLICATE_LABEL or None,
            "sampler_seed": SAMPLER_SEED,
            "certainty": t.params.get("certainty"),
            "effectiveness": t.params.get("effectiveness"),
            "final_selfish_ratio": t.user_attrs.get("final_selfish_ratio"),
            "normalized_selfish_auc": t.user_attrs.get("normalized_selfish_auc"),
            "peak_selfish_ratio": t.user_attrs.get("peak_selfish_ratio"),
            "final_window_selfish_mean": t.user_attrs.get("final_window_selfish_mean"),
            "n_rows": t.user_attrs.get("n_rows"),
            "n_iter": t.user_attrs.get("n_iter"),
            "mean_n_steps": t.user_attrs.get("mean_n_steps"),
            "min_n_steps": t.user_attrs.get("min_n_steps"),
            "max_n_steps": t.user_attrs.get("max_n_steps"),
            "mean_t_max": t.user_attrs.get("mean_t_max"),
            "min_t_max": t.user_attrs.get("min_t_max"),
            "max_t_max": t.user_attrs.get("max_t_max"),
            "score_failed": t.user_attrs.get("score_failed"),
            "trial_elapsed_sec": t.user_attrs.get("trial_elapsed_sec"),
            "simulation_elapsed_sec": t.user_attrs.get("simulation_elapsed_sec"),
            "score_elapsed_sec": t.user_attrs.get("score_elapsed_sec"),
            "trial_info_arrow": t.user_attrs.get("trial_info_arrow"),
            "trial_pop_arrow": t.user_attrs.get("trial_pop_arrow"),
            "trial_agent_arrow": t.user_attrs.get("trial_agent_arrow"),
        })

    df = pd.DataFrame(rows)
    csv_path = LOG_DIR / f"timing_{METHOD_NAME.lower()}.csv"
    df.to_csv(csv_path, index=False)

    complete_df = df[df["state"].str.contains("COMPLETE", na=False)].copy()

    best_trial_attrs = study.best_trial.user_attrs if len(complete_df) > 0 else {}

    summary = {
        "method": METHOD_NAME,
        "method_dir_name": METHOD_DIR_NAME,
        "network": NETWORK_NAME,
        "score_metric": SCORE_METRIC,
        "score_definition": score_definition(),
        "replicate_label": REPLICATE_LABEL or None,
        "sampler_seed": SAMPLER_SEED,
        "keep_trial_raw": KEEP_TRIAL_RAW,
        "trial_raw_dir": None if KEEP_TRIAL_RAW == "none" else format_run_path(TRIAL_RAW_DIR),
        "run_dir": str(NETWORK_OUTPUT_DIR),
        "total_agents": TOTAL_AGENTS,
        "n_trials_total": len(df),
        "n_trials_complete": len(complete_df),
        "total_optimization_sec": total_elapsed_sec,
        "mean_trial_sec": None if complete_df.empty else float(complete_df["trial_elapsed_sec"].mean()),
        "mean_simulation_sec": None if complete_df.empty else float(complete_df["simulation_elapsed_sec"].mean()),
        "mean_score_sec": None if complete_df.empty else float(complete_df["score_elapsed_sec"].mean()),
        "best_score": study.best_value if len(study.trials) > 0 else None,
        "best_params": study.best_params if len(study.trials) > 0 else None,
        "best_trial_metrics": {
            "final_selfish_ratio": best_trial_attrs.get("final_selfish_ratio"),
            "normalized_selfish_auc": best_trial_attrs.get("normalized_selfish_auc"),
            "peak_selfish_ratio": best_trial_attrs.get("peak_selfish_ratio"),
            "final_window_selfish_mean": best_trial_attrs.get("final_window_selfish_mean"),
            "mean_n_steps": best_trial_attrs.get("mean_n_steps"),
            "min_n_steps": best_trial_attrs.get("min_n_steps"),
            "max_n_steps": best_trial_attrs.get("max_n_steps"),
            "mean_t_max": best_trial_attrs.get("mean_t_max"),
            "min_t_max": best_trial_attrs.get("min_t_max"),
            "max_t_max": best_trial_attrs.get("max_t_max"),
        },
        "timing_csv": str(csv_path),
    }

    json_path = LOG_DIR / f"summary_{METHOD_NAME.lower()}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Timing] Trial log saved to: {csv_path}")
    print(f"[Timing] Summary saved to: {json_path}")

def create_sampler(method_name: str, seed: int):
    """手法名に応じて Optuna Sampler を返す"""
    name = method_name.upper()

    if name == "GPR":
        return BoTorchSampler(n_startup_trials=N_STARTUP_TRIALS, seed=seed)

    elif name == "RANDOM":
        return optuna.samplers.RandomSampler(seed=seed)

    elif name == "CMAES":
        return optuna.samplers.CmaEsSampler(seed=seed)

    elif name == "GA":
        # 仮置き。必要ならここを正式な GA 実装に差し替える
        return optuna.samplers.NSGAIISampler(seed=seed)

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
        score_report = calc_score_report(identifier)
        score = score_report["score"]
        score_elapsed = time.perf_counter() - score_start

        for key, value in score_report.items():
            trial.set_user_attr(key, value)

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

        raw_paths = preserve_trial_raw_arrows(identifier, trial.number)
        for key, value in raw_paths.items():
            trial.set_user_attr(key, value)

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

    print(
        "=== Start Optimization "
        f"(Network: {NETWORK_NAME}, Method: {METHOD_NAME}, "
        f"Score: {SCORE_METRIC}, Agents: {TOTAL_AGENTS}, "
        f"Replicate: {REPLICATE_LABEL or 'none'}) ==="
    )
    print(f"Score definition: {score_definition()}")
    print(f"Sampler seed    : {SAMPLER_SEED}")
    print(f"Method dir      : {METHOD_DIR_NAME}")
    print(f"Output dir      : {NETWORK_OUTPUT_DIR}")
    print(f"Keep trial raw  : {KEEP_TRIAL_RAW}")
    if KEEP_TRIAL_RAW != "none":
        print(f"Trial raw dir   : {TRIAL_RAW_DIR}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = create_sampler(METHOD_NAME, SAMPLER_SEED)

    study = optuna.create_study(
        study_name=f"optimize_test_{NETWORK_NAME}_{METHOD_NAME.lower()}_{SCORE_METRIC.replace('-', '_')}",
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
    print(f"Score Metric: {SCORE_METRIC}")
    print(f"Sampler Seed: {SAMPLER_SEED}")
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
