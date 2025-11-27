import pandas as pd
import os
import glob

# ==========================================
# 設定: 確認対象のディレクトリ
# ==========================================
# test_2 の結果フォルダを指定
TARGET_DIR = "./v2/test_2/result"

def check_pop_file(file_path):
    """
    pop.arrow の確認 (マクロな統計)
    - 利己的行動者数(num_selfish)の推移を確認
    """
    print(f"\n[Checking POP file]: {os.path.basename(file_path)}")
    try:
        df = pd.read_feather(file_path)
        print(f"  Shape: {df.shape}")
        
        # 必須カラムの確認
        required = ["t", "num_selfish", "num_iter"]
        if not all(col in df.columns for col in required):
            print(f"  [Warning] Missing columns. Found: {df.columns.tolist()}")
            return

        # 統計量の表示
        print("  --- Statistics ---")
        print(f"  Steps recorded: {df['t'].max() + 1}")
        print(f"  Iterations: {df['num_iter'].nunique()} (Trial IDs: {df['num_iter'].unique()})")
        
        # 利己的行動の発生状況
        max_selfish = df["num_selfish"].max()
        print(f"  Max Selfish Count: {max_selfish}")
        
        if max_selfish == 0:
            print("  [Result] No selfish behavior observed (Peaceful).")
        else:
            print("  [Result] Selfish behavior detected!")
            # 発生した時の様子を表示
            print("  -> Snapshot (Last 5 rows with selfish > 0):")
            print(df[df["num_selfish"] > 0].tail(5).to_string(index=False))

    except Exception as e:
        print(f"  [Error] Failed to read pop file: {e}")

def check_info_file(file_path):
    """
    info.arrow の確認 (情報の拡散状況)
    - どの情報(info_id)がどれくらい拡散したか
    """
    print(f"\n[Checking INFO file]: {os.path.basename(file_path)}")
    try:
        df = pd.read_feather(file_path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")

        # 情報IDごとの拡散数集計
        # カラム名推定: 'info_id', 'count', 'num_agents' など
        # 先行研究コード等の傾向から 'id' や 'count' を探す
        
        # もし 'info_id' がある場合
        if "info_id" in df.columns:
            print("  --- Information Diffusion Summary ---")
            # 情報IDごとに、各ステップでの最大拡散数を表示
            counts = df.groupby("info_id").size()
            print(f"  Unique Info IDs: {df['info_id'].unique()}")
            
            # 各情報の最新ステップでの到達人数を表示したい
            # tの最大値を取得
            last_t = df["t"].max()
            last_df = df[df["t"] == last_t]
            
            print(f"  At final step (t={last_t}):")
            if "count" in df.columns:
                print(last_df[["info_id", "count"]].to_string(index=False))
            elif "num" in df.columns: # カラム名が num の場合
                print(last_df[["info_id", "num"]].to_string(index=False))
            else:
                print(last_df.head().to_string(index=False))
        else:
            print("  [Info] Content overview (Head 5 rows):")
            print(df.head(5).to_string(index=False))

    except Exception as e:
        print(f"  [Error] Failed to read info file: {e}")

def check_agent_file(file_path):
    """
    agent.arrow の確認 (個々のエージェントの状態)
    - エージェントの内部状態やオピニオン値の確認
    """
    print(f"\n[Checking AGENT file]: {os.path.basename(file_path)}")
    try:
        df = pd.read_feather(file_path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # エージェントの状態やオピニオンを確認
        # 例: state, opinion_b, opinion_u など
        
        print("  --- Agent States Summary ---")
        if "id" in df.columns:
            print(f"  Total Agents recorded: {df['id'].nunique()}")
        
        # データの欠損や異常値チェック
        if df.isnull().values.any():
            print("  [Warning] Contains NaN/Null values!")
        else:
            print("  [OK] No NaN values found.")

        # サンプル表示 (最初のエージェントの時系列変化など)
        if "id" in df.columns and "t" in df.columns:
            agent_0 = df[df["id"] == 0]
            print(f"  Agent 0 History (First 5 steps):")
            print(agent_0.head(5).to_string(index=False))
        else:
            print("  Content overview (Head 5 rows):")
            print(df.head(5).to_string(index=False))

    except Exception as e:
        print(f"  [Error] Failed to read agent file: {e}")

def main():
    print(f"=== Inspecting Simulation Results in: {TARGET_DIR} ===\n")
    
    # ファイルを探す
    pop_files = glob.glob(os.path.join(TARGET_DIR, "*_pop.arrow"))
    info_files = glob.glob(os.path.join(TARGET_DIR, "*_info.arrow"))
    agent_files = glob.glob(os.path.join(TARGET_DIR, "*_agent.arrow"))
    
    if not (pop_files or info_files or agent_files):
        print("No .arrow files found. Please run the simulation first.")
        return

    # 1. POPファイルの確認
    for f in pop_files:
        check_pop_file(f)
        
    # 2. INFOファイルの確認
    for f in info_files:
        check_info_file(f)
        
    # 3. AGENTファイルの確認
    for f in agent_files:
        check_agent_file(f)

    print("\n=== Inspection Finished ===")

if __name__ == "__main__":
    main()