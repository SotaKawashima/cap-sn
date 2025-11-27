import pandas as pd
import matplotlib.pyplot as plt
import os

# 結果ファイルのパス
ARROW_PATH = "./v2/test_2/result/test_ba100_pop.arrow"

def plot_simulation_results():
    if not os.path.exists(ARROW_PATH):
        print(f"File not found: {ARROW_PATH}")
        return

    # データの読み込み
    df = pd.read_feather(ARROW_PATH)
    
    # 試行(num_iter)ごとにデータを分ける
    # 今回は1回の実行で複数イテレーション(モンテカルロ)回っている可能性があります
    unique_iters = df["num_iter"].unique()
    
    plt.figure(figsize=(10, 6))
    
    # 全試行の軌跡を薄くプロット
    for i in unique_iters:
        subset = df[df["num_iter"] == i]
        # 時間順にソート
        subset = subset.sort_values("t")
        plt.plot(subset["t"], subset["num_selfish"], color="blue", alpha=0.3, linewidth=1)

    # 平均の軌跡を太くプロット
    mean_df = df.groupby("t")["num_selfish"].mean().reset_index()
    plt.plot(mean_df["t"], mean_df["num_selfish"], color="red", linewidth=2, label="Average")

    plt.title("Transition of Selfish Behavior (test_ba100)")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Number of Selfish Agents")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    # 画像として保存
    plt.savefig("simulation_graph.png")
    print("Graph saved as 'simulation_graph.png'. Please check it.")
    # 環境によっては plt.show() で表示

if __name__ == "__main__":
    plot_simulation_results()