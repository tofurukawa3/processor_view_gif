import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# 設定
GRID_SIZE = 10
FRAMES = 100
INTERVAL = 100

# カラーマップの作成
# LPU: 青系 (Cool / Flow)
colors_lpu = [(0.1, 0.1, 0.1), (0, 0.8, 1)] # Dark Grey -> Cyan
cmap_lpu = LinearSegmentedColormap.from_list("LPU", colors_lpu)

# Traditional: オレンジ系 (Heat / Friction)
colors_trad = [(0.1, 0.1, 0.1), (1, 0.6, 0)] # Dark Grey -> Orange
cmap_trad = LinearSegmentedColormap.from_list("Traditional", colors_trad)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(top=0.85)

# 初期化
grid_lpu = np.zeros((GRID_SIZE, GRID_SIZE))
grid_trad = np.zeros((GRID_SIZE, GRID_SIZE))

im1 = ax1.imshow(grid_lpu, cmap=cmap_lpu, vmin=0, vmax=1)
im2 = ax2.imshow(grid_trad, cmap=cmap_trad, vmin=0, vmax=1)

ax1.set_title("LPU (Deterministic)\nNo Stalls, Perfect Flow", color='tab:blue', fontweight='bold')
ax2.set_title("CPU/GPU (Traditional)\nRandom Access & Memory Stalls", color='tab:orange', fontweight='bold')

ax1.axis('off')
ax2.axis('off')

# ストール（遅延）管理用変数
stall_counter = 0

def update(frame):
    # 修正箇所: grid_trad を global に追加
    global stall_counter, grid_trad
    
    # --- 1. LPUの更新 (常にスムーズ) ---
    # 波のような計算 (y座標と時間tのみで決まる = コンパイル時に確定)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            # 斜めの波を作る
            val = (frame - x - y) % 10
            # 波のピークだけ明るくする
            grid_lpu[y, x] = max(0, 1 - (val / 4)) if val < 4 else 0

    # --- 2. Traditionalの更新 (ランダム性あり) ---
    if stall_counter > 0:
        # ストール中（色が赤っぽくなり、動かない）
        stall_counter -= 1
        ax2.set_title(f"CPU/GPU: STALLING... ({stall_counter})", color='red')
    else:
        # ランダムにストール発生判定 (10%の確率)
        if np.random.rand() < 0.10:
            stall_counter = 8 # 8フレーム停止
        else:
            ax2.set_title("CPU/GPU (Traditional)\nProcessing...", color='tab:orange')
            # ランダムアクセス風の点滅
            # ここで代入を行っているため global宣言が必要でした
            grid_trad = np.random.rand(GRID_SIZE, GRID_SIZE) * 0.8
            # 閾値以下のノイズを消してメリハリをつける
            grid_trad[grid_trad < 0.5] = 0

    im1.set_data(grid_lpu)
    im2.set_data(grid_trad)
    
    return [im1, im2]
ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=INTERVAL, blit=True)

# GIFとして保存 (imagemagickまたはpillowを使用)
ani.save('processor_simulation.gif', writer='pillow', fps=15)
print("GIF generation complete: processor_simulation.gif")