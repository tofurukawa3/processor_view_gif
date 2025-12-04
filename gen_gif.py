import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# --- 設定 ---
# 5つのグラフを表示するため figsize をさらに横長に変更
fig, axes = plt.subplots(1, 5, figsize=(25, 6), facecolor='white')
fig.suptitle("CPU vs AVX2 vs GPU vs TPU vs NPU Processing (10x10 Matrix)", fontsize=20)

# 各軸の共通設定
for ax in axes:
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

# 共通グリッドサイズ
GRID_SIZE = 10
TOTAL_DATA = GRID_SIZE * GRID_SIZE  # 100個

# ==========================================
# 1. CPU: Scalar
# ==========================================
ax_cpu = axes[0]
ax_cpu.set_title("CPU: Scalar", fontsize=14, fontweight='bold')
for i in range(GRID_SIZE + 1):
    ax_cpu.axhline(i, color='black', lw=0.5)
    ax_cpu.axvline(i, color='black', lw=0.5)
ax_cpu.set_xlim(0, GRID_SIZE)
ax_cpu.set_ylim(0, GRID_SIZE)
ax_cpu.set_xlabel(f"Total Cycles: {TOTAL_DATA}\n(1 op / cycle)\n[Baseline]", fontsize=12, color='darkblue')
cpu_dot, = ax_cpu.plot([], [], 'ro', markersize=15)

# ==========================================
# 2. AVX2: SIMD
# ==========================================
ax_avx = axes[1]
ax_avx.set_title("AVX2: SIMD", fontsize=14, fontweight='bold')
for i in range(GRID_SIZE + 1):
    ax_avx.axhline(i, color='black', lw=0.5)
    ax_avx.axvline(i, color='black', lw=0.5)
ax_avx.set_xlim(0, GRID_SIZE)
ax_avx.set_ylim(0, GRID_SIZE)
ax_avx.set_xlabel(f"Total Cycles: {GRID_SIZE}\n(10 ops / cycle)\n[Vector]", fontsize=12, color='darkblue')
avx_dots, = ax_avx.plot([], [], 'ro', markersize=12)

# ==========================================
# 3. GPU: Parallel
# ==========================================
ax_gpu = axes[2]
ax_gpu.set_title("GPU: Parallel", fontsize=14, fontweight='bold')
for i in range(GRID_SIZE + 1):
    ax_gpu.axhline(i, color='black', lw=0.5)
    ax_gpu.axvline(i, color='black', lw=0.5)
ax_gpu.set_xlim(0, GRID_SIZE)
ax_gpu.set_ylim(0, GRID_SIZE)
ax_gpu.set_xlabel(f"Total Cycles: 1\n(100 ops / cycle)\n[Massive Parallel]", fontsize=12, color='darkblue')
gpu_dots, = ax_gpu.plot([], [], 'ro', markersize=8)
gpu_x, gpu_y = np.meshgrid(np.arange(0.5, GRID_SIZE, 1), np.arange(0.5, GRID_SIZE, 1))
gpu_x_flat = gpu_x.flatten()
gpu_y_flat = gpu_y.flatten()

# ==========================================
# 4. TPU: Systolic
# ==========================================
ax_tpu = axes[3]
ax_tpu.set_title("TPU: Systolic", fontsize=14, fontweight='bold')
for i in range(GRID_SIZE + 1):
    ax_tpu.axhline(i, color='black', lw=0.5)
    ax_tpu.axvline(i, color='black', lw=0.5)
ax_tpu.set_xlim(0, GRID_SIZE)
ax_tpu.set_ylim(0, GRID_SIZE)
ax_tpu.set_xlabel(f"Total Cycles: {GRID_SIZE}\n(Flow / Wave)\n[High Throughput]", fontsize=12, color='darkblue')
tpu_dots, = ax_tpu.plot([], [], 'ro', markersize=10)

# ==========================================
# 5. NPU: Tile / Block (New!)
# ==========================================
ax_npu = axes[4]
ax_npu.set_title("NPU: Tile/Block", fontsize=14, fontweight='bold')
for i in range(GRID_SIZE + 1):
    ax_npu.axhline(i, color='black', lw=0.5)
    ax_npu.axvline(i, color='black', lw=0.5)
ax_npu.set_xlim(0, GRID_SIZE)
ax_npu.set_ylim(0, GRID_SIZE)
# 4つの大きなブロック(5x5)に分けて処理するので、サイクル数は4と仮定
ax_npu.set_xlabel(f"Total Cycles: 4\n(25 ops / cycle)\n[Efficient Inference]", fontsize=12, color='darkblue')
npu_dots, = ax_npu.plot([], [], 'ro', markersize=10)


# ==========================================
# アニメーション更新関数
# ==========================================
def update(frame):
    # --- CPU Update (100 steps) ---
    cpu_step = frame % TOTAL_DATA
    cx = (cpu_step % GRID_SIZE) + 0.5
    cy = (GRID_SIZE - 1 - (cpu_step // GRID_SIZE)) + 0.5
    cpu_dot.set_data([cx], [cy])

    # --- AVX2 Update (Loop 10 times) ---
    avx_step = frame % GRID_SIZE
    avx_row = (GRID_SIZE - 1) - avx_step
    avx_x = np.arange(0.5, GRID_SIZE, 1)
    avx_y = np.full(GRID_SIZE, avx_row + 0.5)
    avx_dots.set_data(avx_x, avx_y)

    # --- GPU Update (Always flashing) ---
    if (frame % 10) < 5:
        gpu_dots.set_data(gpu_x_flat, gpu_y_flat)
    else:
        gpu_dots.set_data([], [])

    # --- TPU Update (Wave loop) ---
    tpu_col = frame % GRID_SIZE
    tx = np.full(GRID_SIZE, tpu_col + 0.5)
    ty = np.arange(0.5, GRID_SIZE, 1)
    tpu_dots.set_data(tx, ty)

    # --- NPU Update (Tile based) ---
    # 10x10の領域を4つの5x5ブロックに分割して処理するイメージ
    # 100フレームの間に何周かさせる (例えば25フレームで1周＝4倍速で表現)
    npu_cycle_frame = frame % 20 # 20フレームで1周する速さ
    block_idx = npu_cycle_frame // 5 # 5フレームごとにブロック移動 (計4ブロック)

    nx, ny = [], []
    
    # ブロックごとの座標定義 (左上, 右上, 左下, 右下 の順)
    if block_idx == 0:   # Top-Left
        xx, yy = np.meshgrid(np.arange(0.5, 5, 1), np.arange(5.5, 10, 1))
    elif block_idx == 1: # Top-Right
        xx, yy = np.meshgrid(np.arange(5.5, 10, 1), np.arange(5.5, 10, 1))
    elif block_idx == 2: # Bottom-Left
        xx, yy = np.meshgrid(np.arange(0.5, 5, 1), np.arange(0.5, 5, 1))
    elif block_idx == 3: # Bottom-Right
        xx, yy = np.meshgrid(np.arange(5.5, 10, 1), np.arange(0.5, 5, 1))
    else:
        xx, yy = [], []

    if block_idx < 4:
        npu_dots.set_data(xx.flatten(), yy.flatten())
    else:
        npu_dots.set_data([], [])

    return cpu_dot, avx_dots, gpu_dots, tpu_dots, npu_dots

# --- アニメーション生成と保存 ---
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=False)

output_filename = "processing_complexity_npu.gif"
ani.save(output_filename, writer='pillow', fps=10)

print(f"GIF animation saved to {output_filename}")