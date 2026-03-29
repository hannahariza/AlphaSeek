import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
categories = ['IC', 'Rank IC', 'ARR (%)', 'MDD (%)↓']
alphaseek = [0.0454, 0.0437, 8.28, 6.28]
without_iter = [0.0381, 0.0367, 3.01, 10.29]

# 计算差距
gaps = []
for i in range(4):
    if i == 3:  # MDD越小越好
        gaps.append(without_iter[i] - alphaseek[i])
    else:
        gaps.append(alphaseek[i] - without_iter[i])

# 创建图形 (2x2子图)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

color_main = '#C44536'
color_abl = '#B8B8B8'
color_gap = '#2E7D32'

for idx, (cat, alpha_val, abl_val, gap) in enumerate(zip(categories, alphaseek, without_iter, gaps)):
    ax = axes[idx]
    
    x = [0, 1]
    y = [abl_val, alpha_val]
    labels = ['w/o Self-Iter', 'AlphaSeek']
    
    # 绘制基线
    ax.hlines(y=abl_val, xmin=0, xmax=1, colors=color_abl, linestyles='--', alpha=0.5, linewidth=1.5)
    
    # 绘制棒棒糖
    ax.scatter([0], [abl_val], s=200, c=color_abl, zorder=3, marker='o', edgecolors='white', linewidth=2)
    ax.scatter([1], [alpha_val], s=200, c=color_main, zorder=3, marker='o', edgecolors='white', linewidth=2)
    
    # 连接线
    ax.plot([0, 1], [abl_val, alpha_val], '-', color='gray', alpha=0.5, linewidth=2, zorder=1)
    
    # 添加数值标签
    ax.text(0, abl_val, f'{abl_val:.4f}' if idx < 2 else f'{abl_val:.2f}', 
            ha='center', va='bottom' if alpha_val > abl_val else 'top',
            fontsize=10, color=color_abl, fontweight='bold')
    ax.text(1, alpha_val, f'{alpha_val:.4f}' if idx < 2 else f'{alpha_val:.2f}', 
            ha='center', va='bottom',
            fontsize=10, color=color_main, fontweight='bold')
    
    # 添加提升/降低标签
    mid_x = 0.5
    mid_y = (abl_val + alpha_val) / 2
    if idx == 3:  # MDD
        ax.annotate('', xy=(1, alpha_val), xytext=(0, abl_val),
                   arrowprops=dict(arrowstyle='->', color=color_gap, lw=2))
        ax.text(mid_x, mid_y + gap*0.3, f'↓{gap:.2f}', ha='center', va='center',
               fontsize=11, color=color_gap, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_gap, alpha=0.8))
    else:
        ax.annotate('', xy=(1, alpha_val), xytext=(0, abl_val),
                   arrowprops=dict(arrowstyle='->', color=color_gap, lw=2))
        pct = (gap / abl_val) * 100
        ax.text(mid_x, mid_y + gap*0.3, f'+{pct:.1f}%', ha='center', va='center',
               fontsize=11, color=color_gap, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_gap, alpha=0.8))
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(min(y) * 0.8, max(y) * 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['w/o Self-Iter', 'AlphaSeek'], fontsize=10)
    ax.set_title(cat, fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

fig.suptitle('Self-Iteration Impact: Lollipop Comparison (4 Metrics)', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# 保存
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_lollipop.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_lollipop.pdf', 
            bbox_inches='tight', facecolor='white')

print("棒棒糖图已保存！")
print("文件: /root/lanyun-tmp/补充实验/figure/ablation_lollipop.png")
