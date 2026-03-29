import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
categories = ['IC', 'Rank IC', 'ARR (%)', 'MDD (%)↓']
alphaseek = [0.0454, 0.0437, 8.28, 6.28]
without_iter = [0.0381, 0.0367, 3.01, 10.29]

# 计算提升百分比
improvements = []
for i in range(4):
    if i == 3:  # MDD越小越好，所以是下降
        pct = (without_iter[i] - alphaseek[i]) / without_iter[i] * 100
        improvements.append(f'-{pct:.1f}%')
    else:  # 其他指标越大越好
        pct = (alphaseek[i] - without_iter[i]) / without_iter[i] * 100
        improvements.append(f'+{pct:.1f}%')

# 颜色
color_main = '#C44536'  # 深红色 - AlphaSeek
color_ablation = '#B8B8B8'  # 灰色 - 消融方法

# 创建图形 (1行2列)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# 左图: IC 和 Rank IC
ax1 = axes[0]
x = np.arange(2)
width = 0.35

bars1 = ax1.bar(x - width/2, alphaseek[:2], width, label='AlphaSeek (Ours)',
                color=color_main, edgecolor='white', linewidth=0.8)
bars2 = ax1.bar(x + width/2, without_iter[:2], width, label='w/o Self-Iteration',
                color=color_ablation, edgecolor='white', linewidth=0.8)

# 添加纹理
for bar in bars1:
    bar.set_hatch('///')
for bar in bars2:
    bar.set_hatch('...')

# 添加数值标签和提升百分比
for i, (xi, vi_main, vi_abl, imp) in enumerate(zip(x, alphaseek[:2], without_iter[:2], improvements[:2])):
    # AlphaSeek数值
    ax1.text(xi - width/2, vi_main + 0.001, f'{vi_main:.4f}', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=color_main)
    # 消融方法数值
    ax1.text(xi + width/2, vi_abl + 0.001, f'{vi_abl:.4f}', ha='center', va='bottom',
            fontsize=8, color='#666666')
    # 提升百分比（放在柱子中间上方）
    max_height = max(vi_main, vi_abl)
    ax1.text(xi, max_height + 0.003, imp, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#2E7D32')

ax1.set_xticks(x)
ax1.set_xticklabels(['IC', 'Rank IC'], fontsize=11)
ax1.set_ylim(0.035, 0.050)
ax1.set_yticks([0.036, 0.038, 0.040, 0.042, 0.044, 0.046, 0.048, 0.050])
ax1.set_title('Factor Predictive Power (Higher is Better)', fontsize=12, fontweight='bold', pad=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.set_axisbelow(True)
ax1.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray', fontsize=9)

# 右图: ARR 和 MDD
ax2 = axes[1]
x = np.arange(2)

bars1 = ax2.bar(x - width/2, alphaseek[2:], width, label='AlphaSeek (Ours)',
                color=color_main, edgecolor='white', linewidth=0.8)
bars2 = ax2.bar(x + width/2, without_iter[2:], width, label='w/o Self-Iteration',
                color=color_ablation, edgecolor='white', linewidth=0.8)

# 添加纹理
for bar in bars1:
    bar.set_hatch('///')
for bar in bars2:
    bar.set_hatch('...')

# 添加数值标签
for i, (xi, vi_main, vi_abl, imp) in enumerate(zip(x, alphaseek[2:], without_iter[2:], improvements[2:])):
    # AlphaSeek数值
    ax2.text(xi - width/2, vi_main + 0.3, f'{vi_main:.2f}', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=color_main)
    # 消融方法数值
    ax2.text(xi + width/2, vi_abl + 0.3, f'{vi_abl:.2f}', ha='center', va='bottom',
            fontsize=8, color='#666666')
    # 提升百分比
    max_height = max(vi_main, vi_abl)
    ax2.text(xi, max_height + 0.8, imp, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#2E7D32')

ax2.set_xticks(x)
ax2.set_xticklabels(['ARR (%)↑', 'MDD (%)↓'], fontsize=11)
ax2.set_ylim(0, 12)
ax2.set_title('Strategy Performance', fontsize=12, fontweight='bold', pad=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_axisbelow(True)

# 添加总标题
fig.suptitle('Impact of Self-Iteration Mechanism (Ablation Study)', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# 保存
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_self_iteration.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_self_iteration.pdf', 
            bbox_inches='tight', facecolor='white')

print("图表已保存！")
print(f"IC提升: {improvements[0]}")
print(f"Rank IC提升: {improvements[1]}")
print(f"ARR提升: {improvements[2]}")
print(f"MDD降低: {improvements[3]}")
