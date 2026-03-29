import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
categories = ['IC', 'Rank IC', 'ARR (%)', 'MDD (%)↓']
alphaseek = [0.0454, 0.0437, 8.28, 6.28]
without_iter = [0.0381, 0.0367, 3.01, 10.29]

# 计算百分比提升
improvements = []
for i in range(4):
    if i == 3:  # MDD越小越好
        pct = (without_iter[i] - alphaseek[i]) / without_iter[i] * 100
        improvements.append(f'-{pct:.1f}%')
    else:
        pct = (alphaseek[i] - without_iter[i]) / without_iter[i] * 100
        improvements.append(f'+{pct:.1f}%')

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

y = np.arange(len(categories))
height = 0.35

color_main = '#C44536'
color_abl = '#B8B8B8'

# 绘制水平条形图
bars1 = ax.barh(y - height/2, alphaseek, height, label='AlphaSeek (Ours)',
                color=color_main, edgecolor='white', linewidth=0.8)
bars2 = ax.barh(y + height/2, without_iter, height, label='w/o Self-Iteration',
                color=color_abl, edgecolor='white', linewidth=0.8)

# 添加纹理
for bar in bars1:
    bar.set_hatch('///')
for bar in bars2:
    bar.set_hatch('...')

# 添加数值标签
for i, (cat, alpha_val, abl_val, imp) in enumerate(zip(categories, alphaseek, without_iter, improvements)):
    # AlphaSeek数值
    ax.text(alpha_val + 0.001, i - height/2, f'{alpha_val:.4f}' if i < 2 else f'{alpha_val:.2f}', 
            ha='left', va='center', fontsize=9, fontweight='bold', color=color_main)
    # 消融方法数值
    ax.text(abl_val + 0.001, i + height/2, f'{abl_val:.4f}' if i < 2 else f'{abl_val:.2f}', 
            ha='left', va='center', fontsize=9, color='#666666')
    
    # 提升百分比（放在条形之间）
    mid_point = (alpha_val + abl_val) / 2
    ax.text(mid_point, i, imp, ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2E7D32', alpha=0.9))

# 添加连接线
for i in range(len(categories)):
    ax.plot([without_iter[i], alphaseek[i]], [i + height/2, i - height/2], 
            'k-', alpha=0.2, linewidth=1.5, zorder=0)

ax.set_yticks(y)
ax.set_yticklabels(categories, fontsize=11)
ax.set_xlabel('Metric Value', fontsize=12)
ax.set_xlim(0, max(max(alphaseek), max(without_iter)) * 1.3)
ax.set_title('Self-Iteration Impact: Horizontal Bar Comparison', 
            fontsize=13, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='gray', fontsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()

# 保存
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_horizontal.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_horizontal.pdf', 
            bbox_inches='tight', facecolor='white')

print("水平条形图已保存！")
print("文件: /root/lanyun-tmp/补充实验/figure/ablation_horizontal.png")
