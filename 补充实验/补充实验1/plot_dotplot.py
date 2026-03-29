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

color_main = '#C44536'
color_abl = '#999999'

# 绘制点
ax.scatter(without_iter, y, s=300, c=color_abl, marker='s', 
          edgecolors='white', linewidth=2, label='w/o Self-Iteration', zorder=3)
ax.scatter(alphaseek, y, s=400, c=color_main, marker='D', 
          edgecolors='white', linewidth=2, label='AlphaSeek (Ours)', zorder=3)

# 绘制连接线
for i, (cat, alpha_val, abl_val) in enumerate(zip(categories, alphaseek, without_iter)):
    ax.plot([abl_val, alpha_val], [i, i], '--', color='gray', alpha=0.5, linewidth=2, zorder=1)
    
    # 添加数值标签
    ax.text(abl_val - 0.002 if i < 2 else abl_val - 0.5, i - 0.2, 
            f'{abl_val:.4f}' if i < 2 else f'{abl_val:.2f}', 
            ha='right', va='top', fontsize=9, color=color_abl, fontweight='bold')
    ax.text(alpha_val + 0.002 if i < 2 else alpha_val + 0.3, i + 0.2, 
            f'{alpha_val:.4f}' if i < 2 else f'{alpha_val:.2f}', 
            ha='left', va='bottom', fontsize=9, color=color_main, fontweight='bold')
    
    # 提升百分比
    mid_x = (abl_val + alpha_val) / 2
    ax.annotate('', xy=(alpha_val, i), xytext=(abl_val, i),
               arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.5))
    ax.text(mid_x, i + 0.25, improvements[i], ha='center', va='bottom',
           fontsize=10, fontweight='bold', color='#2E7D32',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#F1F8E9', edgecolor='#2E7D32', alpha=0.9))

ax.set_yticks(y)
ax.set_yticklabels(categories, fontsize=12)
ax.set_xlabel('Metric Value', fontsize=12)
ax.set_xlim(min(min(alphaseek), min(without_iter)) * 0.7, max(max(alphaseek), max(without_iter)) * 1.2)
ax.set_ylim(-0.5, len(categories) - 0.5)
ax.set_title('Self-Iteration Impact: Dot Plot Comparison\n(Square: w/o Self-Iter, Diamond: AlphaSeek)', 
            fontsize=13, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', fontsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.invert_yaxis()  # 反转y轴，让第一个类别在顶部

plt.tight_layout()

# 保存
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_dotplot.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_dotplot.pdf', 
            bbox_inches='tight', facecolor='white')

print("分组点图已保存！")
print("文件: /root/lanyun-tmp/补充实验/figure/ablation_dotplot.png")
