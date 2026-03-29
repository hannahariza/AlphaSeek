import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据 - 展示相对于消融方法的提升
categories = ['IC', 'Rank IC', 'ARR\n(%)', 'MDD\n(improvement)']
baseline = [0.0381, 0.0367, 3.01, -10.29]  # 消融方法作为基准
gains = [0.0073, 0.0070, 5.27, 4.01]  # 提升量（MDD为降低量）
final = [0.0454, 0.0437, 8.28, -6.28]  # AlphaSeek最终值

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 颜色
color_baseline = '#B8B8B8'
color_gain = '#C44536'
color_final = '#2E7D32'

x = np.arange(len(categories))
width = 0.6

# 绘制瀑布效果
for i, (cat, base, gain, fin) in enumerate(zip(categories, baseline, gains, final)):
    # 基线柱（消融方法）
    ax.bar(i - 0.25, abs(base), width*0.45, bottom=min(0, base), 
           color=color_baseline, alpha=0.6, edgecolor='white', label='w/o Self-Iter' if i == 0 else '')
    
    # 提升柱
    if i == 3:  # MDD
        ax.bar(i, gain, width*0.45, bottom=abs(base), 
               color=color_gain, alpha=0.9, edgecolor='white', label='Improvement' if i == 0 else '')
        # 最终值标记
        ax.scatter(i + 0.25, abs(fin), s=150, c=color_final, marker='*', zorder=5)
    else:
        ax.bar(i, gain, width*0.45, bottom=abs(base), 
               color=color_gain, alpha=0.9, edgecolor='white', hatch='///')
        ax.scatter(i + 0.25, abs(fin), s=150, c=color_final, marker='*', zorder=5)
    
    # 添加数值标签
    ax.text(i - 0.25, abs(base)/2, f'{abs(base):.4f}' if i < 2 else f'{abs(base):.2f}', 
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # 提升百分比
    pct = (gain / abs(base)) * 100
    ax.text(i, abs(base) + gain/2, f'+{pct:.1f}%', 
            ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # 最终值标签
    ax.text(i + 0.25, abs(fin) + 0.005 if i < 2 else abs(fin) + 0.3, 
            f'{abs(fin):.4f}' if i < 2 else f'{abs(fin):.2f}★', 
            ha='center', va='bottom', fontsize=10, color=color_final, fontweight='bold')

# 添加连接线显示累积
for i in range(len(categories)):
    ax.plot([i - 0.02, i + 0.23], [abs(baseline[i]) + gains[i]]*2, 
            'k--', alpha=0.3, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Self-Iteration Contribution: Waterfall View\n(Gray: Baseline, Red: Contribution, Green Star: Final)', 
            fontsize=13, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

# 保存
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_waterfall.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_waterfall.pdf', 
            bbox_inches='tight', facecolor='white')

print("瀑布图已保存！")
print("文件: /root/lanyun-tmp/补充实验/figure/ablation_waterfall.png")
