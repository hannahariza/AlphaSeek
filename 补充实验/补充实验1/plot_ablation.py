import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
methods = ['AlphaSeek', 'w/o Consistency', 'w/o Complexity', 'w/o Redundancy Controls']
ic_values = [0.0454, 0.0447, 0.0449, 0.0438]
rank_ic_values = [0.0437, 0.0431, 0.0433, 0.0431]
arr_values = [8.28, 2.99, 4.58, 1.37]
mdd_values = [6.28, 12.72, 11.63, 18.38]

# 颜色方案 - 仿照原图的红色系
colors = ['#C44536', '#D47F76', '#D49891', '#D4B4B0']  # 深红到浅红
hatches = ['///', '///', '///', '///']  # 斜线纹理

# 创建图形 (1行2列)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图: IC 和 Rank IC
ax1 = axes[0]
x = np.arange(2)  # 两个类别: IC, Rank IC
width = 0.18

# 绘制4组柱状图
for i in range(4):
    values = [ic_values[i], rank_ic_values[i]]
    offset = (i - 1.5) * width
    bars = ax1.bar(x + offset, values, width, label=methods[i], 
                   color=colors[i], edgecolor='white', linewidth=0.5)
    # 添加斜线纹理
    for bar in bars:
        bar.set_hatch(hatches[i])
    # 添加数值标签
    for j, (xi, vi) in enumerate(zip(x + offset, values)):
        if i == 0:  # AlphaSeek (主方法) 显示实际值
            ax1.text(xi, vi + 0.0003, f'{vi:.4f}', ha='center', va='bottom', 
                    fontsize=7, fontweight='bold', color='#C44536')
        else:  # 其他显示相对主方法的下降
            drop = vi - (ic_values[0] if j == 0 else rank_ic_values[0])
            ax1.text(xi, vi + 0.0003, f'{drop:.4f}', ha='center', va='bottom', 
                    fontsize=7, color='#666666')

ax1.set_xticks(x)
ax1.set_xticklabels(['IC', 'Rank IC'], fontsize=11)
# 调整y轴范围以放大差距
ax1.set_ylim(0.042, 0.047)
ax1.set_yticks([0.042, 0.043, 0.044, 0.045, 0.046, 0.047])
ax1.set_title('IC↑ and Rank IC↑', fontsize=13, fontweight='bold', pad=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_axisbelow(True)

# 右图: ARR 和 MDD (分开的两个子区域)
ax2 = axes[1]

# 使用两个独立的x位置组
width = 0.18
x_arr = np.array([0])  # ARR位置
x_mdd = np.array([1])  # MDD位置

# 绘制ARR部分
for i in range(4):
    offset = (i - 1.5) * width
    bars = ax2.bar(x_arr + offset, [arr_values[i]], width, 
                   color=colors[i], edgecolor='white', linewidth=0.5)
    for bar in bars:
        bar.set_hatch(hatches[i])
    # 添加数值标签
    vi = arr_values[i]
    xi = x_arr[0] + offset
    if i == 0:
        ax2.text(xi, vi + 0.6, f'{vi:.2f}', ha='center', va='bottom', 
                fontsize=7, fontweight='bold', color='#C44536')
    else:
        drop = vi - arr_values[0]
        ax2.text(xi, vi + 0.6, f'{drop:.2f}', ha='center', va='bottom', 
                fontsize=7, color='#666666')

# 绘制MDD部分
for i in range(4):
    offset = (i - 1.5) * width
    bars = ax2.bar(x_mdd + offset, [mdd_values[i]], width, 
                   color=colors[i], edgecolor='white', linewidth=0.5)
    for bar in bars:
        bar.set_hatch(hatches[i])
    # 添加数值标签
    vi = mdd_values[i]
    xi = x_mdd[0] + offset
    if i == 0:
        ax2.text(xi, vi + 0.6, f'{vi:.2f}', ha='center', va='bottom', 
                fontsize=7, fontweight='bold', color='#C44536')
    else:
        change = vi - mdd_values[0]
        ax2.text(xi, vi + 0.6, f'+{change:.2f}', ha='center', va='bottom', 
                fontsize=7, color='#666666')

ax2.set_xticks([0, 1])
ax2.set_xticklabels(['ARR (%)↑', 'MDD (%)↓'], fontsize=11)
ax2.set_ylim(0, 21)
ax2.set_title('ARR (%)↑ and MDD (%)↓', fontsize=13, fontweight='bold', pad=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_axisbelow(True)

# 添加图例 (放在左上角)
legend_labels = ['AlphaSeek', 'w/o Consistency', 'w/o Complexity', 'w/o Redundancy']
ax1.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
           frameon=True, fancybox=False, edgecolor='gray', fontsize=9)

plt.tight_layout()

# 保存图片
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_study.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_study.pdf', 
            bbox_inches='tight', facecolor='white')

print("图表已保存到 /root/lanyun-tmp/补充实验/figure/ablation_study.png")
print("PDF版本已保存到 /root/lanyun-tmp/补充实验/figure/ablation_study.pdf")
