import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
categories = ['IC', 'Rank IC', 'ARR', 'MDD (inverse)']
alphaseek_raw = [0.0454, 0.0437, 8.28, 6.28]
without_iter_raw = [0.0381, 0.0367, 3.01, 10.29]

# 归一化到0-1范围用于雷达图（MDD反向处理，越小越好）
def normalize_metrics(alpha, without):
    # 前三个指标越大越好，MDD需要反向
    alpha_norm = []
    without_norm = []
    for i in range(4):
        if i == 3:  # MDD反向
            max_val = max(15 - alpha[i], 15 - without[i])
            alpha_norm.append((15 - alpha[i]) / 15)
            without_norm.append((15 - without[i]) / 15)
        else:
            max_val = max(alpha[i], without[i])
            alpha_norm.append(alpha[i] / max_val if max_val > 0 else 0)
            without_norm.append(without[i] / max_val if max_val > 0 else 0)
    return alpha_norm, without_norm

alphaseek, without_iter = normalize_metrics(alphaseek_raw, without_iter_raw)

# 雷达图设置
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
alphaseek += alphaseek[:1]
without_iter += without_iter[:1]
angles += angles[:1]

# 创建图形
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制AlphaSeek
ax.plot(angles, alphaseek, 'o-', linewidth=2.5, label='AlphaSeek (Ours)', color='#C44536')
ax.fill(angles, alphaseek, alpha=0.25, color='#C44536')

# 绘制消融方法
ax.plot(angles, without_iter, 's--', linewidth=2, label='w/o Self-Iteration', color='#999999')
ax.fill(angles, without_iter, alpha=0.15, color='#999999')

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1.1)

# 添加标题和图例
# ax.set_title('Self-Iteration Impact: Multi-dimensional Comparison', 
#             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, 
         fancybox=False, edgecolor='gray', fontsize=10)

plt.tight_layout()

# 保存
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_radar.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/root/lanyun-tmp/补充实验/figure/ablation_radar.pdf', 
            bbox_inches='tight', facecolor='white')

print("雷达图已保存！")
print("文件: /root/lanyun-tmp/补充实验/figure/ablation_radar.png")
