"""
补充实验：IC / ARR / IR / MDD 四指标柱状图对比
从论文表格提取数据，仿照参考图绘制 2x2 柱状图，样式美观。

运行: cd /root/lanyun-tmp/补充实验 && python plot_metrics_comparison.py
输出: ./figure/metrics_comparison_bar_charts.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figure"

# 完整数据：IC, RankIC, ARR(%), IR, MDD
# 来源：Alpha158, MLP, Transformer, RD-Agent, Alpha360, AlphaSeek
TABLE_DATA = [
    ("Alpha158", 0.0131, 0.0334, 2.66, 0.4099, 10.15),
    ("MLP", 0.0321, 0.0438, 1.46, 0.1716, 18.15),
    ("Transformer", 0.0331, 0.0451, 5.21, 0.4502, 13.81),
    ("RD-Agent", 0.0401, 0.0522, 7.81, 0.8202, 18.03),
    ("Alpha360", 0.0105, 0.0306, 4.09, 0.6009, 11.52),  # 替换RDAgent为Alpha360
    ("AlphaSeek", 0.0447, 0.0431, 8.28, 1.29, 6.28),
]


def plot_metrics_comparison(output_path: Path = None):
    """绘制 2x2 四指标柱状图，参考上传图片样式。"""
    output_path = output_path or (FIGURE_DIR / "metrics_comparison_bar_charts.png")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    names = [r[0] for r in TABLE_DATA]
    ic_vals = np.array([r[1] for r in TABLE_DATA])
    arr_vals = np.array([r[3] for r in TABLE_DATA])
    ir_vals = np.array([r[4] for r in TABLE_DATA])
    mdd_vals = np.array([r[5] for r in TABLE_DATA])

    # 设置图形样式
    plt.rcParams['font.family'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 颜色方案 - 学术风格
    color_alpha158 = "#5B8DB8"      # 柔和蓝色 - Alpha158
    color_alpha360 = "#6B9BC8"      # 中蓝色 - Alpha360
    color_deep_learning = "#8BB0D8" # 浅蓝 - 深度学习 (MLP/Transformer)
    color_rdagent = "#7BA37D"        # 柔和绿色 - RD-Agent
    color_alphaseek = "#C44536"      # 红色 - AlphaSeek

    color_map = {
        "Alpha158": color_alpha158,
        "Alpha360": color_alpha360,
        "MLP": color_deep_learning,
        "Transformer": color_deep_learning,
        "RD-Agent": color_rdagent,
        "AlphaSeek": color_alphaseek,
    }

    # 定义四个子图的数据和标题
    subplots_config = [
        ("IC", ic_vals, "Information Coefficient", False),
        ("ARR (%)", arr_vals, "Annualized Return(%)", False),
        ("IR (SHR*)", ir_vals, "Information Ratio", False),
        ("MDD (%)", mdd_vals, "Maximum Drawdown(%) (lower is better)", True),
    ]

    for ax, (metric_name, values, ylabel, descending) in zip(axes.flat, subplots_config):
        # 排序
        if descending:
            order = np.argsort(values)[::-1]
        else:
            order = np.argsort(values)

        sorted_names = [names[i] for i in order]
        sorted_vals = values[order]
        sorted_colors = [color_map.get(n, "#888") for n in sorted_names]

        x = np.arange(len(sorted_names))
        bars = ax.bar(x, sorted_vals, color=sorted_colors, 
                      edgecolor="white", linewidth=0.8, width=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(sorted_names, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Average {metric_name}", fontsize=12, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis="y", linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # 添加总标题
    # plt.suptitle("Factor Mining Methods: Multi-Metric Comparison", 
    #             fontsize=14, fontweight='bold', y=0.98)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_alpha158, edgecolor='white', label='Alpha158'),
        Patch(facecolor=color_alpha360, edgecolor='white', label='Alpha360'),
        Patch(facecolor=color_deep_learning, edgecolor='white', label='Deep Learning (MLP/Transformer)'),
        Patch(facecolor=color_rdagent, edgecolor='white', label='RD-Agent'),
        Patch(facecolor=color_alphaseek, edgecolor='white', label='AlphaSeek (Ours)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=True, fancybox=False, edgecolor='gray', fontsize=8)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"图表已保存: {output_path}")


if __name__ == "__main__":
    plot_metrics_comparison()
