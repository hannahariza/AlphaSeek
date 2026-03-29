"""
补充实验：IC / ARR / IR / MDD 四指标柱状图对比
从论文表格提取 LightGBM、Transformer、Alpha158(20)、Alpha158、RD-Agent(Deepseek-V3.2)、AlphaSeek 数据，
仿照参考图绘制 2x2 柱状图，每个指标按值从小到大排列。

运行: cd /root/lanyun-tmp/补充实验 && python plot_metrics_comparison.py
输出: ./figure/metrics_comparison_bar_charts.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figure"

# 数据：IC, ARR%, IR, MDD%
# MDD (Maximum Drawdown) 值取自论文表格，正值表示回撤幅度
TABLE_DATA = [
    ("LightGBM", 0.0247, 0.07, 0.0092, 21.80),
    ("Transformer", 0.0331, 5.21, 0.4502, 13.81),
    ("Alpha158(20)", 0.0051, 4.63, 0.5044, 22.19),
    ("Alpha158", 0.0131, 2.66, 0.4099, 10.15),
    ("RD-Agent", 0.0401, 7.81, 0.8202, 18.03),
    ("AlphaSeek", 0.0447, 8.28, 1.29, 6.28),
]


def plot_metrics_comparison(output_path: Path = None):
    """绘制 2x2 四指标柱状图，每个子图按指标值从小到大排列。"""
    output_path = output_path or (FIGURE_DIR / "metrics_comparison_bar_charts.png")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    names = [r[0] for r in TABLE_DATA]
    ic_vals = np.array([r[1] for r in TABLE_DATA])
    arr_vals = np.array([r[2] for r in TABLE_DATA])
    ir_vals = np.array([r[3] for r in TABLE_DATA])
    mdd_vals = np.array([r[4] for r in TABLE_DATA])  # MDD (Maximum Drawdown)

    metrics = [
        ("IC", ic_vals, "Information Coefficient"),
        ("ARR (%)", arr_vals, "Annualized Return"),
        ("IR (SHR*)", ir_vals, "Information Ratio"),
        ("MDD (%)$\\downarrow$", mdd_vals, "Maximum Drawdown (lower is better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 颜色：前4个用蓝色系，RD-Agent用绿色，AlphaSeek用红色突出
    color_map = {
        "LightGBM": "#1f77b4",
        "Transformer": "#1f77b4",
        "Alpha158(20)": "#1f77b4",
        "Alpha158": "#1f77b4",
        "RD-Agent": "#2ca02c",  # 绿色
        "AlphaSeek": "#d62728",  # 红色突出显示
    }

    for ax, (metric_name, values, ylabel) in zip(axes.flat, metrics):
        # 按值从小到大排序
        order = np.argsort(values)
        sorted_names = [names[i] for i in order]
        sorted_vals = values[order]
        sorted_colors = [color_map.get(n, "#888") for n in sorted_names]

        x = np.arange(len(sorted_names))
        bars = ax.bar(x, sorted_vals, color=sorted_colors, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(sorted_names, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Average {metric_name}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    # plt.suptitle("Factor Mining Methods: IC / ARR / ICIR / IR Comparison (Sorted Ascending)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {output_path}")


if __name__ == "__main__":
    plot_metrics_comparison()
