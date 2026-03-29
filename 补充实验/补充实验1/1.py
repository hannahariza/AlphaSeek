"""
补充实验：IC / ARR / IR / MDD 四指标柱状图对比（学术风增强版）

运行: cd /root/lanyun-tmp/补充实验 && python plot_metrics_comparison.py
输出: ./figure/metrics_comparison_bar_charts.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figure"

# 方法名, IC, RankIC, ARR(%), IR, MDD(%)
TABLE_DATA = [
    ("Alpha158", 0.0131, 0.0334, 2.66, 0.4099, 10.15),
    ("MLP", 0.0321, 0.0438, 1.46, 0.1716, 18.15),
    ("Transformer", 0.0331, 0.0451, 5.21, 0.4502, 13.81),
    ("RD-Agent", 0.0401, 0.0522, 7.81, 0.8202, 18.03),
    ("Alpha360", 0.0105, 0.0306, 4.09, 0.6009, 11.52),
    ("AlphaSeek", 0.0447, 0.0431, 8.28, 1.29, 6.28),
]


def style_axis(ax):
    """统一坐标轴风格。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AFAFAF")
    ax.spines["bottom"].set_color("#AFAFAF")
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)

    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.22, color="#B8B8B8")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10, pad=4, colors="#2F2F2F")
    ax.tick_params(axis="y", labelsize=10, colors="#2F2F2F")


def plot_single_metric(ax, values, names, color_map, higher_better=True, ylabel=""):
    """绘制单个指标子图。"""
    values = np.array(values)

    # 统一按“从弱到强”排序
    if higher_better:
        order = np.argsort(values)
    else:
        order = np.argsort(values)[::-1]

    sorted_names = [names[i] for i in order]
    sorted_vals = values[order]
    sorted_colors = [color_map[n] for n in sorted_names]

    x = np.arange(len(sorted_names))
    bars = ax.bar(
        x,
        sorted_vals,
        color=sorted_colors,
        width=0.68,
        edgecolor="#F2F2F2",
        linewidth=1.0,
        zorder=3,
    )

    # 轻微突出 AlphaSeek
    for bar, name in zip(bars, sorted_names):
        if name == "AlphaSeek":
            bar.set_edgecolor("#7F4A45")
            bar.set_linewidth(1.6)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=11, color="#222222")

    ymax = sorted_vals.max()
    ax.set_ylim(0, ymax * 1.16)

    style_axis(ax)


def plot_metrics_comparison(output_path: Path = None):
    output_path = output_path or (FIGURE_DIR / "metrics_comparison_bar_charts.png")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    names = [r[0] for r in TABLE_DATA]
    ic_vals = [r[1] for r in TABLE_DATA]
    arr_vals = [r[3] for r in TABLE_DATA]
    ir_vals = [r[4] for r in TABLE_DATA]
    mdd_vals = [r[5] for r in TABLE_DATA]

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    # 中等饱和度、学术风配色
    color_map = {
        "Alpha158": "#6F8FAF",      # muted steel blue
        "Alpha360": "#89A6BE",      # soft blue
        "MLP": "#A7B7C6",           # cool gray-blue
        "Transformer": "#7FA3BF",   # dusty blue
        "RD-Agent": "#7FA08A",      # muted sage green
        "AlphaSeek": "#B97C74",     # softened brick red
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.patch.set_facecolor("white")

    plot_single_metric(axes[0, 0], ic_vals, names, color_map, higher_better=True, ylabel="IC")
    plot_single_metric(axes[0, 1], arr_vals, names, color_map, higher_better=True, ylabel="ARR (%)")
    plot_single_metric(axes[1, 0], ir_vals, names, color_map, higher_better=True, ylabel="IR")
    plot_single_metric(axes[1, 1], mdd_vals, names, color_map, higher_better=False, ylabel="MDD (%)")

    legend_elements = [
        Patch(facecolor=color_map["Alpha158"], edgecolor="white", label="Alpha158"),
        Patch(facecolor=color_map["Alpha360"], edgecolor="white", label="Alpha360"),
        Patch(facecolor=color_map["MLP"], edgecolor="white", label="MLP"),
        Patch(facecolor=color_map["Transformer"], edgecolor="white", label="Transformer"),
        Patch(facecolor=color_map["RD-Agent"], edgecolor="white", label="RD-Agent"),
        Patch(facecolor=color_map["AlphaSeek"], edgecolor="#7F4A45", label="AlphaSeek"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=6,
        frameon=False,
        fontsize=9.5,
        handlelength=1.5,
        columnspacing=1.4,
    )

    plt.tight_layout(rect=[0.03, 0.07, 0.97, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"图表已保存: {output_path}")


if __name__ == "__main__":
    plot_metrics_comparison()