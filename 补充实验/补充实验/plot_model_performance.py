"""
补充实验：ARR vs 波动率散点图（IR斜率，MDD气泡大小）
根据表格数据绘制模型性能对比图。

运行: cd /root/lanyun-tmp/补充实验 && python plot_model_performance.py
输出: ./figure/model_performance_arr_volatility.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figure"

# 表格数据: (模型名称, ARR%, IR, MDD%, 类别, 颜色)
# 波动率 = ARR / IR (根据 IR = ARR / Volatility 推算)
MODEL_DATA = [
    # Machine Learning
    ("LightGBM", 0.07, 0.0092, 21.80, "ML", "#4a90d9"),
    ("MLP", 1.46, 0.1716, 18.15, "ML", "#6ab0f7"),
    # Deep Learning
    ("Transformer", 5.21, 0.4502, 13.81, "DL", "#ff7f0e"),
    ("LSTM", 6.01, 0.6802, 14.81, "DL", "#ffbb78"),
    # Factor Libraries
    ("Alpha158(20)", 4.63, 0.5044, 22.19, "Factor Lib", "#2ca02c"),
    ("Alpha158", 2.66, 0.4099, 10.15, "Factor Lib", "#98df8a"),
    ("Alpha360", 4.09, 0.6009, 11.52, "Factor Lib", "#55a868"),
    # LLM-based Agentic Factor Mining
    ("RD-Agent", 7.81, 0.8202, 18.03, "LLM-Agent", "#9467bd"),
    ("AlphaSeek", 8.28, 1.29, 6.28, "LLM-Agent", "#d62728"),  # 红色突出显示
]


def calculate_volatility(arr: float, ir: float) -> float:
    """根据 IR = ARR / Volatility 计算波动率"""
    if abs(ir) < 1e-6:
        return float('inf')
    return abs(arr / ir)


def plot_model_performance(output_path: Path = None):
    """绘制 ARR vs 波动率 气泡图，气泡大小表示MDD，标注 IR。"""
    output_path = output_path or (FIGURE_DIR / "model_performance_arr_volatility.png")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 解析数据并计算波动率
    rows = []
    for item in MODEL_DATA:
        name, arr, ir, mdd, category, color = item
        volatility = calculate_volatility(arr, ir)
        rows.append({
            "name": name,
            "arr": arr,
            "ir": ir,
            "mdd": mdd,
            "volatility": volatility,
            "category": category,
            "color": color
        })

    names = [r["name"] for r in rows]
    arr_vals = np.array([r["arr"] for r in rows])
    vol_vals = np.array([r["volatility"] for r in rows])
    ir_vals = np.array([r["ir"] for r in rows])
    mdd_vals = np.array([r["mdd"] for r in rows])
    colors = [r["color"] for r in rows]

    # 过滤掉波动率为无穷大的数据点
    valid_mask = np.isfinite(vol_vals)
    print(f"有效数据点: {np.sum(valid_mask)}/{len(rows)}")

    # 气泡大小：以MDD为尺度（MDD越大，气泡越大）
    area_min, area_max = 100, 800
    mdd_min, mdd_max = mdd_vals.min(), mdd_vals.max()
    sizes = area_min + (mdd_vals - mdd_min) / (mdd_max - mdd_min) * (area_max - area_min)

    # 设置坐标轴范围
    finite_vol = vol_vals[valid_mask]
    vol_plot_min = max(finite_vol.min() - 2, -1)
    vol_plot_max = 12  # 固定横坐标最大值为12
    arr_plot_min = min(arr_vals.min() - 2, 0)
    arr_plot_max = max(arr_vals.max() + 2, 10)

    fig, ax = plt.subplots(figsize=(12, 9))

    # 绘制 IR 等值线
    vol_line = np.linspace(max(0, vol_plot_min), vol_plot_max, 100)
    for ir_level, ls in [(1.3, "-"), (1.0, "--"), (0.7, "--"), (0.4, "--"), (0.1, ":")]:
        aer_line = ir_level * vol_line
        mask = (aer_line >= arr_plot_min) & (aer_line <= arr_plot_max)
        if np.any(mask):
            ax.plot(vol_line[mask], aer_line[mask], color="gray", linestyle=ls, linewidth=1, alpha=0.6)
            idx = np.where(mask)[0][len(np.where(mask)[0]) // 2]
            ax.annotate(f"IR={ir_level}", (vol_line[idx], aer_line[idx]),
                       fontsize=8, color="gray", alpha=0.8, ha="center")

    # 绘制有效数据点
    for i in range(len(rows)):
        if valid_mask[i]:
            edge_width = 2.5 if names[i] == "AlphaSeek" else 0.5
            edge_color = "darkred" if names[i] == "AlphaSeek" else "black"
            z_order = 15 if names[i] == "AlphaSeek" else 5
            ax.scatter(
                vol_vals[i], arr_vals[i],
                s=sizes[i], c=colors[i], alpha=0.7,
                edgecolors=edge_color,
                linewidth=edge_width,
                zorder=z_order
            )

    # 处理IR接近0或负的点 - 放在图表左边缘
    for i in range(len(rows)):
        if not valid_mask[i]:
            x_pos = vol_plot_min + 0.5
            ax.scatter(
                x_pos, arr_vals[i],
                s=sizes[i], c=colors[i], alpha=0.7,
                edgecolors="black", linewidth=0.5,
                marker="s",
                zorder=5
            )
            label = f"{names[i]}\nIR: {ir_vals[i]:.3f}"
            ax.annotate(
                label,
                (x_pos, arr_vals[i]),
                fontsize=8,
                ha="left",
                va="center",
                alpha=0.9,
                xytext=(15, 0),
                textcoords="offset points"
            )

    # 标注有效点的标签
    for i in range(len(rows)):
        if valid_mask[i]:
            label = f"{names[i]}\nIR: {ir_vals[i]:.2f}"
            ha = "left" if vol_vals[i] < (vol_plot_max + vol_plot_min) / 2 else "right"
            xytext = (12, 0) if ha == "left" else (-12, 0)

            ax.annotate(
                label,
                (vol_vals[i], arr_vals[i]),
                fontsize=8,
                ha=ha,
                va="center",
                alpha=0.9,
                xytext=xytext,
                textcoords="offset points",
                fontweight="bold" if names[i] == "AlphaSeek" else "normal"
            )

    ax.set_xlabel("Annualized Volatility % ↓", fontsize=13)
    ax.set_ylabel("ARR (Annualized Return) % ↑", fontsize=13)
    ax.set_xlim(vol_plot_min, vol_plot_max)
    ax.set_ylim(arr_plot_min, arr_plot_max)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("auto")

    # 类别图例
    category_colors = {
        "ML": "#1f77b4",
        "DL": "#ff7f0e",
        "Factor Lib": "#2ca02c",
        "LLM-Agent": "#d62728"
    }
    category_labels = {
        "ML": "Machine Learning",
        "DL": "Deep Learning",
        "Factor Lib": "Factor Libraries",
        "LLM-Agent": "LLM-based Agentic"
    }

    legend_elements = [Patch(facecolor=color, edgecolor='black', label=category_labels[cat])
                      for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    # MDD 气泡大小图例（右上角）
    mdd_legend_vals = [10, 20, 30]
    leg_x = vol_plot_max - 1
    leg_y_start = arr_plot_max - 0.3  # 向上平移
    leg_spacing = 0.8  # 减小纵向间距

    ax.annotate("MDD ↓", (leg_x, leg_y_start + 1), fontsize=10, ha="center", fontweight="bold")

    for i, mdd_val in enumerate(mdd_legend_vals):
        leg_size = area_min + (mdd_val - mdd_min) / (mdd_max - mdd_min) * (area_max - area_min)
        y_pos = leg_y_start - i * leg_spacing
        ax.scatter(leg_x, y_pos, s=leg_size, c="gray", alpha=0.6,
                  edgecolors="black", linewidth=0.5, zorder=10)
        ax.annotate(f"{mdd_val}%", (leg_x, y_pos), fontsize=9,
                   ha="left", va="center", xytext=(20, 0), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"图表已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="绘制 ARR vs 波动率 补充实验图（基于表格数据）")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    plot_model_performance(output_path=output_path)


if __name__ == "__main__":
    main()
