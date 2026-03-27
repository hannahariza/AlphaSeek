"""
补充实验：AER vs 波动率散点图（含成本和IR）
整合论文/参考图中的基线数据与本项目的回测数据，生成性能对比图。

运行: cd /root/lanyun-tmp/补充实验 && python plot_model_performance.py
输出: ./figure/model_performance_aer_volatility.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figure"
PROJECT_ROOT = SCRIPT_DIR.parent
BACKTEST_JSON = PROJECT_ROOT / "RD-Agent" / "result" / "backtest_result" / "backtest_with_quantaalpha_factors.json"

# 论文/参考图中的基线数据 (AER%, Volatility%, IR, Cost$) —— 已移除 Ours (Gemini/GPT4.1/Deepseek-v3)
BASELINE_DATA = [
    ("ToT", 9.8, 9.1, 0.98, 32.1, "blue"),
    ("MCTS", 9.6, 10.1, 0.96, 23.3, "purple"),
    ("CoT", 8.7, 9.4, 0.87, 24.3, "orange"),
    ("AlphaGen", 6.7, 8.6, 0.67, 42.4, "cyan"),
    ("GP", 7.5, 10.1, 0.76, 87.0, "green"),
    ("AlphaForge", 7.1, 9.9, 0.71, 1.3, "pink"),
    ("DSO", 5.5, 10.4, 0.56, 26.8, "gold"),
]


def load_project_backtest(backtest_path: Path, experiment_cost: float = None) -> dict | None:
    """
    从项目回测结果加载 AER、IR，并推算波动率。
    若未提供 experiment_cost，则根据 Accumulated Cost 或使用默认值。
    """
    if not backtest_path.exists():
        return None
    with open(backtest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", data)
    ann_ret = metrics.get("annualized_return") or metrics.get(
        "1day.excess_return_with_cost.annualized_return", 0
    )
    ir = metrics.get("information_ratio") or metrics.get(
        "1day.excess_return_with_cost.information_ratio", 0
    )
    if ann_ret is None or ir is None or ir <= 0:
        return None
    aer_pct = float(ann_ret) * 100
    vol_pct = aer_pct / float(ir) if ir > 0 else 0
    cost = float(experiment_cost) if experiment_cost is not None else 0.03  # 默认 $0.03
    return {
        "name": "AlphaSeek (Ours)",
        "aer": aer_pct,
        "volatility": vol_pct,
        "ir": float(ir),
        "cost": cost,
        "color": "darkviolet",
    }


def plot_model_performance(
    project_data: dict | None = None,
    experiment_cost: float = None,
    output_path: Path = None,
):
    """绘制 AER vs 波动率 气泡图，气泡大小表示成本，标注 IR 和 Cost。"""
    output_path = output_path or (FIGURE_DIR / "model_performance_aer_volatility.png")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 合并数据
    rows = []
    for item in BASELINE_DATA:
        name, aer, vol, ir, cost, color = item
        rows.append({"name": name, "aer": aer, "volatility": vol, "ir": ir, "cost": cost, "color": color})

    if project_data:
        rows.append(project_data)

    names = [r["name"] for r in rows]
    aer = np.array([r["aer"] for r in rows])
    vol = np.array([r["volatility"] for r in rows])
    ir_vals = np.array([r["ir"] for r in rows])
    costs = np.array([r["cost"] for r in rows])
    colors = [r["color"] for r in rows]

    # 气泡大小：以成本为尺度，与论文图例($10/$25/$50)大致对应
    area_min, area_max = 50, 500
    cost_min, cost_max = costs.min(), costs.max()
    if cost_max - cost_min < 1e-6:
        sizes = np.ones(len(costs)) * 150
    else:
        sizes = area_min + (costs - cost_min) / (cost_max - cost_min) * (area_max - area_min)

    # 扩展波动率范围以容纳项目数据（如 RD-Agent 可能 < 8）
    vol_plot_min = min(vol.min() - 0.5, 5)
    vol_plot_max = max(vol.max() + 0.5, 11)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制 IR 等值线: IR = AER / Vol => AER = IR * Vol
    vol_line = np.linspace(vol_plot_min, vol_plot_max, 100)
    for ir_level, ls in [(1.1, "--"), (0.9, "--"), (0.7, "--")]:
        aer_line = ir_level * vol_line
        ax.plot(vol_line, aer_line, color="gray", linestyle=ls, linewidth=1, alpha=0.7)
        # 在合适位置标注
        idx = min(60, len(vol_line) - 1)
        ax.annotate(f"IR = {ir_level}", (vol_line[idx], aer_line[idx]), fontsize=9, color="gray", alpha=0.8)

    # 散点（气泡）
    for i in range(len(rows)):
        ax.scatter(
            vol[i], aer[i], s=sizes[i], c=colors[i], alpha=0.6, edgecolors="black", linewidth=0.5
        )
        # 标签：模型名 + IR | Cost
        label = f"{names[i]}\nIR: {ir_vals[i]:.2f} | Cost: ${costs[i]:.1f}"
        ax.annotate(
            label,
            (vol[i], aer[i]),
            fontsize=8,
            ha="left" if vol[i] < 10 else "right",
            va="center",
            alpha=0.9,
        )

    ax.set_xlabel("Annualized Volatility % ↓", fontsize=12)
    ax.set_ylabel("AER (Annualized Excess Return) % ↑", fontsize=12)
    ax.set_title("Model Performance: AER vs Volatility with Cost and IR", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")

    # 图例：气泡大小对应成本（左上角，手动绘制，等距排列不重叠）
    # 大小保留：小/中/大 对应 $10/$25/$50
    cost_legend_items = [("$10", 40), ("$25", 80), ("$50", 140)]
    # 在左上角数据区域放置，y 方向等距：每个示例间隔 2.2 单位
    leg_x = vol_plot_min + 0.35
    leg_spacing = 0.5  # 等距间隔，确保大小气泡不重叠
    # 从高到低排列 $50/$25/$10，中心点等距
    leg_y_centers = [aer.max() + 1.6 - i * leg_spacing for i in range(3)]
    for i, (lbl, sz) in enumerate(cost_legend_items):
        y_pos = leg_y_centers[i]
        ax.scatter(leg_x, y_pos, s=sz, c="gray", alpha=0.6, edgecolors="black", linewidth=0.5, zorder=10)
        ax.annotate(lbl, (leg_x, y_pos), fontsize=10, ha="left", va="center", xytext=(10, 0), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="绘制 AER vs 波动率 补充实验图")
    parser.add_argument("--backtest", type=str, default=str(BACKTEST_JSON), help="回测结果 JSON 路径")
    parser.add_argument("--cost", type=float, default=None, help="本实验的 API/计算成本 (美元)，如 0.03")
    parser.add_argument("--no-project", action="store_true", help="不叠加本项目数据，仅绘制基线")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    args = parser.parse_args()

    project_data = None
    if not args.no_project:
        project_data = load_project_backtest(
            Path(args.backtest),
            experiment_cost=args.cost,
        )
        if project_data:
            print(f"已加载项目数据: AER={project_data['aer']:.2f}%, IR={project_data['ir']:.2f}, Cost=${project_data['cost']:.2f}")
        else:
            print("未找到有效回测数据，仅绘制基线")

    output_path = Path(args.output) if args.output else None
    plot_model_performance(project_data=project_data, experiment_cost=args.cost, output_path=output_path)


if __name__ == "__main__":
    main()
