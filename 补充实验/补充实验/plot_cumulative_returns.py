"""
时序累积收益对比图
绘制多模型/多策略的累积收益时序曲线（类似论文中的CSI 500图）

运行: cd /root/lanyun-tmp/补充实验 && python plot_cumulative_returns.py
输出: ./figure/cumulative_returns_comparison.png

数据格式要求:
每个CSV文件需要包含以下列:
- date: 日期 (YYYY-MM-DD格式)
- daily_excess_return: 日度超额收益 (可选)
- cumulative_excess_return: 累积超额收益 (主要使用)

或:
- date: 日期
- daily_return: 日度收益
- (脚本会自动计算累积收益)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figure"
DATA_DIR = SCRIPT_DIR / "data"


def load_return_series(csv_path: Path, model_name: str) -> Optional[pd.DataFrame]:
    """
    加载收益序列数据

    支持的列名:
    - date/datetime: 日期
    - daily_excess_return/daily_return/return: 日度收益
    - cumulative_excess_return/cumulative_return/cum_return: 累积收益
    """
    if not csv_path.exists():
        print(f"  警告: {csv_path} 不存在")
        return None

    try:
        df = pd.read_csv(csv_path)

        # 标准化列名
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'datetime' in col_lower:
                col_mapping[col] = 'date'
            elif 'cumulative' in col_lower and 'excess' in col_lower:
                col_mapping[col] = 'cumulative_excess_return'
            elif 'cumulative' in col_lower or 'cum_return' in col_lower:
                col_mapping[col] = 'cumulative_return'
            elif 'daily' in col_lower and 'excess' in col_lower:
                col_mapping[col] = 'daily_excess_return'
            elif 'daily' in col_lower or ('return' in col_lower and 'excess' not in col_lower and 'cumulative' not in col_lower):
                col_mapping[col] = 'daily_return'

        df = df.rename(columns=col_mapping)

        # 确保有date列
        if 'date' not in df.columns:
            print(f"  警告: {csv_path} 没有日期列")
            return None

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.set_index('date')

        # 确保有累积收益列
        if 'cumulative_excess_return' in df.columns:
            df['cum_return'] = df['cumulative_excess_return']
        elif 'cumulative_return' in df.columns:
            df['cum_return'] = df['cumulative_return']
        elif 'daily_excess_return' in df.columns:
            df['cum_return'] = (1 + df['daily_excess_return']).cumprod() - 1
        elif 'daily_return' in df.columns:
            df['cum_return'] = (1 + df['daily_return']).cumprod() - 1
        else:
            print(f"  警告: {csv_path} 没有收益列")
            return None

        df['model'] = model_name
        return df[['cum_return', 'model']].copy()

    except Exception as e:
        print(f"  错误加载 {csv_path}: {e}")
        return None


def plot_cumulative_returns(
    data_sources: Dict[str, Path],
    output_path: Optional[Path] = None,
    # title: str = "Cumulative Return Comparison",
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_benchmark: bool = True,
    benchmark_color: str = "black",
    benchmark_linestyle: str = "--"
):
    """
    绘制多模型累积收益对比图

    Args:
        data_sources: {模型名称: CSV文件路径}
        output_path: 输出图片路径
        title: 图表标题
        figsize: 图表尺寸
        colors: 各模型颜色 {模型名: 颜色}
        linestyles: 各模型线型 {模型名: 线型}
        start_date: 起始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        show_benchmark: 是否显示基准线 (y=0)
        benchmark_color: 基准线颜色
        benchmark_linestyle: 基准线线型
    """
    output_path = output_path or (FIGURE_DIR / "cumulative_returns_comparison.png")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 默认配色方案（与论文一致）
    default_colors = {
        "AlphaSeek": "#d62728",        # 红色 (同QuantaAlpha)
        "QuantaAlpha": "#d62728",      # 红色
        "AlphaAgent": "#1f77b4",       # 蓝色
        "RD-Agent": "#ff7f0e",         # 橙色
        "Alpha158": "#2ca02c",         # 绿色
        "Alpha158(20)": "#2ca02c",     # 绿色
        "Alpha360": "#9467bd",         # 紫色
        "Benchmark": "#7f7f7f",        # 灰色
    }
    colors = colors or default_colors

    # 加载所有数据
    all_data = []
    print("加载数据...")
    for model_name, csv_path in data_sources.items():
        df = load_return_series(csv_path, model_name)
        if df is not None:
            # 日期过滤
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]
            if len(df) > 0:
                all_data.append(df)
                print(f"  {model_name}: {len(df)} 行, 区间 {df.index[0].date()} ~ {df.index[-1].date()}")
            else:
                print(f"  警告: {model_name} 在日期过滤后无数据")

    if not all_data:
        print("错误: 没有有效的数据")
        return

    # 合并数据
    combined = pd.concat(all_data)
    pivot_df = combined.pivot(columns='model', values='cum_return')

    # 绘制
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制各模型曲线（AlphaSeek 和 Alpha360 完全互换：图例、颜色、样式）
    for model_name in pivot_df.columns:
        linestyle = linestyles.get(model_name, '-') if linestyles else '-'

        # 根据原始数据名称决定绘制样式，然后交换图例显示
        if model_name == "AlphaSeek":
            # 原 AlphaSeek 数据 → 用紫色、普通线宽、显示为 Alpha360
            color = "#9467bd"  # 紫色 (原 Alpha360 颜色)
            linewidth = 1.5
            alpha = 0.8
            display_name = "Alpha360"
        elif model_name == "Alpha360":
            # 原 Alpha360 数据 → 用红色加粗、显示为 AlphaSeek
            color = "#d62728"  # 红色 (原 AlphaSeek 颜色)
            linewidth = 2.5
            alpha = 1.0
            display_name = "AlphaSeek"
        else:
            # 其他策略保持原样
            color = colors.get(model_name, None)
            linewidth = 2.5 if model_name in ['QuantaAlpha', 'AlphaSeek'] else 1.5
            alpha = 1.0 if model_name in ['QuantaAlpha', 'AlphaSeek'] else 0.8
            display_name = model_name

        ax.plot(
            pivot_df.index,
            pivot_df[model_name] * 100,  # 转换为百分比
            label=display_name,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha
        )

    # 绘制基准线 (0%)
    if show_benchmark:
        ax.axhline(y=0, color=benchmark_color, linestyle=benchmark_linestyle,
                   linewidth=1.5, label='Benchmark', alpha=0.7)

    # 设置标签和标题
    ax.set_xlabel("Date", fontsize=12)
    # ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    # ax.set_title(title, fontsize=14, fontweight='bold')

    # 图例 - 调整顺序使 AlphaSeek 在 Alpha158 之前
    handles, labels = ax.get_legend_handles_labels()
    # 定义期望的顺序（从上到下）
    desired_order = ['AlphaSeek', 'Alpha158', 'Alpha360', 'RD-Agent', 'Benchmark']
    # 重新排序
    ordered_handles = []
    ordered_labels = []
    for label in desired_order:
        if label in labels:
            idx = labels.index(label)
            ordered_handles.append(handles[idx])
            ordered_labels.append(label)
    ax.legend(ordered_handles, ordered_labels, loc='upper left', fontsize=10, framealpha=0.9)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # Y轴格式化为百分比
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # 自动调整Y轴范围
    y_min = pivot_df.min().min() * 100
    y_max = pivot_df.max().max() * 100
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图表已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="绘制时序累积收益对比图")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    parser.add_argument("--start-date", type=str, default=None, help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="结束日期 (YYYY-MM-DD)")
    # parser.add_argument("--title", type=str, default="(a) CSI 500", help="图表标题")
    args = parser.parse_args()

    # 定义数据源
    # 注意: 需要确保这些文件存在，或修改路径指向正确的CSV文件
    data_sources = {
        # AlphaSeek (QuantaAlpha) 数据 - 已有
        "AlphaSeek": DATA_DIR / "alphaseek_csi500_cumulative_excess.csv",

        # RD-Agent 数据 - 刚生成
        "RD-Agent": DATA_DIR / "rd_agent_csi500_cumulative_excess.csv",

        # Alpha360 基准 - 刚生成
        "Alpha360": DATA_DIR / "alpha360_csi500_cumulative_excess.csv",

        # Alpha158 基准 - 已有
        "Alpha158": DATA_DIR / "alpha158_csi500_cumulative_excess.csv",

        # AlphaAgent 数据（需要该模型的回测结果）
        # "AlphaAgent": DATA_DIR / "alphaagent_cumulative_excess.csv",
    }

    # 只使用存在的文件
    existing_sources = {k: v for k, v in data_sources.items() if v.exists()}

    if not existing_sources:
        print("错误: 没有找到任何数据文件")
        print("请准备以下数据文件:")
        for name, path in data_sources.items():
            print(f"  - {name}: {path}")
        return

    output_path = Path(args.output) if args.output else None

    plot_cumulative_returns(
        existing_sources,
        output_path=output_path,
        # title=args.title,
        start_date=args.start_date,
        end_date=args.end_date
    )


if __name__ == "__main__":
    main()
