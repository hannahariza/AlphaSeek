# 时序累积收益对比图 - 实验指南

要生成类似论文中的CSI 500时序累积收益图，需要以下数据：

## 需要的数据文件

| 模型/策略 | 文件名 | 状态 | 获取方式 |
|-----------|--------|------|----------|
| **QuantaAlpha** | `all_factors_library_total_cumulative_excess.csv` | ✅ 已有 | 已完成回测 |
| **RD-Agent** | `rd_agent_cumulative_excess.csv` | ❌ 需生成 | 需修改回测脚本 |
| **Alpha158** | `alpha158_cumulative_excess.csv` | ❌ 需生成 | 运行基准回测 |
| **Alpha360** | `alpha360_cumulative_excess.csv` | ❌ 需生成 | 运行基准回测 |
| **AlphaAgent** | `alphaagent_cumulative_excess.csv` | ❌ 需获取 | 需要该模型数据 |

## 实验步骤

### 1. 获取 Alpha158/Alpha360 基准数据

修改 `QuantaAlpha-main/configs/backtest.yaml`：

```yaml
factor_source:
  type: "alpha158"  # 或 "alpha360"
```

运行回测：
```bash
cd /root/lanyun-tmp/QuantaAlpha-main
python -m quantaalpha.backtest.run_backtest --config configs/backtest.yaml
```

将生成的 `cumulative_excess.csv` 复制到补充实验数据目录：
```bash
cp data/results/backtest_v2_results/XXX_cumulative_excess.csv \
   /root/lanyun-tmp/补充实验/data/alpha158_cumulative_excess.csv
```

### 2. 获取 RD-Agent 时序数据

RD-Agent 当前只保存汇总指标，需要修改回测脚本保存日度收益。

编辑 `run_local_backtest_v6.py`，在回测完成后添加保存代码：

```python
# 在 qlib_backtest 调用后，提取 report_df 并保存
if portfolio_metric_dict and "1day" in portfolio_metric_dict:
    report_df, positions_df = portfolio_metric_dict["1day"]
    
    # 保存日度收益数据
    if isinstance(report_df, pd.DataFrame):
        report_df.to_csv("/root/lanyun-tmp/补充实验/data/rd_agent_returns.csv")
        
        # 计算并保存累积收益
        daily_returns = report_df['return'].replace([np.inf, -np.inf], np.nan).fillna(0)
        bench_returns = report_df['bench'].replace([np.inf, -np.inf], np.nan).fillna(0) if 'bench' in report_df.columns else 0
        costs = report_df['cost'].replace([np.inf, -np.inf], np.nan).fillna(0) if 'cost' in report_df.columns else 0
        
        excess_returns = daily_returns - bench_returns - costs
        cumulative = (1 + excess_returns).cumprod() - 1
        
        result_df = pd.DataFrame({
            'date': report_df.index,
            'daily_excess_return': excess_returns.values,
            'cumulative_excess_return': cumulative.values
        })
        result_df.to_csv("/root/lanyun-tmp/补充实验/data/rd_agent_cumulative_excess.csv", index=False)
```

### 3. 绘图

数据准备完成后，运行：

```bash
cd /root/lanyun-tmp/补充实验
python plot_cumulative_returns.py
```

## 数据结构要求

CSV文件需要包含以下列：

```csv
date,daily_excess_return,cumulative_excess_return
2022-01-04,0.000671,0.000671
2022-01-05,0.012531,0.013202
...
```

支持的列名变体：
- 日期：`date`, `datetime`
- 日度收益：`daily_excess_return`, `daily_return`, `return`
- 累积收益：`cumulative_excess_return`, `cumulative_return`, `cum_return`

## 当前可用数据

当前已有 **QuantaAlpha** 数据：
- 文件：`/root/lanyun-tmp/QuantaAlpha-main/data/results/backtest_v2_results/all_factors_library_total_cumulative_excess.csv`
- 时间范围：2022-01-04 ~ 2025-12-26
- 数据行数：约 1000 行（日度数据）

可以先用这份数据测试绘图脚本。
