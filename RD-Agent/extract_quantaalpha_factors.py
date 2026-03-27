#!/usr/bin/env python3
"""
提取QuantaAlpha因子库信息，用于RD-Agent分析
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_quantaalpha_factors(factor_lib_path: str) -> Dict[str, Any]:
    """加载QuantaAlpha因子库"""
    with open(factor_lib_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_factor_summary(factors_data: Dict) -> List[Dict]:
    """提取因子摘要信息"""
    summary = []
    
    for factor_id, factor_info in factors_data.get('factors', {}).items():
        backtest = factor_info.get('backtest_results', {})
        
        summary.append({
            'factor_id': factor_id,
            'factor_name': factor_info.get('factor_name', ''),
            'expression': factor_info.get('factor_expression', ''),
            'ic': backtest.get('IC', 0),
            'icir': backtest.get('ICIR', 0),
            'rank_ic': backtest.get('Rank IC', 0),
            'rank_icir': backtest.get('Rank ICIR', 0),
            'annual_return': backtest.get('1day.excess_return_without_cost.annualized_return', 0),
            'info_ratio': backtest.get('1day.excess_return_without_cost.information_ratio', 0),
            'max_drawdown': backtest.get('1day.excess_return_without_cost.max_drawdown', 0),
        })
    
    return summary


def get_high_ic_factors(summary: List[Dict], min_ic: float = 0.01) -> List[Dict]:
    """获取高IC因子"""
    return [f for f in summary if abs(f['ic']) >= min_ic]


def format_for_rdagent(factor_info: Dict) -> str:
    """将因子格式化为RD-Agent可以理解的描述"""
    return f"""
因子名称: {factor_info['factor_name']}
因子表达式: {factor_info['expression']}
IC: {factor_info['ic']:.6f}
ICIR: {factor_info['icir']:.6f}
Rank IC: {factor_info['rank_ic']:.6f}
年化收益: {factor_info['annual_return']:.4f}
信息比率: {factor_info['info_ratio']:.4f}
最大回撤: {factor_info['max_drawdown']:.4f}
"""


def main():
    # 因子库路径
    factor_lib_path = "/root/lanyun-tmp/QuantaAlpha-main/data/factorlib/all_factors_library.json"
    
    print("=" * 60)
    print("QuantaAlpha 因子库分析报告")
    print("=" * 60)
    
    # 加载因子库
    print(f"\n正在加载因子库: {factor_lib_path}")
    factors_data = load_quantaalpha_factors(factor_lib_path)
    
    # 基本信息
    metadata = factors_data.get('metadata', {})
    print(f"\n因子库信息:")
    print(f"  - 总因子数: {metadata.get('total_factors', 0)}")
    print(f"  - 版本: {metadata.get('version', 'unknown')}")
    print(f"  - 创建时间: {metadata.get('created_at', 'unknown')}")
    print(f"  - 最后更新: {metadata.get('last_updated', 'unknown')}")
    
    # 提取摘要
    summary = extract_factor_summary(factors_data)
    
    # 统计信息
    print(f"\n因子IC统计:")
    ics = [f['ic'] for f in summary]
    rank_ics = [f['rank_ic'] for f in summary]
    print(f"  - IC均值: {sum(ics)/len(ics):.6f}")
    print(f"  - IC最大值: {max(ics):.6f}")
    print(f"  - IC最小值: {min(ics):.6f}")
    print(f"  - Rank IC均值: {sum(rank_ics)/len(rank_ics):.6f}")
    
    # 高IC因子
    high_ic_factors = get_high_ic_factors(summary, min_ic=0.01)
    print(f"\n高IC因子 (|IC| >= 0.01): {len(high_ic_factors)} 个")
    
    # 显示前5个高IC因子
    print("\n前5个高IC因子详情:")
    print("-" * 60)
    for i, factor in enumerate(high_ic_factors[:5], 1):
        print(f"\n{i}. {factor['factor_name']} (ID: {factor['factor_id'][:16]}...)")
        print(f"   表达式: {factor['expression'][:60]}...")
        print(f"   IC: {factor['ic']:.6f} | Rank IC: {factor['rank_ic']:.6f}")
        print(f"   年化收益: {factor['annual_return']:.4f} | 信息比率: {factor['info_ratio']:.4f}")
    
    # 保存摘要到文件
    output_path = Path("/root/lanyun-tmp/RD-Agent/quantaalpha_factor_summary.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': metadata,
            'factors': summary,
            'high_ic_factors': high_ic_factors
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n摘要已保存到: {output_path}")
    
    # 生成RD-Agent可用的因子描述文件
    rdagent_input = []
    for factor in summary[:20]:  # 取前20个因子
        rdagent_input.append({
            'name': factor['factor_name'],
            'expression': factor['expression'],
            'performance': {
                'ic': factor['ic'],
                'rank_ic': factor['rank_ic'],
                'annual_return': factor['annual_return']
            }
        })
    
    rdagent_path = Path("/root/lanyun-tmp/RD-Agent/rdagent_factor_input.json")
    with open(rdagent_path, 'w', encoding='utf-8') as f:
        json.dump(rdagent_input, f, ensure_ascii=False, indent=2)
    
    print(f"RD-Agent输入文件已保存到: {rdagent_path}")
    print("\n" + "=" * 60)
    print("分析完成! 可以开始使用RD-Agent进行因子组合优化。")
    print("=" * 60)


if __name__ == "__main__":
    main()
