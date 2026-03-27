"""
QuantaAlpha Factor Loader for RD-Agent

This module provides integration with QuantaAlpha's factor library:
1. Load factors from QuantaAlpha's JSON output
2. Calculate factor values using QuantaAlpha's function library
3. Integrate with RD-Agent's factor processing pipeline
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Add QuantaAlpha to path
QUANTAALPHA_PATH = Path("/root/lanyun-tmp/QuantaAlpha-main")
if str(QUANTAALPHA_PATH) not in sys.path:
    sys.path.insert(0, str(QUANTAALPHA_PATH))


@dataclass
class QuantaAlphaFactor:
    """Represents a single QuantaAlpha factor"""
    factor_id: str
    factor_name: str
    factor_expression: str
    factor_description: str
    backtest_results: Dict
    metadata: Dict
    
    @property
    def ic(self) -> float:
        """Get IC from backtest results"""
        return self.backtest_results.get("IC", 0.0)
    
    @property
    def rank_ic(self) -> float:
        """Get Rank IC from backtest results"""
        return self.backtest_results.get("Rank IC", 0.0)
    
    @property
    def arr(self) -> float:
        """Get annualized return"""
        return self.backtest_results.get("1day.excess_return_with_cost.annualized_return", 0.0)
    
    @property
    def ir(self) -> float:
        """Get information ratio"""
        return self.backtest_results.get("1day.excess_return_with_cost.information_ratio", 0.0)


class QuantaAlphaFactorLibrary:
    """
    Manages QuantaAlpha factor library loading and processing
    """
    
    def __init__(self, factor_lib_path: Optional[str] = None):
        """
        Initialize factor library
        
        Args:
            factor_lib_path: Path to QuantaAlpha factor library JSON file
        """
        self.factor_lib_path = factor_lib_path or self._find_latest_factor_lib()
        self.factors: Dict[str, QuantaAlphaFactor] = {}
        self.raw_data: Optional[pd.DataFrame] = None
        
    def _find_all_factor_lib_files(self) -> List[Path]:
        """
        查找 factorlib 下所有以 all_factors_library 为前缀的因子文件
        包括: all_factors_library.json, all_factors_library_*.json
        """
        factor_lib_dir = Path("/root/lanyun-tmp/QuantaAlpha-main/data/factorlib")
        if not factor_lib_dir.exists():
            raise FileNotFoundError(f"Factor library directory not found: {factor_lib_dir}")

        # 匹配 all_factors_library.json 和 all_factors_library_*.json
        files = []
        for p in factor_lib_dir.iterdir():
            if p.suffix == ".json" and p.stem.startswith("all_factors_library"):
                files.append(p)
        if not files:
            raise FileNotFoundError(f"No factor library files (all_factors_library*.json) found in {factor_lib_dir}")
        return sorted(files, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)

    def _find_latest_factor_lib(self) -> str:
        """兼容旧接口：返回最新文件的路径"""
        files = self._find_all_factor_lib_files()
        return str(files[0])
    
    def load_factors(self, ic_threshold: float = 0.0) -> Dict[str, QuantaAlphaFactor]:
        """
        从 factorlib 下所有 all_factors_library*.json 文件加载并合并因子
        
        Args:
            ic_threshold: Minimum IC to include factor (0.0 to include all)
        
        Returns:
            Dictionary of factor_id -> QuantaAlphaFactor (去重后，同 factor_id 保留首次出现)
        """
        all_files = self._find_all_factor_lib_files()
        self.factors = {}
        seen_ids = set()

        for file_path in all_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            factors_dict = data.get('factors', {})

            for factor_id, factor_data in factors_dict.items():
                if factor_id in seen_ids:
                    continue
                factor = QuantaAlphaFactor(
                    factor_id=factor_id,
                    factor_name=factor_data.get('factor_name', factor_id),
                    factor_expression=factor_data.get('factor_expression', ''),
                    factor_description=factor_data.get('factor_description', ''),
                    backtest_results=factor_data.get('backtest_results', {}),
                    metadata=factor_data.get('metadata', {})
                )
                if factor.ic >= ic_threshold:
                    self.factors[factor_id] = factor
                    seen_ids.add(factor_id)

        self.factor_lib_path = str(all_files[0])  # 兼容旧逻辑，指向最新文件
        print(f"Loaded {len(self.factors)} factors from {len(all_files)} files: {[p.name for p in all_files]}")
        print(f"  IC threshold: {ic_threshold}")
        return self.factors
    
    def load_market_data(
        self,
        market: str = "csi300",
        start_time: str = "2016-01-01",
        end_time: str = "2025-12-26"
    ) -> pd.DataFrame:
        """
        Load market data using QuantaAlpha's data loader
        """
        try:
            from qlib import init
            from qlib.data import D
            from qlib.config import REG_CN
            
            # Initialize qlib
            init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
            
            # Load data
            fields = ["$open", "$high", "$low", "$close", "$volume"]
            self.raw_data = D.features(
                D.instruments(market),
                fields,
                start_time=start_time,
                end_time=end_time,
                freq="day"
            )
            
            print(f"Loaded market data: {self.raw_data.shape}")
            return self.raw_data
            
        except Exception as e:
            print(f"Error loading market data: {e}")
            raise
    
    def calculate_factor_values(
        self,
        factor: QuantaAlphaFactor,
        data: Optional[pd.DataFrame] = None
    ) -> Optional[pd.Series]:
        """
        Calculate factor values using QuantaAlpha's function library.
        
        Uses parse_symbol + parse_expression from QuantaAlpha to convert
        $var syntax to valid Python before eval (raw $high/$low etc. cause invalid syntax).
        """
        if data is None:
            data = self.raw_data
        
        if data is None:
            raise ValueError("No market data loaded. Call load_market_data() first.")
        
        try:
            from quantaalpha.factors.coder.expr_parser import parse_symbol, parse_expression
            from quantaalpha.factors.coder import function_lib as fl
            
            # Ensure $return exists (not a raw qlib field)
            if "$return" not in data.columns:
                ret = data["$close"].groupby(level="instrument", group_keys=False).pct_change(fill_method=None)
                data = data.copy()
                data["$return"] = ret
            
            columns = list(data.columns)
            
            # Step 1: parse_symbol ($high -> high, $return -> return, etc.)
            expr = parse_symbol(factor.factor_expression, columns)
            # Step 2: parse_expression (arithmetic -> ADD/SUBTRACT/etc.)
            expr = parse_expression(expr)
            # Step 3: replace symbol names with data['$col']
            for col in sorted(columns, key=lambda c: -len(str(c))):
                sym = col.replace("$", "")
                expr = expr.replace(sym, f"data['{col}']")
            
            env = {
                "data": data,
                "np": np,
                "pd": pd,
                "ADD": fl.ADD,
                "SUBTRACT": fl.SUBTRACT,
                "MULTIPLY": fl.MULTIPLY,
                "DIVIDE": fl.DIVIDE,
                "GT": fl.GT,
                "LT": fl.LT,
                "GE": fl.GE,
                "LE": fl.LE,
                "EQ": fl.EQ,
                "NE": fl.NE,
                "AND": fl.AND,
                "OR": fl.OR,
                "WHERE": fl.WHERE,
                "TS_ZSCORE": fl.TS_ZSCORE,
                "TS_MEAN": fl.TS_MEAN,
                "TS_STD": fl.TS_STD,
                "TS_SUM": fl.TS_SUM,
                "TS_MAX": fl.TS_MAX,
                "TS_MIN": fl.TS_MIN,
                "TS_CORR": fl.TS_CORR,
                "TS_QUANTILE": fl.TS_QUANTILE,
                "TS_PCTCHANGE": fl.TS_PCTCHANGE,
                "RANK": fl.RANK,
                "ZSCORE": fl.ZSCORE,
                "EMA": fl.EMA,
                "DELTA": fl.DELTA,
                "DELAY": fl.DELAY,
                "ABS": fl.ABS,
                "LOG": fl.LOG,
                "SIGN": fl.SIGN,
                "COUNT": fl.COUNT,
            }
            
            result = eval(expr, env)
            
            if isinstance(result, pd.DataFrame):
                result = result.iloc[:, 0]
            
            result.name = factor.factor_name
            return result
            
        except Exception as e:
            print(f"Error calculating factor {factor.factor_name}: {e}")
            return None
    
    def calculate_all_factors(
        self,
        max_factors: Optional[int] = None,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate all factor values
        
        Args:
            max_factors: Maximum number of factors to calculate (None for all)
            data: Market data (uses self.raw_data if not provided)
        
        Returns:
            DataFrame with factor values (columns = factors, index = data index)
        """
        if data is None:
            data = self.raw_data
        
        if not self.factors:
            self.load_factors()
        
        factor_list = []
        success_count = 0
        failed_factors = []
        
        factors_to_calc = list(self.factors.values())
        if max_factors is not None:
            factors_to_calc = factors_to_calc[:max_factors]
        
        print(f"\nCalculating {len(factors_to_calc)} factors...")
        
        for i, factor in enumerate(factors_to_calc):
            result = self.calculate_factor_values(factor, data)
            
            if result is not None:
                factor_list.append(result)
                success_count += 1
            else:
                failed_factors.append(factor.factor_name)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(factors_to_calc)}, Success: {success_count}")
        
        print(f"\nFactor calculation complete:")
        print(f"  Success: {success_count}/{len(factors_to_calc)}")
        print(f"  Failed: {len(failed_factors)}")
        
        if failed_factors and len(failed_factors) <= 10:
            print(f"  Failed factors: {failed_factors}")
        
        if factor_list:
            return pd.concat(factor_list, axis=1)
        else:
            raise ValueError("No factors were successfully calculated")
    
    def get_factor_by_name(self, name: str) -> Optional[QuantaAlphaFactor]:
        """Get factor by name"""
        for factor in self.factors.values():
            if factor.factor_name == name:
                return factor
        return None
    
    def get_top_factors(self, n: int = 10, metric: str = "ic") -> List[QuantaAlphaFactor]:
        """
        Get top N factors by specified metric
        
        Args:
            n: Number of factors to return
            metric: Metric to sort by ("ic", "rank_ic", "arr", "ir")
        """
        if not self.factors:
            self.load_factors()
        
        factors_list = list(self.factors.values())
        
        # Sort by specified metric
        if metric == "ic":
            factors_list.sort(key=lambda f: f.ic, reverse=True)
        elif metric == "rank_ic":
            factors_list.sort(key=lambda f: f.rank_ic, reverse=True)
        elif metric == "arr":
            factors_list.sort(key=lambda f: f.arr, reverse=True)
        elif metric == "ir":
            factors_list.sort(key=lambda f: f.ir, reverse=True)
        
        return factors_list[:n]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert factors to summary DataFrame"""
        if not self.factors:
            self.load_factors()
        
        data = []
        for factor in self.factors.values():
            data.append({
                "factor_id": factor.factor_id,
                "factor_name": factor.factor_name,
                "ic": factor.ic,
                "rank_ic": factor.rank_ic,
                "arr": factor.arr,
                "ir": factor.ir,
                "expression": factor.factor_expression[:100] + "..." if len(factor.factor_expression) > 100 else factor.factor_expression,
            })
        
        return pd.DataFrame(data)


class QuantaAlphaFactorPool:
    """
    Manages a pool of QuantaAlpha factors for incremental loading
    """
    
    def __init__(self, pool_path: Optional[str] = None):
        self.pool_path = pool_path or "/root/lanyun-tmp/QuantaAlpha-main/data/factorlib"
        self.library = QuantaAlphaFactorLibrary()
        self.evaluated_factors: set = set()  # Set of factor_ids that have been evaluated
        self.sota_factor_ids: set = set()  # Set of factor_ids in SOTA
        
    def load_new_factors(self, ic_threshold: float = 0.01) -> Dict[str, QuantaAlphaFactor]:
        """
        Load new factors that haven't been evaluated yet
        
        Returns:
            Dictionary of new factors
        """
        all_factors = self.library.load_factors(ic_threshold=ic_threshold)
        
        # Filter out already evaluated factors
        new_factors = {
            fid: factor for fid, factor in all_factors.items()
            if fid not in self.evaluated_factors
        }
        
        print(f"New factors to evaluate: {len(new_factors)}")
        print(f"Already evaluated: {len(self.evaluated_factors)}")
        
        return new_factors
    
    def mark_evaluated(self, factor_ids: List[str], accepted: bool = False) -> None:
        """
        Mark factors as evaluated
        
        Args:
            factor_ids: List of factor IDs
            accepted: Whether these factors were accepted into SOTA
        """
        for fid in factor_ids:
            self.evaluated_factors.add(fid)
            if accepted:
                self.sota_factor_ids.add(fid)
    
    def get_sota_factors(self) -> Dict[str, QuantaAlphaFactor]:
        """Get current SOTA factors"""
        if not self.library.factors:
            self.library.load_factors()
        
        return {
            fid: factor for fid, factor in self.library.factors.items()
            if fid in self.sota_factor_ids
        }
    
    def save_state(self, path: str) -> None:
        """Save pool state to file"""
        state = {
            "evaluated_factors": list(self.evaluated_factors),
            "sota_factor_ids": list(self.sota_factor_ids),
            "library_path": self.library.factor_lib_path,
        }
        with open(path, 'w') as f:
            json.dump(state, f)
    
    def load_state(self, path: str) -> None:
        """Load pool state from file"""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.evaluated_factors = set(state.get("evaluated_factors", []))
        self.sota_factor_ids = set(state.get("sota_factor_ids", []))
        
        if "library_path" in state:
            self.library.factor_lib_path = state["library_path"]


# Convenience function for RD-Agent integration
def load_quantaalpha_factors(
    ic_threshold: float = 0.01,
    max_factors: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and calculate QuantaAlpha factors
    
    Args:
        ic_threshold: Minimum IC threshold
        max_factors: Maximum number of factors
    
    Returns:
        Tuple of (factor_values_df, factor_info_df)
    """
    library = QuantaAlphaFactorLibrary()
    library.load_factors(ic_threshold=ic_threshold)
    library.load_market_data()
    
    factor_values = library.calculate_all_factors(max_factors=max_factors)
    factor_info = library.to_dataframe()
    
    return factor_values, factor_info
