"""
Local Factor Runner for RD-Agent (Non-Docker Version)

This module implements a local execution version of the factor runner that:
1. Integrates with QuantaAlpha's factor library
2. Uses QuantaAlpha's function library for factor calculation
3. Executes backtests directly in the conda environment
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import pandas as pd
import numpy as np
from pandarallel import pandarallel

pandarallel.initialize(verbose=0, nb_workers=4)

# Import RD-Agent components
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import cache_with_pickle
from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger

# Import QuantaAlpha loader
from rdagent.scenarios.qlib.utils.quantaalpha_loader import (
    QuantaAlphaFactorLibrary,
    QuantaAlphaFactorPool,
)

# Import qlib for local execution
try:
    from qlib import init
    from qlib.data import D
    from qlib.config import REG_CN
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("qlib not available. Factor runner will not function.")

# Import experiment classes
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.app.qlib_rd_loop.conf_v2 import FactorBasePropSettingV2 as FactorBasePropSetting


class LocalFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    Local factor runner that executes backtests directly without Docker
    
    Key features:
    1. Loads factors from QuantaAlpha library
    2. Calculates factor values using QuantaAlpha functions
    3. Runs backtests using qlib directly in the conda environment
    4. Returns standard metrics (IC, ARR, IR, MDD, Calmar)
    """
    
    def __init__(self, scen=None):
        super().__init__(scen)
        self.factor_pool = QuantaAlphaFactorPool()
        self.sota_factors: Optional[pd.DataFrame] = None
        
    def calculate_information_coefficient(
        self, 
        concat_feature: pd.DataFrame, 
        SOTA_feature_column_size: int, 
        new_feature_columns_size: int
    ) -> pd.Series:
        """
        Calculate IC between SOTA and new factors
        """
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size), dtype=float)
        
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                idx = col1 * new_feature_columns_size + col2 - SOTA_feature_column_size
                res.iloc[idx] = concat_feature.iloc[:, col1].corr(concat_feature.iloc[:, col2])
        
        return res
    
    def deduplicate_new_factors(
        self, 
        SOTA_feature: pd.DataFrame, 
        new_feature: pd.DataFrame,
        ic_threshold: float = 0.99
    ) -> pd.DataFrame:
        """
        Remove new factors that are highly correlated with SOTA factors
        """
        if SOTA_feature is None or SOTA_feature.empty:
            return new_feature
        
        if new_feature is None or new_feature.empty:
            return new_feature
        
        logger.info(f"Deduplicating factors: SOTA={SOTA_feature.shape[1]}, New={new_feature.shape[1]}")
        
        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        
        try:
            # Calculate IC using parallel apply
            IC_max = (
                concat_feature.groupby("datetime")
                .parallel_apply(
                    lambda x: self.calculate_information_coefficient(
                        x, SOTA_feature.shape[1], new_feature.shape[1]
                    )
                )
                .mean()
            )
        except Exception as e:
            logger.warning(f"Parallel apply failed: {e}, using regular apply")
            IC_max = (
                concat_feature.groupby("datetime")
                .apply(
                    lambda x: self.calculate_information_coefficient(
                        x, SOTA_feature.shape[1], new_feature.shape[1]
                    )
                )
                .mean()
            )
        
        # Reshape and find max IC for each new factor
        IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        IC_max = IC_max.unstack().max(axis=0)
        
        # Keep factors with IC < threshold
        keep_mask = IC_max < ic_threshold
        deduplicated = new_feature.iloc[:, keep_mask.values]
        
        removed_count = new_feature.shape[1] - deduplicated.shape[1]
        logger.info(f"Deduplication: Kept {deduplicated.shape[1]}, Removed {removed_count}")
        
        return deduplicated
    
    def run_local_backtest(
        self,
        features_df: pd.DataFrame,
        config: FactorBasePropSetting,
    ) -> Dict:
        """
        Run backtest locally using qlib
        
        Returns metrics dict with:
        - IC, ICIR, Rank IC, Rank ICIR
        - annualized_return
        - information_ratio
        - max_drawdown
        - calmar_ratio
        """
        if not QLIB_AVAILABLE:
            raise RuntimeError("qlib is not available for backtest")
        
        # Initialize qlib
        init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
        
        # Load label data
        stock_list = D.instruments(config.market)
        label_expr = "Ref($close, -2) / Ref($close, -1) - 1"
        label_df = D.features(
            stock_list,
            [label_expr],
            start_time=config.train_start,
            end_time=config.test_end,
            freq='day'
        )
        label_df.columns = ['LABEL0']
        
        # Align features and labels
        common_index = features_df.index.intersection(label_df.index)
        features_df = features_df.loc[common_index]
        label_df = label_df.loc[common_index]
        
        # Combine
        combined_df = pd.concat([features_df, label_df], axis=1)
        
        # CSRankNorm normalization (group by datetime for cross-sectional rank)
        feature_cols = list(features_df.columns)
        names = combined_df.index.names or [None, None]
        if 'datetime' in names:
            dt_level = 'datetime'
        else:
            lev0, lev1 = combined_df.index.get_level_values(0), combined_df.index.get_level_values(1)
            try:
                pd.to_datetime(lev0[:1])
                dt_level = 0  # level 0 is datetime
            except (TypeError, ValueError):
                dt_level = 1  # level 1 is datetime
        
        # Fillna and normalize
        for col in feature_cols:
            combined_df[col] = combined_df[col].fillna(0)
            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], 0)
            combined_df[col] = combined_df.groupby(level=dt_level)[col].transform(
                lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
            )
        
        # Drop na labels and normalize
        combined_df = combined_df.dropna(subset=['LABEL0'])
        combined_df['LABEL0'] = combined_df.groupby(level=dt_level)['LABEL0'].transform(
            lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
        )
        
        # Build multi-index columns
        feature_tuples = [('feature', col) for col in feature_cols]
        label_tuples = [('label', 'LABEL0')]
        combined_df.columns = pd.MultiIndex.from_tuples(feature_tuples + label_tuples)
        
        # Define segments
        segments = {
            'train': [config.train_start, config.train_end],
            'valid': [config.valid_start, config.valid_end],
            'test': [config.test_start, config.test_end],
        }
        
        # Split data
        def _get_datetime_level(idx):
            names = idx.names or [None, None]
            if 'datetime' in names:
                return idx.get_level_values('datetime')
            lev0, lev1 = idx.get_level_values(0), idx.get_level_values(1)
            try:
                pd.to_datetime(lev0[:1])
                return lev0
            except (TypeError, ValueError):
                return lev1

        def fetch_data(selector):
            if isinstance(selector, str):
                selector = segments.get(selector, selector)
            dates = _get_datetime_level(combined_df.index)
            mask = (dates >= pd.Timestamp(selector[0])) & (dates <= pd.Timestamp(selector[1]))
            return combined_df.loc[mask]
        
        train_data = fetch_data('train')
        valid_data = fetch_data('valid')
        test_data = fetch_data('test')
        
        X_train = train_data['feature']
        y_train = train_data['label']['LABEL0']
        X_valid = valid_data['feature'] if len(valid_data) > 0 else None
        y_valid = valid_data['label']['LABEL0'] if len(valid_data) > 0 else None
        X_test = test_data['feature']
        y_test = test_data['label']['LABEL0']
        
        # Train LGBM
        import lightgbm as lgb
        
        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=train_dataset) if X_valid is not None else None
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 210,
            'colsample_bytree': 0.8879,
            'subsample': 0.8789,
            'lambda_l1': 205.6999,
            'lambda_l2': 580.9768,
            'min_child_samples': 100,
            'feature_fraction_bynode': 0.8,
            'num_threads': 20,
            'seed': 42,
            'verbose': -1
        }
        
        model = lgb.train(
            lgb_params,
            train_dataset,
            num_boost_round=500,
            valid_sets=[valid_dataset] if valid_dataset else None,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )
        
        # Calculate IC metrics
        def calculate_ic_metrics(X, y, model, period_name):
            if X is None or len(X) == 0:
                return {}
            
            preds = model.predict(X, num_iteration=model.best_iteration)
            pred_df = pd.DataFrame({'pred': preds, 'label': y.values}, index=X.index)
            
            from scipy.stats import pearsonr, spearmanr
            
            ic_list = []
            rank_ic_list = []
            
            for date, group in pred_df.groupby(level='datetime'):
                if len(group) > 1:
                    ic, _ = pearsonr(group['pred'], group['label'])
                    rank_ic, _ = spearmanr(group['pred'], group['label'])
                    if not np.isnan(ic):
                        ic_list.append(ic)
                    if not np.isnan(rank_ic):
                        rank_ic_list.append(rank_ic)
            
            ic_mean = np.mean(ic_list) if ic_list else 0.0
            rank_ic_mean = np.mean(rank_ic_list) if rank_ic_list else 0.0
            icir = ic_mean / (np.std(ic_list) + 1e-12) if ic_list else 0.0
            rank_icir = rank_ic_mean / (np.std(rank_ic_list) + 1e-12) if rank_ic_list else 0.0
            
            return {
                'ic': ic_mean,
                'rank_ic': rank_ic_mean,
                'icir': icir,
                'rank_icir': rank_icir,
            }
        
        train_metrics = calculate_ic_metrics(X_train, y_train, model, "train")
        valid_metrics = calculate_ic_metrics(X_valid, y_valid, model, "valid") if X_valid is not None else {}
        test_metrics = calculate_ic_metrics(X_test, y_test, model, "test")
        
        # Run backtest
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred_series = pd.Series(test_pred, index=X_test.index)
        
        # Ensure index format: (datetime, instrument) for qlib backtest
        # Index level order may be (datetime, instrument) or (instrument, datetime)
        if isinstance(pred_series.index, pd.MultiIndex):
            names = pred_series.index.names or [None, None]
            if 'datetime' in names and 'instrument' in names:
                datetime_level = pred_series.index.get_level_values('datetime')
                instrument_level = pred_series.index.get_level_values('instrument')
            else:
                lev0, lev1 = pred_series.index.get_level_values(0), pred_series.index.get_level_values(1)
                # Detect: datetime parses as date, instrument is stock code (e.g. SH600000)
                try:
                    pd.to_datetime(lev0[:1])
                    datetime_level, instrument_level = lev0, lev1
                except (TypeError, ValueError):
                    datetime_level, instrument_level = lev1, lev0
            if not isinstance(datetime_level, pd.DatetimeIndex):
                datetime_level = pd.to_datetime(datetime_level)
            pred_series.index = pd.MultiIndex.from_arrays(
                [datetime_level, instrument_level],
                names=['datetime', 'instrument']
            )
            pred_series = pred_series.sort_index()
        
        # Backtest
        instruments = D.instruments(config.market)
        stock_list = D.list_instruments(
            instruments,
            start_time=config.test_start,
            end_time=config.test_end,
            as_list=True
        )
        
        portfolio_metric_dict, indicator_dict = qlib_backtest(
            executor={
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                    "verbose": False,
                }
            },
            strategy={
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {
                    "signal": pred_series,
                    "topk": config.topk,
                    "n_drop": config.n_drop,
                }
            },
            start_time=config.test_start,
            end_time=config.test_end,
            account=100000000.0,
            benchmark=config.benchmark,
            exchange_kwargs={
                "codes": stock_list,
                "limit_threshold": 0.095,
                "deal_price": "open",
                "open_cost": config.open_cost,
                "close_cost": config.close_cost,
                "min_cost": 5.0,
            }
        )
        
        # Extract metrics
        if portfolio_metric_dict and "1day" in portfolio_metric_dict:
            report_df, _ = portfolio_metric_dict["1day"]
            
            if isinstance(report_df, pd.DataFrame) and 'return' in report_df.columns:
                portfolio_return = report_df['return'].replace([np.inf, -np.inf], np.nan).fillna(0)
                bench_return = report_df['bench'].replace([np.inf, -np.inf], np.nan).fillna(0) if 'bench' in report_df.columns else 0
                cost = report_df['cost'].replace([np.inf, -np.inf], np.nan).fillna(0) if 'cost' in report_df.columns else 0
                
                excess_return_with_cost = portfolio_return - bench_return - cost
                excess_return_with_cost = excess_return_with_cost.dropna()
                
                if len(excess_return_with_cost) > 0:
                    analysis = risk_analysis(excess_return_with_cost)
                    
                    if isinstance(analysis, pd.DataFrame):
                        analysis = analysis['risk'] if 'risk' in analysis.columns else analysis.iloc[:, 0]
                    
                    ann_ret = float(analysis.get('annualized_return', 0))
                    info_ratio = float(analysis.get('information_ratio', 0))
                    max_dd = float(analysis.get('max_drawdown', 0))
                    
                    calmar = ann_ret / abs(max_dd) if max_dd != 0 and ann_ret != 0 else 0.0
                    
                    return {
                        'IC': test_metrics.get('ic', 0),
                        'ICIR': test_metrics.get('icir', 0),
                        'Rank IC': test_metrics.get('rank_ic', 0),
                        'Rank ICIR': test_metrics.get('rank_icir', 0),
                        'annualized_return': ann_ret,
                        'information_ratio': info_ratio,
                        'max_drawdown': max_dd,
                        'calmar_ratio': calmar,
                    }
        
        # Fallback if backtest fails
        return {
            'IC': test_metrics.get('ic', 0),
            'ICIR': test_metrics.get('icir', 0),
            'Rank IC': test_metrics.get('rank_ic', 0),
            'Rank ICIR': test_metrics.get('rank_icir', 0),
            'annualized_return': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
        }
    
    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        """
        Main development function for factor experiments
        
        This runs the complete factor evaluation pipeline:
        1. Load/calculate factors
        2. Deduplicate with SOTA
        3. Run backtest
        4. Return results
        """
        logger.info("=" * 60)
        logger.info("LocalFactorRunner: Starting factor evaluation")
        logger.info("=" * 60)
        
        config = FactorBasePropSetting()
        
        # Step 1: Get factors to evaluate
        if exp.based_experiments:
            # This is a subsequent round - use QuantaAlpha factor pool
            logger.info("Loading new factors from QuantaAlpha pool...")
            
            # Get SOTA factors from previous experiments
            sota_factor_exps = [
                be for be in exp.based_experiments 
                if isinstance(be, QlibFactorExperiment) and be.result is not None
            ]
            
            if sota_factor_exps:
                logger.info(f"Found {len(sota_factor_exps)} previous factor experiments")
                # Load SOTA factors
                self.sota_factors = self._load_sota_factors(sota_factor_exps)
            
            # Load new factors from QuantaAlpha
            new_factors = self.factor_pool.load_new_factors(ic_threshold=config.factor_ic_threshold)
            
            if not new_factors:
                raise FactorEmptyError("No new factors available from QuantaAlpha pool")
            
            # Calculate new factor values
            logger.info(f"Calculating {len(new_factors)} new factor values...")
            library = QuantaAlphaFactorLibrary()
            library.raw_data = library.load_market_data(
                market=config.market,
                start_time=config.train_start,
                end_time=config.test_end
            )
            
            new_factor_values = []
            for factor_id, factor in list(new_factors.items())[:config.max_factors_per_round]:
                values = library.calculate_factor_values(factor, library.raw_data)
                if values is not None:
                    new_factor_values.append(values)
                    # Mark as evaluated
                    self.factor_pool.mark_evaluated([factor_id], accepted=False)
            
            if not new_factor_values:
                raise FactorEmptyError("Failed to calculate any new factor values")
            
            new_factors_df = pd.concat(new_factor_values, axis=1)
            
            # Deduplicate
            if self.sota_factors is not None and not self.sota_factors.empty:
                new_factors_df = self.deduplicate_new_factors(
                    self.sota_factors, 
                    new_factors_df,
                    ic_threshold=0.99
                )
            
            if new_factors_df.empty:
                raise FactorEmptyError("All new factors are too similar to SOTA factors")
            
            # Combine with SOTA
            if self.sota_factors is not None and not self.sota_factors.empty:
                combined_factors = pd.concat([self.sota_factors, new_factors_df], axis=1)
            else:
                combined_factors = new_factors_df
            
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            
        else:
            # First round - initialize with top QuantaAlpha factors
            logger.info("First round: Loading top QuantaAlpha factors...")
            
            library = QuantaAlphaFactorLibrary()
            library.load_factors(ic_threshold=config.factor_ic_threshold)
            library.load_market_data(
                market=config.market,
                start_time=config.train_start,
                end_time=config.test_end
            )
            
            # Get top factors
            top_factors = library.get_top_factors(n=config.initial_factor_count, metric="ic")
            
            logger.info(f"Selected top {len(top_factors)} factors by IC")
            
            # Calculate values
            factor_values = []
            for factor in top_factors:
                values = library.calculate_factor_values(factor, library.raw_data)
                if values is not None:
                    factor_values.append(values)
            
            if not factor_values:
                raise FactorEmptyError("Failed to calculate any factor values")
            
            combined_factors = pd.concat(factor_values, axis=1)
        
        # Step 2: Run backtest
        logger.info(f"Running backtest with {combined_factors.shape[1]} factors...")
        
        try:
            result = self.run_local_backtest(combined_factors, config)
            
            logger.info("Backtest completed successfully")
            logger.info(f"  IC: {result.get('IC', 0):.4f}")
            logger.info(f"  ARR: {result.get('annualized_return', 0):.4f}")
            logger.info(f"  IR: {result.get('information_ratio', 0):.4f}")
            logger.info(f"  MDD: {result.get('max_drawdown', 0):.4f}")
            logger.info(f"  Calmar: {result.get('calmar_ratio', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            raise FactorEmptyError(f"Backtest execution failed: {e}")
        
        # Store results
        exp.result = result

        # 保存 combined_factors 供 model runner 加载 SOTA 因子
        factor_path = Path(exp.experiment_workspace.workspace_path) / "combined_factors_df.parquet"
        combined_factors.to_parquet(factor_path, engine="pyarrow")
        logger.info(f"Saved combined factors to {factor_path}")

        # Save factor state
        self.factor_pool.save_state(
            Path(exp.experiment_workspace.workspace_path) / "factor_pool_state.json"
        )

        logger.info("=" * 60)
        logger.info("LocalFactorRunner: Factor evaluation completed")
        logger.info("=" * 60)
        
        return exp
    
    def _load_sota_factors(self, factor_experiments: List[QlibFactorExperiment]) -> Optional[pd.DataFrame]:
        """
        Load SOTA factors from previous experiments
        """
        # Find the best experiment (highest IC or ARR)
        best_exp = None
        best_score = -float('inf')
        
        for exp in factor_experiments:
            if exp.result:
                score = exp.result.get('IC', 0) + exp.result.get('annualized_return', 0)
                if score > best_score:
                    best_score = score
                    best_exp = exp
        
        if best_exp is None:
            return None
        
        # Try to load factors from workspace
        factor_path = Path(best_exp.experiment_workspace.workspace_path) / "combined_factors_df.parquet"
        if factor_path.exists():
            return pd.read_parquet(factor_path)
        
        return None


# Compatibility alias
QlibFactorRunner = LocalFactorRunner
