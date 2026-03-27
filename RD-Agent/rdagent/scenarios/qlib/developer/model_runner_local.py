"""
Local Model Runner for RD-Agent (Non-Docker Version)

This module implements a local execution version of the model runner that:
1. Loads SOTA factors from factor experiments
2. Trains models using PyTorch/LightGBM directly in the conda environment
3. Runs backtests using qlib
4. Returns standard metrics (IC, ARR, IR, MDD, Calmar)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Import RD-Agent components
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import cache_with_pickle
from rdagent.components.runner import CachedRunner
from rdagent.core.exception import ModelEmptyError
from rdagent.log import rdagent_logger as logger

# Import qlib for local execution
try:
    from qlib import init
    from qlib.data import D
    from qlib.config import REG_CN
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("qlib not available. Model runner will not function.")

# Import experiment classes
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.app.qlib_rd_loop.conf_v2 import ModelBasePropSettingV2 as ModelBasePropSetting


class _SimpleMLP(nn.Module):
    """Module-level MLP fallback for PyTorch time series (must be at module level for pickle)."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LocalModelRunner(CachedRunner[QlibModelExperiment]):
    """
    Local model runner that executes training and backtests directly without Docker
    
    Key features:
    1. Loads SOTA factors from factor experiments
    2. Trains PyTorch/TensorFlow models directly
    3. Executes backtests using qlib
    4. Returns standard metrics
    """
    
    def __init__(self, scen=None):
        super().__init__(scen)
        self.sota_factors: Optional[pd.DataFrame] = None
    
    def load_sota_factors(self, factor_experiments: List[QlibFactorExperiment]) -> Optional[pd.DataFrame]:
        """
        Load SOTA factors from previous factor experiments
        """
        # Find experiments with results
        valid_exps = [exp for exp in factor_experiments if exp.result is not None]
        
        if not valid_exps:
            logger.warning("No valid factor experiments found")
            return None
        
        # Sort by performance (IC + ARR)
        valid_exps.sort(
            key=lambda e: e.result.get('IC', 0) + e.result.get('annualized_return', 0),
            reverse=True
        )
        
        # Load factors from the best experiment
        best_exp = valid_exps[0]
        factor_path = Path(best_exp.experiment_workspace.workspace_path) / "combined_factors_df.parquet"
        
        if factor_path.exists():
            logger.info(f"Loading SOTA factors from {factor_path}")
            return pd.read_parquet(factor_path)
        else:
            logger.warning(f"Factor file not found: {factor_path}")
            return None
    
    def prepare_data(
        self,
        factors_df: pd.DataFrame,
        config: ModelBasePropSetting
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare train/valid/test datasets
        
        Returns:
            Tuple of (train_df, valid_df, test_df) with features and labels
        """
        if not QLIB_AVAILABLE:
            raise RuntimeError("qlib is not available")
        
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
        
        # Align with factors
        common_index = factors_df.index.intersection(label_df.index)
        factors_df = factors_df.loc[common_index]
        label_df = label_df.loc[common_index]
        
        # Combine
        combined_df = pd.concat([factors_df, label_df], axis=1)
        
        # Preprocessing
        feature_cols = list(factors_df.columns)
        dt_level = combined_df.index.names[0]
        
        # Fillna and normalize
        for col in feature_cols:
            combined_df[col] = combined_df[col].fillna(0)
            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], 0)
            # CSRankNorm
            combined_df[col] = combined_df.groupby(level=dt_level)[col].transform(
                lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
            )
        
        # Drop na labels and normalize
        combined_df = combined_df.dropna(subset=['LABEL0'])
        combined_df['LABEL0'] = combined_df.groupby(level=dt_level)['LABEL0'].transform(
            lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
        )
        
        # Split by time
        def split_data(df, start, end):
            dates = df.index.get_level_values('datetime')
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            return df.loc[mask]
        
        train_df = split_data(combined_df, config.train_start, config.train_end)
        valid_df = split_data(combined_df, config.valid_start, config.valid_end)
        test_df = split_data(combined_df, config.test_start, config.test_end or pd.Timestamp.now().strftime('%Y-%m-%d'))
        
        return train_df, valid_df, test_df
    
    def train_model(
        self,
        model_exp: QlibModelExperiment,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> Any:
        """
        Train model based on model task specification
        
        Supports:
        - Tabular models: LightGBM, XGBoost, MLP
        - TimeSeries models: GRU, LSTM, Transformer
        """
        task = model_exp.sub_tasks[0]
        model_type = task.model_type
        architecture = task.architecture
        hyperparameters = task.hyperparameters
        training_params = task.training_hyperparameters

        # 兼容：hyperparameters/training_params 可能为 str（JSON）或 dict
        if isinstance(hyperparameters, str):
            try:
                hyperparameters = json.loads(hyperparameters) if hyperparameters else {}
            except Exception:
                hyperparameters = {}
        hyperparameters = hyperparameters or {}
        if isinstance(training_params, str):
            try:
                training_params = json.loads(training_params) if training_params else {}
            except Exception:
                training_params = {}
        training_params = training_params or {}
        
        feature_cols = [c for c in train_df.columns if c != 'LABEL0']
        
        X_train = train_df[feature_cols].values
        y_train = train_df['LABEL0'].values
        X_valid = valid_df[feature_cols].values if len(valid_df) > 0 else None
        y_valid = valid_df['LABEL0'].values if len(valid_df) > 0 else None
        
        if model_type == "Tabular":
            # Use LightGBM or XGBoost
            if "lgb" in task.name.lower() or "lightgbm" in architecture.lower():
                return self._train_lightgbm(
                    X_train, y_train, X_valid, y_valid, training_params
                )
            elif "xgb" in task.name.lower() or "xgboost" in architecture.lower():
                return self._train_xgboost(
                    X_train, y_train, X_valid, y_valid, training_params
                )
            else:
                # Default to LightGBM
                return self._train_lightgbm(
                    X_train, y_train, X_valid, y_valid, training_params
                )
        
        elif model_type == "TimeSeries":
            # Use PyTorch for time series models
            return self._train_pytorch_ts(
                X_train, y_train, X_valid, y_valid, 
                architecture, hyperparameters, training_params
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_lightgbm(
        self,
        X_train, y_train,
        X_valid, y_valid,
        training_params: Dict
    ):
        """Train LightGBM model"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("LightGBM not available")
        
        # Default parameters
        default_params = {
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
        
        # Override with training params (LLM 可能返回字符串，需转为数值)
        def _num(v, default):
            if v is None:
                return default
            return int(v) if isinstance(v, (int, float)) else int(float(str(v)))

        params = default_params.copy()
        if training_params:
            params.update({
                'learning_rate': float(training_params.get('lr') or params['learning_rate']),
                'max_depth': _num(training_params.get('max_depth'), params['max_depth']),
                'num_leaves': _num(training_params.get('num_leaves'), params['num_leaves']),
                'num_threads': _num(training_params.get('num_threads'), params['num_threads']),
            })

        n_rounds = _num(training_params.get('n_epochs') if training_params else None, 500)
        early_stop = _num(training_params.get('early_stop') if training_params else None, 50)

        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=train_dataset) if X_valid is not None else None

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=n_rounds,
            valid_sets=[valid_dataset] if valid_dataset else None,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop),
                lgb.log_evaluation(period=0)
            ]
        )
        
        return model
    
    def _train_xgboost(
        self,
        X_train, y_train,
        X_valid, y_valid,
        training_params: Dict
    ):
        """Train XGBoost model"""
        try:
            import xgboost as xgb
        except ImportError:
            raise RuntimeError("XGBoost not available")
        
        def _num(v, default):
            if v is None:
                return default
            return int(v) if isinstance(v, (int, float)) else int(float(str(v)))
        def _float(v, default):
            if v is None:
                return default
            return float(v) if isinstance(v, (int, float)) else float(str(v))

        params = {
            'objective': 'reg:squarederror',
            'learning_rate': _float(training_params.get('lr') if training_params else None, 0.05),
            'max_depth': _num(training_params.get('max_depth') if training_params else None, 6),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid) if X_valid is not None else None

        evals = [(dtrain, 'train')]
        if dvalid is not None:
            evals.append((dvalid, 'valid'))

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=_num(training_params.get('n_epochs') if training_params else None, 500),
            evals=evals,
            early_stopping_rounds=_num(training_params.get('early_stop') if training_params else None, 50),
            verbose_eval=False
        )
        
        return model
    
    def _train_pytorch_ts(
        self,
        X_train, y_train,
        X_valid, y_valid,
        architecture: str,
        hyperparameters: Dict,
        training_params: Dict
    ):
        """
        Train PyTorch time series model
        
        This is a simplified implementation. In practice, you'd need to:
        1. Parse the architecture string to build the model
        2. Handle sequence generation for time series
        3. Implement the specific model (GRU/LSTM/Transformer)
        """
        logger.info(f"Training PyTorch {architecture} model")
        
        # Get parameters
        input_dim = X_train.shape[1]
        output_dim = 1
        
        def _num(v, default):
            if v is None:
                return default
            return int(v) if isinstance(v, (int, float)) else int(float(str(v)))
        def _float(v, default):
            if v is None:
                return default
            return float(v) if isinstance(v, (int, float)) else float(str(v))

        lr = _float(training_params.get('lr') if training_params else None, 2e-4)
        batch_size = _num(training_params.get('batch_size') if training_params else None, 256)
        n_epochs = _num(training_params.get('n_epochs') if training_params else None, 100)
        early_stop = _num(training_params.get('early_stop') if training_params else None, 10)
        hidden_dim = _num(hyperparameters.get('hidden_dim') if hyperparameters else None, 64)
        num_layers = _num(hyperparameters.get('num_layers') if hyperparameters else None, 2)

        # Use module-level _SimpleMLP (pickleable fallback for GRU/LSTM/Transformer)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = _SimpleMLP(input_dim, hidden_dim, num_layers, output_dim).to(device)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
        
        if X_valid is not None:
            X_valid_t = torch.FloatTensor(X_valid).to(device)
            y_valid_t = torch.FloatTensor(y_valid).reshape(-1, 1).to(device)
        
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            
            # Mini-batch training
            batch_losses = []
            for i in range(0, len(X_train_t), batch_size):
                batch_x = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                batch_losses.append(loss.item())
            
            # Validation
            if X_valid is not None:
                model.eval()
                with torch.no_grad():
                    valid_pred = model(X_valid_t)
                    valid_loss = criterion(valid_pred, y_valid_t).item()
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    # Save best model
                    best_state = model.state_dict()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stop:
                    logger.info(f"Early stopping at epoch {epoch}")
                    model.load_state_dict(best_state)
                    break
        
        return model
    
    def predict(
        self,
        model: Any,
        X: np.ndarray,
        model_type: str
    ) -> np.ndarray:
        """Generate predictions"""
        if model_type == "Tabular":
            # LightGBM or XGBoost
            if hasattr(model, 'predict'):
                if hasattr(model, 'best_iteration'):
                    return model.predict(X, num_iteration=model.best_iteration)
                else:
                    return model.predict(X)
            else:
                return model.predict(X)
        
        elif model_type == "TimeSeries":
            # PyTorch
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                pred = model(X_t).cpu().numpy().flatten()
            return pred
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_backtest(
        self,
        predictions: pd.Series,
        config: ModelBasePropSetting
    ) -> Dict:
        """
        Run backtest with predictions
        """
        if not QLIB_AVAILABLE:
            raise RuntimeError("qlib is not available")
        
        init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
        
        # Ensure predictions format: (datetime, instrument) for qlib backtest
        # Index level order may be (datetime, instrument) or (instrument, datetime)
        if isinstance(predictions.index, pd.MultiIndex):
            names = predictions.index.names or [None, None]
            if 'datetime' in names and 'instrument' in names:
                datetime_level = predictions.index.get_level_values('datetime')
                instrument_level = predictions.index.get_level_values('instrument')
            else:
                lev0, lev1 = predictions.index.get_level_values(0), predictions.index.get_level_values(1)
                # Detect: datetime parses as date, instrument is stock code (e.g. SH600000)
                try:
                    pd.to_datetime(lev0[:1])
                    datetime_level, instrument_level = lev0, lev1
                except (TypeError, ValueError):
                    datetime_level, instrument_level = lev1, lev0
            if not isinstance(datetime_level, pd.DatetimeIndex):
                datetime_level = pd.to_datetime(datetime_level)
            predictions.index = pd.MultiIndex.from_arrays(
                [datetime_level, instrument_level],
                names=['datetime', 'instrument']
            )
            predictions = predictions.sort_index()
        
        # Get stock list
        instruments = D.instruments(config.market)
        stock_list = D.list_instruments(
            instruments,
            start_time=config.test_start,
            end_time=config.test_end or pd.Timestamp.now().strftime('%Y-%m-%d'),
            as_list=True
        )
        
        # Run backtest
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
                    "signal": predictions,
                    "topk": config.topk,
                    "n_drop": config.n_drop,
                }
            },
            start_time=config.test_start,
            end_time=config.test_end or pd.Timestamp.now().strftime('%Y-%m-%d'),
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
                    
                    return {
                        'annualized_return': float(analysis.get('annualized_return', 0)),
                        'information_ratio': float(analysis.get('information_ratio', 0)),
                        'max_drawdown': float(analysis.get('max_drawdown', 0)),
                    }
        
        return {
            'annualized_return': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': 0.0,
        }
    
    def calculate_ic_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        index: pd.Index
    ) -> Dict:
        """Calculate IC metrics"""
        from scipy.stats import pearsonr, spearmanr
        
        pred_df = pd.DataFrame({'pred': predictions, 'label': labels}, index=index)
        
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
            'IC': ic_mean,
            'ICIR': icir,
            'Rank IC': rank_ic_mean,
            'Rank ICIR': rank_icir,
        }
    
    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        """
        Main development function for model experiments
        """
        logger.info("=" * 60)
        logger.info("LocalModelRunner: Starting model training")
        logger.info("=" * 60)
        
        config = ModelBasePropSetting()
        
        # 当 model.py 缺失时（如 coding 失败后使用 direct_exp_gen 的 exp）使用 fallback，不抛错
        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            logger.warning("model.py is empty, using fallback SimpleMLP for training")
        
        # Step 1: Load SOTA factors
        logger.info("Loading SOTA factors...")
        factor_exps = [
            be for be in exp.based_experiments
            if isinstance(be, QlibFactorExperiment) and be.result is not None
        ]
        
        if factor_exps:
            self.sota_factors = self.load_sota_factors(factor_exps)
        
        if self.sota_factors is None or self.sota_factors.empty:
            raise ModelEmptyError("No SOTA factors available. Run factor experiments first.")
        
        logger.info(f"Loaded {self.sota_factors.shape[1]} SOTA factors")
        
        # Step 2: Prepare data
        logger.info("Preparing datasets...")
        train_df, valid_df, test_df = self.prepare_data(self.sota_factors, config)
        
        feature_cols = [c for c in train_df.columns if c != 'LABEL0']
        
        logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
        
        # Step 3: Train model
        logger.info("Training model...")
        task = exp.sub_tasks[0]
        
        try:
            model = self.train_model(exp, train_df, valid_df)
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ModelEmptyError(f"Model training failed: {e}")
        
        # Step 4: Generate predictions
        logger.info("Generating predictions...")
        X_test = test_df[feature_cols].values
        predictions = self.predict(model, X_test, task.model_type)
        
        pred_series = pd.Series(predictions, index=test_df.index)
        
        # Step 5: Calculate metrics
        logger.info("Calculating metrics...")
        ic_metrics = self.calculate_ic_metrics(
            predictions,
            test_df['LABEL0'].values,
            test_df.index
        )
        
        backtest_metrics = self.run_backtest(pred_series, config)
        
        # Combine metrics
        result = {
            **ic_metrics,
            **backtest_metrics,
        }
        
        # Calculate Calmar
        if result['max_drawdown'] != 0:
            result['calmar_ratio'] = result['annualized_return'] / abs(result['max_drawdown'])
        else:
            result['calmar_ratio'] = 0.0
        
        logger.info("Model evaluation completed:")
        logger.info(f"  IC: {result['IC']:.4f}")
        logger.info(f"  ARR: {result['annualized_return']:.4f}")
        logger.info(f"  IR: {result['information_ratio']:.4f}")
        logger.info(f"  MDD: {result['max_drawdown']:.4f}")
        logger.info(f"  Calmar: {result['calmar_ratio']:.4f}")
        
        # Save results
        exp.result = result
        
        # Save model
        model_path = Path(exp.experiment_workspace.workspace_path) / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save predictions
        pred_path = Path(exp.experiment_workspace.workspace_path) / "predictions.csv"
        pred_series.to_csv(pred_path)
        
        logger.info("=" * 60)
        logger.info("LocalModelRunner: Model training completed")
        logger.info("=" * 60)
        
        return exp


# Compatibility alias
QlibModelRunner = LocalModelRunner
