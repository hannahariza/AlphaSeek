"""
Enhanced Configuration for RD-Agent Quant Integration with QuantaAlpha

This configuration provides:
1. Time ranges aligned with QuantaAlpha
2. Multi-objective reward weights
3. Local execution settings (non-Docker)
4. Factor pool management settings
"""

from typing import Optional, Dict
from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class ModelBasePropSettingV2(BasePropSetting):
    """
    Model configuration with QuantaAlpha-aligned time ranges and local execution
    """
    model_config = SettingsConfigDict(env_prefix="QLIB_MODEL_", protected_namespaces=())
    
    # Scenario and component classes
    scen: str = "rdagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario"
    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesisGen"
    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    coder: str = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    runner: str = "rdagent.scenarios.qlib.developer.model_runner_local.LocalModelRunner"
    summarizer: str = "rdagent.scenarios.qlib.developer.feedback_v2.QlibModelExperiment2FeedbackV2"
    
    # Evolution settings
    evolving_n: int = 30  # Maximum number of evolutions
    early_stop_rounds: int = 5  # Stop if no improvement for N rounds
    
    # Time ranges (aligned with QuantaAlpha)
    train_start: str = "2016-01-01"
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: Optional[str] = "2025-12-26"
    
    # Market settings
    market: str = "csi300"
    benchmark: str = "SH000300"
    
    # Backtest settings
    topk: int = 50
    n_drop: int = 5
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    
    # Multi-objective reward weights
    reward_weights: Dict[str, float] = {
        "arr": 0.40,      # Annualized return (primary)
        "ir": 0.20,       # Information ratio
        "ic": 0.20,       # IC
        "mdd": 0.15,      # Max drawdown (risk control)
        "calmar": 0.05,   # Calmar ratio
    }
    
    # Target values for normalization
    reward_targets: Dict[str, float] = {
        "ic": 0.05,
        "icir": 0.30,
        "rank_ic": 0.05,
        "rank_icir": 0.30,
        "arr": 0.15,
        "ir": 1.0,
        "mdd": -0.10,
        "calmar": 1.5,
    }
    
    # Local execution (non-Docker)
    env_type: str = "conda"  # "conda" or "docker"
    conda_env_name: str = "rdagent"


class FactorBasePropSettingV2(BasePropSetting):
    """
    Factor configuration with QuantaAlpha integration
    """
    model_config = SettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())
    
    # Scenario and component classes
    scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen"
    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    runner: str = "rdagent.scenarios.qlib.developer.factor_runner_local.LocalFactorRunner"
    summarizer: str = "rdagent.scenarios.qlib.developer.feedback_v2.QlibFactorExperiment2FeedbackV2"
    
    # Evolution settings
    evolving_n: int = 30
    early_stop_rounds: int = 5
    
    # Time ranges (same as model)
    train_start: str = "2016-01-01"
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: Optional[str] = "2025-12-26"
    
    # Market settings
    market: str = "csi300"
    benchmark: str = "SH000300"
    
    # Backtest settings
    topk: int = 50
    n_drop: int = 5
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    
    # Factor pool settings
    factor_ic_threshold: float = 0.01  # Minimum IC for factor selection
    initial_factor_count: int = 50  # Number of factors to start with
    max_factors_per_round: int = 20  # Maximum new factors per round
    deduplication_threshold: float = 0.99  # IC threshold for deduplication
    
    # Multi-objective reward weights (same as model)
    reward_weights: Dict[str, float] = {
        "arr": 0.40,
        "ir": 0.20,
        "ic": 0.20,
        "mdd": 0.15,
        "calmar": 0.05,
    }
    
    # Target values
    reward_targets: Dict[str, float] = {
        "ic": 0.05,
        "icir": 0.30,
        "rank_ic": 0.05,
        "rank_icir": 0.30,
        "arr": 0.15,
        "ir": 1.0,
        "mdd": -0.10,
        "calmar": 1.5,
    }
    
    # Local execution
    env_type: str = "conda"
    conda_env_name: str = "rdagent"


class QuantBasePropSettingV2(BasePropSetting):
    """
    Quant (Factor + Model) joint optimization configuration
    """
    model_config = SettingsConfigDict(env_prefix="QLIB_QUANT_", protected_namespaces=())
    
    # Scenario
    scen: str = "rdagent.scenarios.qlib.experiment.quant_experiment_v2.QlibQuantScenarioV2"
    
    # Quant hypothesis generator with enhanced Bandit
    quant_hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.quant_proposal_v2.QlibQuantHypothesisGenV2"
    
    # Model components
    model_hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    model_coder: str = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    model_runner: str = "rdagent.scenarios.qlib.developer.model_runner_local.LocalModelRunner"
    model_summarizer: str = "rdagent.scenarios.qlib.developer.feedback_v2.QlibModelExperiment2FeedbackV2"
    
    # Factor components
    factor_hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    factor_coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    factor_runner: str = "rdagent.scenarios.qlib.developer.factor_runner_local.LocalFactorRunner"
    factor_summarizer: str = "rdagent.scenarios.qlib.developer.feedback_v2.QlibFactorExperiment2FeedbackV2"
    
    # Evolution settings
    evolving_n: int = 30
    early_stop_rounds: int = 5
    
    # Action selection strategy
    action_selection: str = "bandit_v2"  # "bandit_v2", "factor_only", "bandit", "llm", "random"
    
    # Time ranges
    train_start: str = "2016-01-01"
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: Optional[str] = "2025-12-26"
    
    # Market settings
    market: str = "csi300"
    benchmark: str = "SH000300"
    
    # Backtest settings
    topk: int = 50
    n_drop: int = 5
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    
    # Factor pool settings
    factor_ic_threshold: float = 0.01
    initial_factor_count: int = 50
    max_factors_per_round: int = 20
    
    # Multi-objective reward weights
    reward_weights: Dict[str, float] = {
        "arr": 0.40,
        "ir": 0.20,
        "ic": 0.20,
        "mdd": 0.15,
        "calmar": 0.05,
    }
    
    # Target values
    reward_targets: Dict[str, float] = {
        "ic": 0.05,
        "icir": 0.30,
        "rank_ic": 0.05,
        "rank_icir": 0.30,
        "arr": 0.15,
        "ir": 1.0,
        "mdd": -0.10,
        "calmar": 1.5,
    }
    
    # Local execution
    env_type: str = "conda"
    conda_env_name: str = "rdagent"
    
    # Cross-arm communication
    enable_cross_arm_feedback: bool = True
    max_cross_arm_suggestions: int = 5


# Create instances for easy import
MODEL_PROP_SETTING_V2 = ModelBasePropSettingV2()
FACTOR_PROP_SETTING_V2 = FactorBasePropSettingV2()
QUANT_PROP_SETTING_V2 = QuantBasePropSettingV2()

# Backward compatibility
MODEL_PROP_SETTING = MODEL_PROP_SETTING_V2
FACTOR_PROP_SETTING = FACTOR_PROP_SETTING_V2
QUANT_PROP_SETTING = QUANT_PROP_SETTING_V2
