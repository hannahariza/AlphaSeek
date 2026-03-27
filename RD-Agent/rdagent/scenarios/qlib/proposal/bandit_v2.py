"""
Enhanced Bandit System for RD-Agent Quant with Multi-Objective Reward Function

This module implements:
1. Multi-objective reward function: f(IC, IR, ARR, MDD, Calmar)
2. UCB-based arm selection with dynamic exploration
3. Cross-arm influence tracking
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Dict, Optional

import numpy as np


@dataclass
class Metrics:
    """Comprehensive metrics for reward calculation"""
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    rank_icir: float = 0.0
    arr: float = 0.0  # annualized_return
    ir: float = 0.0   # information_ratio
    mdd: float = 0.0  # max_drawdown (negative value)
    calmar: float = 0.0  # calmar_ratio
    
    def as_vector(self) -> np.ndarray:
        """Convert metrics to vector for bandit learning"""
        return np.array([
            self.ic,
            self.icir,
            self.rank_ic,
            self.rank_icir,
            self.arr,
            self.ir,
            -self.mdd,  # Convert to positive for reward (lower MDD is better)
            self.calmar,
        ])


class MultiObjectiveReward:
    """
    Multi-objective reward function with configurable weights
    
    Default weights (sum to 1.0):
    - ARR (年化收益): 40% - Primary objective
    - IR (信息比率): 20% - Risk-adjusted return
    - IC: 20% - Prediction quality
    - MDD (最大回撤): 15% - Risk control (negative weight)
    - Calmar: 5% - Risk-adjusted consistency
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        targets: Optional[Dict[str, float]] = None
    ):
        """
        Initialize reward function with custom weights and targets
        
        Args:
            weights: Dict of metric weights (must sum to ~1.0)
            targets: Dict of target values for normalization
        """
        # Default weights
        self.weights = weights or {
            "arr": 0.40,
            "ir": 0.20,
            "ic": 0.20,
            "mdd": 0.15,
            "calmar": 0.05,
        }
        
        # Target values for normalization
        self.targets = targets or {
            "ic": 0.05,
            "icir": 0.30,
            "rank_ic": 0.05,
            "rank_icir": 0.30,
            "arr": 0.15,  # 15% annualized return
            "ir": 1.0,     # IR = 1.0
            "mdd": -0.10,  # Max drawdown -10%
            "calmar": 1.5,  # Calmar ratio 1.5
        }
    
    def normalize(self, value: float, target: float, epsilon: float = 1e-6) -> float:
        """
        Normalize value to [-1, 1] range based on target
        
        Uses tanh-like normalization: value / (|target| + |value| + epsilon)
        This ensures:
        - value = target → ~0.5
        - value >> target → ~1.0
        - value << target → ~0.0 (or negative if opposite sign)
        """
        if target == 0:
            return math.tanh(value)
        
        # Normalize relative to target
        ratio = value / (abs(target) + epsilon)
        
        # Apply sigmoid-like compression to [-1, 1]
        return 2 / (1 + math.exp(-2 * ratio)) - 1
    
    def compute(self, metrics: Metrics) -> float:
        """
        Compute multi-objective reward from metrics
        
        Returns:
            float: Reward value in range roughly [-1, 1]
        """
        # Normalize each metric
        ic_norm = self.normalize(metrics.ic, self.targets["ic"])
        ir_norm = self.normalize(metrics.ir, self.targets["ir"])
        arr_norm = self.normalize(metrics.arr, self.targets["arr"])
        calmar_norm = self.normalize(metrics.calmar, self.targets["calmar"])
        
        # For MDD: lower (more negative) is worse, so we negate
        # MDD is already stored as negative, so -MDD is positive (good)
        mdd_norm = self.normalize(-metrics.mdd, abs(self.targets["mdd"]))
        
        # Weighted sum
        reward = (
            self.weights["arr"] * arr_norm +
            self.weights["ir"] * ir_norm +
            self.weights["ic"] * ic_norm +
            self.weights["mdd"] * mdd_norm +
            self.weights["calmar"] * calmar_norm
        )
        
        return reward
    
    def compute_detailed(self, metrics: Metrics) -> Dict[str, float]:
        """
        Compute reward with detailed breakdown for debugging
        """
        ic_norm = self.normalize(metrics.ic, self.targets["ic"])
        ir_norm = self.normalize(metrics.ir, self.targets["ir"])
        arr_norm = self.normalize(metrics.arr, self.targets["arr"])
        calmar_norm = self.normalize(metrics.calmar, self.targets["calmar"])
        mdd_norm = self.normalize(-metrics.mdd, abs(self.targets["mdd"]))
        
        return {
            "total_reward": (
                self.weights["arr"] * arr_norm +
                self.weights["ir"] * ir_norm +
                self.weights["ic"] * ic_norm +
                self.weights["mdd"] * mdd_norm +
                self.weights["calmar"] * calmar_norm
            ),
            "components": {
                "arr": {"raw": metrics.arr, "norm": arr_norm, "weighted": self.weights["arr"] * arr_norm},
                "ir": {"raw": metrics.ir, "norm": ir_norm, "weighted": self.weights["ir"] * ir_norm},
                "ic": {"raw": metrics.ic, "norm": ic_norm, "weighted": self.weights["ic"] * ic_norm},
                "mdd": {"raw": metrics.mdd, "norm": mdd_norm, "weighted": self.weights["mdd"] * mdd_norm},
                "calmar": {"raw": metrics.calmar, "norm": calmar_norm, "weighted": self.weights["calmar"] * calmar_norm},
            }
        }


class LinearThompsonTwoArm:
    """
    Linear Thompson Sampling for two-arm bandit (factor vs model)
    
    This implements a contextual bandit where:
    - Context = normalized metrics vector
    - Reward = multi-objective reward
    - Two arms: "factor" and "model"
    """
    
    def __init__(
        self,
        dim: int = 8,
        prior_var: float = 1.0,
        noise_var: float = 1.0,
        reward_fn: Optional[MultiObjectiveReward] = None
    ):
        self.dim = dim
        self.noise_var = noise_var
        self.reward_fn = reward_fn or MultiObjectiveReward()
        
        # Each arm has its own posterior: mean & precision matrix
        self.mean = {
            "factor": np.zeros(dim),
            "model": np.zeros(dim),
        }
        self.precision = {
            "factor": np.eye(dim) / prior_var,
            "model": np.eye(dim) / prior_var,
        }
        
        # Pull counts for UCB exploration
        self.pull_counts = {
            "factor": 0,
            "model": 0,
        }
        
        # Cumulative rewards for tracking
        self.cumulative_rewards = {
            "factor": 0.0,
            "model": 0.0,
        }
    
    def sample_reward(self, arm: str, x: np.ndarray) -> float:
        """
        Thompson sampling: sample from posterior predictive distribution
        
        Args:
            arm: "factor" or "model"
            x: context vector (normalized metrics)
        
        Returns:
            Sampled reward
        """
        P = self.precision[arm]
        # Ensure symmetry
        P = 0.5 * (P + P.T)
        
        eps = 1e-6
        try:
            # Sample from multivariate normal
            cov = np.linalg.inv(P + eps * np.eye(self.dim))
            L = np.linalg.cholesky(cov)
            z = np.random.randn(self.dim)
            w_sample = self.mean[arm] + L @ z
        except np.linalg.LinAlgError:
            # Fallback to mean if Cholesky fails
            w_sample = self.mean[arm]
        
        return float(np.dot(w_sample, x))
    
    def ucb_score(self, arm: str, x: np.ndarray, c: float = 1.0) -> float:
        """
        Upper Confidence Bound score with exploration bonus
        
        UCB = Q(a) + c * sqrt(2 * ln(N) / n_a)
        where:
        - Q(a) = expected reward for arm a
        - N = total pulls
        - n_a = pulls for arm a
        - c = exploration constant
        """
        # Expected reward (exploitation)
        q_value = float(np.dot(self.mean[arm], x))
        
        # Exploration bonus
        total_pulls = sum(self.pull_counts.values())
        if self.pull_counts[arm] == 0:
            exploration_bonus = float('inf')  # Force exploration of unpulled arms
        else:
            exploration_bonus = c * math.sqrt(2 * math.log(total_pulls + 1) / self.pull_counts[arm])
        
        return q_value + exploration_bonus
    
    def update(self, arm: str, x: np.ndarray, r: float) -> None:
        """
        Update posterior after observing reward
        
        Args:
            arm: "factor" or "model"
            x: context vector
            r: observed reward
        """
        # Update precision matrix
        P = self.precision[arm]
        P += np.outer(x, x) / self.noise_var
        self.precision[arm] = P
        
        # Update mean using Sherman-Morrison style update
        self.mean[arm] = np.linalg.solve(
            P,
            P @ self.mean[arm] + (r / self.noise_var) * x
        )
        
        # Update statistics
        self.pull_counts[arm] += 1
        self.cumulative_rewards[arm] += r
    
    def next_arm(self, x: np.ndarray, use_ucb: bool = True, c: float = 1.0) -> str:
        """
        Select next arm using UCB or Thompson Sampling
        
        Args:
            x: context vector (normalized metrics from previous round)
            use_ucb: If True, use UCB; else use Thompson Sampling
            c: exploration constant for UCB
        
        Returns:
            Selected arm: "factor" or "model"
        """
        if use_ucb:
            scores = {arm: self.ucb_score(arm, x, c) for arm in ["factor", "model"]}
        else:
            scores = {arm: self.sample_reward(arm, x) for arm in ["factor", "model"]}
        
        return max(scores, key=scores.get)
    
    def get_arm_stats(self) -> Dict:
        """Get statistics for both arms"""
        return {
            "factor": {
                "pulls": self.pull_counts["factor"],
                "cumulative_reward": self.cumulative_rewards["factor"],
                "mean": self.mean["factor"].tolist(),
            },
            "model": {
                "pulls": self.pull_counts["model"],
                "cumulative_reward": self.cumulative_rewards["model"],
                "mean": self.mean["model"].tolist(),
            },
        }


class EnvControllerV2:
    """
    Enhanced Environment Controller with multi-objective reward
    and cross-arm influence tracking
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        targets: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize controller with multi-objective reward function
        """
        self.reward_fn = MultiObjectiveReward(weights, targets)
        self.bandit = LinearThompsonTwoArm(
            dim=8,
            prior_var=10.0,
            noise_var=0.5,
            reward_fn=self.reward_fn
        )
        
        # Track cross-arm influences
        self.factor_to_model_suggestions: List[str] = []
        self.model_to_factor_suggestions: List[str] = []
        
        # Track which arm produced better results
        self.best_reward_by_arm: Dict[str, float] = {
            "factor": -float('inf'),
            "model": -float('inf'),
        }
    
    def extract_metrics(self, experiment) -> Metrics:
        """
        Extract metrics from experiment result
        """
        try:
            result = experiment.result
            if result is None:
                return Metrics()
            
            # Handle different result formats
            if isinstance(result, dict):
                ic = result.get("IC", 0.0)
                icir = result.get("ICIR", 0.0)
                rank_ic = result.get("Rank IC", 0.0)
                rank_icir = result.get("Rank ICIR", 0.0)
                arr = result.get("annualized_return", 0.0)
                ir = result.get("information_ratio", 0.0)
                mdd = result.get("max_drawdown", 0.0)
                calmar = result.get("calmar_ratio", 0.0)
            elif hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
                ic = result_dict.get("IC", 0.0)
                icir = result_dict.get("ICIR", 0.0)
                rank_ic = result_dict.get("Rank IC", 0.0)
                rank_icir = result_dict.get("Rank ICIR", 0.0)
                arr = result_dict.get("1day.excess_return_with_cost.annualized_return", 0.0)
                ir = result_dict.get("1day.excess_return_with_cost.information_ratio", 0.0)
                mdd = result_dict.get("1day.excess_return_with_cost.max_drawdown", 0.0)
                calmar = result_dict.get("calmar_ratio", arr / abs(mdd) if mdd != 0 else 0.0)
            else:
                return Metrics()
            
            return Metrics(
                ic=ic,
                icir=icir,
                rank_ic=rank_ic,
                rank_icir=rank_icir,
                arr=arr,
                ir=ir,
                mdd=mdd,
                calmar=calmar
            )
        except Exception as e:
            print(f"Error extracting metrics: {e}")
            return Metrics()
    
    def reward(self, metrics: Metrics) -> float:
        """Compute reward from metrics"""
        return self.reward_fn.compute(metrics)
    
    def decide(self, metrics: Metrics, use_ucb: bool = True) -> str:
        """
        Decide next arm based on metrics from previous round
        
        Args:
            metrics: Metrics from previous experiment
            use_ucb: Use UCB (True) or Thompson Sampling (False)
        
        Returns:
            Selected arm: "factor" or "model"
        """
        x = metrics.as_vector()
        return self.bandit.next_arm(x, use_ucb=use_ucb)
    
    def record(self, metrics: Metrics, arm: str) -> Dict:
        """
        Record metrics and update bandit
        
        Returns:
            Dict with reward details
        """
        r = self.reward(metrics)
        x = metrics.as_vector()
        self.bandit.update(arm, x, r)
        
        # Update best reward tracking
        if r > self.best_reward_by_arm[arm]:
            self.best_reward_by_arm[arm] = r
        
        # Return detailed reward info
        detailed = self.reward_fn.compute_detailed(metrics)
        return {
            "arm": arm,
            "reward": r,
            "detailed": detailed,
            "bandit_stats": self.bandit.get_arm_stats(),
        }
    
    def record_cross_arm_influence(
        self,
        from_arm: str,
        to_arm: str,
        suggestion: str
    ) -> None:
        """
        Record feedback from one arm to another
        
        Args:
            from_arm: Source arm ("factor" or "model")
            to_arm: Target arm ("factor" or "model")
            suggestion: Text suggestion for the target arm
        """
        if from_arm == "factor" and to_arm == "model":
            self.factor_to_model_suggestions.append(suggestion)
        elif from_arm == "model" and to_arm == "factor":
            self.model_to_factor_suggestions.append(suggestion)
    
    def get_cross_arm_suggestions(self, for_arm: str) -> List[str]:
        """
        Get suggestions from the other arm
        
        Args:
            for_arm: The arm requesting suggestions ("factor" or "model")
        
        Returns:
            List of suggestions from the other arm
        """
        if for_arm == "factor":
            return self.model_to_factor_suggestions[-5:]  # Last 5 suggestions
        else:
            return self.factor_to_model_suggestions[-5:]


def extract_metrics_from_experiment(experiment) -> Metrics:
    """Helper function for backward compatibility"""
    controller = EnvControllerV2()
    return controller.extract_metrics(experiment)
