"""
Enhanced Quant Proposal System with Multi-Objective Bandit

This module extends the original quant_proposal.py with:
1. Enhanced Bandit controller using multi-objective rewards
2. Cross-arm influence tracking
3. Better action selection strategies
"""

import json
import random
from typing import Tuple, Optional

from rdagent.app.qlib_rd_loop.conf_v2 import QUANT_PROP_SETTING_V2
from rdagent.components.proposal import FactorAndModelHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.proposal.quant_proposal import QuantTrace, QlibQuantHypothesis
from rdagent.scenarios.qlib.proposal.bandit_v2 import EnvControllerV2, Metrics
from rdagent.utils.agent.tpl import T


class QlibQuantTraceV2(QuantTrace):
    """
    Enhanced QuantTrace with V2 controller
    """
    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        # Replace controller with V2 version
        self.controller = EnvControllerV2(
            weights=QUANT_PROP_SETTING_V2.reward_weights,
            targets=QUANT_PROP_SETTING_V2.reward_targets
        )


class QlibQuantHypothesisGenV2(FactorAndModelHypothesisGen):
    """
    Enhanced hypothesis generator with multi-objective Bandit
    """
    
    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        self.current_action: Optional[str] = None
    
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        """
        Prepare context for hypothesis generation with enhanced Bandit
        """
        settings = QUANT_PROP_SETTING_V2
        
        # ========= Enhanced Bandit V2 ==========
        if settings.action_selection == "bandit_v2":
            if len(trace.hist) > 0:
                # Extract metrics from previous experiment
                last_exp, last_feedback = trace.hist[-1]
                metrics = trace.controller.extract_metrics(last_exp)
                
                # Record and update
                prev_action = last_exp.hypothesis.action if hasattr(last_exp.hypothesis, 'action') else "factor"
                reward_info = trace.controller.record(metrics, prev_action)
                
                # Log reward details
                logger = getattr(trace, 'logger', None) or print
                logger(f"Previous round reward: {reward_info['reward']:.4f}")
                logger(f"Reward components: {reward_info['detailed']}")
                
                # Decide next action using UCB
                action = trace.controller.decide(metrics, use_ucb=True)
            else:
                # First round - start with factor
                action = "factor"
        
        # ========= Original Bandit ==========
        elif settings.action_selection == "bandit":
            if len(trace.hist) > 0:
                metric = trace.controller.extract_metrics(trace.hist[-1][0])
                prev_action = trace.hist[-1][0].hypothesis.action
                trace.controller.record(metric, prev_action)
                action = trace.controller.decide(metric)
            else:
                action = "factor"
        
        # ========= LLM-based selection ==========
        elif settings.action_selection == "llm":
            action = self._llm_action_selection(trace)
        
        # ========= Random selection ==========
        elif settings.action_selection == "random":
            action = random.choice(["factor", "model"])

        # ========= Factor only (跳过 model 轮，确保因子回测成功) ==========
        elif settings.action_selection == "factor_only":
            action = "factor"

        else:
            raise ValueError(f"Unknown action_selection: {settings.action_selection}")
        
        self.current_action = action
        self.targets = action
        
        # Build RAG based on action
        rag = self._build_rag(action, trace)
        
        # Get cross-arm suggestions if enabled
        cross_arm_suggestions = ""
        if settings.enable_cross_arm_feedback:
            cross_arm_suggestions = self._get_cross_arm_suggestions(trace, action)
        
        # Build hypothesis and feedback context
        context_dict = self._build_context_dict(trace, action, rag, cross_arm_suggestions)
        
        return context_dict, True
    
    def _llm_action_selection(self, trace: Trace) -> str:
        """Use LLM to select action"""
        if len(trace.hist) == 0:
            return "factor"
        
        # Build prompt for LLM
        hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:hypothesis_and_feedback").r(trace=trace)
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available."
        )
        
        system_prompt = T("scenarios.qlib.prompts:action_gen.system").r()
        user_prompt = T("scenarios.qlib.prompts:action_gen.user").r(
            hypothesis_and_feedback=hypothesis_and_feedback,
            last_hypothesis_and_feedback="",
        )
        
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, 
            system_prompt, 
            json_mode=True
        )
        
        return json.loads(resp).get("action", "factor")
    
    def _build_rag(self, action: str, trace: Trace) -> str:
        """Build RAG prompt based on action and history"""
        if action == "factor":
            base = (
                "Data: CSI300 OHLCV only (open, high, low, close, volume). "
                "Do NOT propose factors requiring options, macro, sector, sentiment, earnings, or other external data. "
            )
            if len(trace.hist) < 6:
                return (
                    base
                    + "Try the easiest and fastest factors from price-volume, momentum, and volatility. "
                )
            else:
                return (
                    base
                    + "Try factors that can achieve high IC (e.g., ML-based factors from OHLCV). "
                    "Do not include factors similar to those in the SOTA library. "
                    "Consider non-linear combinations of OHLCV."
                )
        
        elif action == "model":
            return (
                "1. In Quantitative Finance, market data is time-series, and GRU/LSTM models are suitable. "
                "Avoid GNN models for now.\n"
                "2. Training data: ~478K samples for training, ~128K for validation. "
                "Design hyperparameters accordingly and control model size.\n"
                "3. If the previous model architecture is good but training failed, "
                "consider adjusting hyperparameters rather than changing architecture entirely.\n"
                "4. Pay attention to suggestions from Factor Arm about what factor types would help."
            )
        
        return ""
    
    def _get_cross_arm_suggestions(self, trace: Trace, for_action: str) -> str:
        """Get suggestions from the other arm"""
        if not QUANT_PROP_SETTING_V2.enable_cross_arm_feedback:
            return ""
        
        if hasattr(trace, 'controller') and hasattr(trace.controller, 'get_cross_arm_suggestions'):
            suggestions = trace.controller.get_cross_arm_suggestions(
                for_arm=for_action
            )
            if suggestions:
                return "Suggestions from the other arm:\n" + "\n".join([f"- {s}" for s in suggestions[-5:]])
        
        return ""
    
    def _build_context_dict(
        self, 
        trace: Trace, 
        action: str, 
        rag: str,
        cross_arm_suggestions: str
    ) -> dict:
        """Build the complete context dictionary"""
        
        # Build history specific to action
        if len(trace.hist) == 0:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
            last_hypothesis_and_feedback = None
            sota_hypothesis_and_feedback = None
        else:
            # Filter history for relevant experiments
            specific_trace = Trace(trace.scen)
            
            if action == "factor":
                # For factor action: include all factor experiments and best model
                model_inserted = False
                for i in range(len(trace.hist) - 1, -1, -1):
                    hist_exp, hist_feedback = trace.hist[i]
                    if hasattr(hist_exp.hypothesis, 'action'):
                        if hist_exp.hypothesis.action == "factor":
                            specific_trace.hist.insert(0, trace.hist[i])
                        elif hist_exp.hypothesis.action == "model" and hist_feedback.decision and not model_inserted:
                            specific_trace.hist.insert(0, trace.hist[i])
                            model_inserted = True
            
            elif action == "model":
                # For model action: include all model experiments and best factor
                factor_inserted = False
                for i in range(len(trace.hist) - 1, -1, -1):
                    hist_exp, hist_feedback = trace.hist[i]
                    if hasattr(hist_exp.hypothesis, 'action'):
                        if hist_exp.hypothesis.action == "model":
                            specific_trace.hist.insert(0, trace.hist[i])
                        elif hist_exp.hypothesis.action == "factor" and hist_feedback.decision and not factor_inserted:
                            specific_trace.hist.insert(0, trace.hist[i])
                            factor_inserted = True
            
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                    trace=specific_trace,
                )
            else:
                hypothesis_and_feedback = "No previous hypothesis and feedback available."
            
            # Get last experiment of the same action type
            last_hypothesis_and_feedback = None
            for i in range(len(trace.hist) - 1, -1, -1):
                if trace.hist[i][0].hypothesis.action == action:
                    last_hypothesis_and_feedback = T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                        experiment=trace.hist[i][0], 
                        feedback=trace.hist[i][1]
                    )
                    break
            
            # Get SOTA for model action
            sota_hypothesis_and_feedback = None
            if action == "model":
                for i in range(len(trace.hist) - 1, -1, -1):
                    if trace.hist[i][0].hypothesis.action == "model" and trace.hist[i][1].decision:
                        sota_hypothesis_and_feedback = T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
                            experiment=trace.hist[i][0], 
                            feedback=trace.hist[i][1]
                        )
                        break
        
        # Build final context
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback or "No previous experiments of this type.",
            "SOTA_hypothesis_and_feedback": sota_hypothesis_and_feedback or "No SOTA available.",
            "RAG": rag,
            "cross_arm_suggestions": cross_arm_suggestions,
            "hypothesis_output_format": T("scenarios.qlib.prompts:hypothesis_output_format_with_action").r(),
            "hypothesis_specification": (
                T("scenarios.qlib.prompts:factor_hypothesis_specification").r()
                if action == "factor"
                else T("scenarios.qlib.prompts:model_hypothesis_specification").r()
            ),
        }
        
        return context_dict
    
    def convert_response(self, response: str) -> QlibQuantHypothesis:
        """Convert LLM response to hypothesis"""
        response_dict = json.loads(response)
        
        hypothesis = QlibQuantHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
            action=response_dict.get("action", self.current_action or "factor"),
        )
        
        return hypothesis


# Backward compatibility
QlibQuantHypothesisGen = QlibQuantHypothesisGenV2
