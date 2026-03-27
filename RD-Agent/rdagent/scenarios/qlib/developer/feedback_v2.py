"""
Enhanced Feedback System for RD-Agent Quant with Cross-Arm Influence Analysis

This module implements feedback generation for both Factor and Model arms,
including cross-arm influence tracking and suggestions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

# Import enhanced bandit system
from rdagent.scenarios.qlib.proposal.bandit_v2 import MultiObjectiveReward

# Important metrics for comparison
IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
    "information_ratio",
    "calmar_ratio",
]


def process_results(current_result: Dict, sota_result: Dict) -> str:
    """
    Process and format results for comparison
    """
    if current_result is None:
        return "Current experiment failed - no results available"
    
    if sota_result is None:
        return "No SOTA results available for comparison"
    
    # Convert to dataframes
    current_df = pd.DataFrame(current_result, index=[0]).T
    sota_df = pd.DataFrame(sota_result, index=[0]).T
    
    current_df.index.name = "metric"
    sota_df.index.name = "metric"
    
    current_df.rename(columns={0: "Current"}, inplace=True)
    sota_df.rename(columns={0: "SOTA"}, inplace=True)
    
    # Combine
    combined = pd.concat([current_df, sota_df], axis=1)
    
    # Format results
    results = []
    for metric in combined.index:
        current_val = combined.loc[metric, "Current"]
        sota_val = combined.loc[metric, "SOTA"]
        
        if pd.notna(current_val) and pd.notna(sota_val):
            results.append(f"{metric}: Current={current_val:.6f}, SOTA={sota_val:.6f}")
    
    return "; ".join(results) if results else "No comparable metrics"


def calculate_metric_changes(current: Dict, sota: Dict) -> Dict[str, float]:
    """
    Calculate percentage changes between current and SOTA
    """
    changes = {}
    
    for metric in ["IC", "annualized_return", "information_ratio", "calmar_ratio"]:
        if metric in current and metric in sota:
            current_val = current[metric]
            sota_val = sota[metric]
            
            if sota_val != 0:
                change_pct = (current_val - sota_val) / abs(sota_val) * 100
                changes[metric] = change_pct
            else:
                changes[metric] = float('inf') if current_val > 0 else float('-inf')
    
    return changes


class QlibFactorExperiment2FeedbackV2(Experiment2Feedback):
    """
    Enhanced Factor Feedback with cross-arm influence analysis
    """
    
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for factor experiments
        """
        hypothesis = exp.hypothesis
        current_result = exp.result
        
        logger.info("Generating Factor feedback...")
        
        # Get SOTA result
        sota_result = None
        if exp.based_experiments:
            for base_exp in reversed(exp.based_experiments):
                if hasattr(base_exp, 'result') and base_exp.result:
                    sota_result = base_exp.result
                    break
        
        # Process results
        combined_result = process_results(current_result, sota_result)
        metric_changes = calculate_metric_changes(current_result or {}, sota_result or {})
        
        # Get Model Arm feedback for cross-influence
        model_suggestions = self._get_model_arm_suggestions(trace)
        
        # Generate system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )
        
        # Enhanced user prompt with cross-arm influence
        usr_prompt = f"""
# Factor Experiment Feedback Request

## Hypothesis
{hypothesis.hypothesis if hypothesis else "No hypothesis provided"}

## Results Comparison
{combined_result}

## Metric Changes (%)
"""
        for metric, change in metric_changes.items():
            usr_prompt += f"- {metric}: {change:+.2f}%\n"
        
        usr_prompt += f"""
## Model Arm Suggestions
The following suggestions come from the Model Arm based on its training experience:
{model_suggestions if model_suggestions else "No specific suggestions from Model Arm yet."}

## Task Details
"""
        
        # Add task details
        for task in exp.sub_tasks:
            task_info = task.get_task_information_and_implementation_result() if hasattr(task, 'get_task_information_and_implementation_result') else str(task)
            usr_prompt += f"\n{task_info}"
        
        usr_prompt += """

## Required Output
Please provide feedback in the following JSON format:
{
    "Observations": "Detailed observations about factor performance",
    "Feedback for Hypothesis": "Evaluation of whether the hypothesis was validated",
    "Impact on Model Arm": "How these factors might help/hinder model training",
    "New Hypothesis": "Suggested next hypothesis for factor exploration",
    "Suggestions for Model Arm": "Specific suggestions to pass to Model Arm",
    "Reasoning": "Reasoning for the decision",
    "Replace Best Result": true/false
}
"""
        
        # Call LLM (允许 Replace Best Result 为 bool 或 str，LLM 可能返回 true/false)
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Union[str, bool]],
        )
        
        # Parse response
        response_json = json.loads(response)
        
        # Extract fields
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_eval = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis")
        reasoning = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))
        
        # Record cross-arm suggestions
        impact_on_model = response_json.get("Impact on Model Arm", "")
        suggestions_for_model = response_json.get("Suggestions for Model Arm", "")
        
        if hasattr(trace, 'controller'):
            if hasattr(trace.controller, 'record_cross_arm_influence'):
                trace.controller.record_cross_arm_influence(
                    from_arm="factor",
                    to_arm="model",
                    suggestion=f"{impact_on_model} {suggestions_for_model}".strip()
                )
        
        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_eval,
            new_hypothesis=new_hypothesis,
            reason=reasoning,
            decision=decision,
        )
    
    def _get_model_arm_suggestions(self, trace: Trace) -> str:
        """Get suggestions from Model Arm"""
        if hasattr(trace, 'controller'):
            if hasattr(trace.controller, 'get_cross_arm_suggestions'):
                suggestions = trace.controller.get_cross_arm_suggestions(for_arm="factor")
                if suggestions:
                    return "\n".join([f"- {s}" for s in suggestions])
        return ""


class QlibModelExperiment2FeedbackV2(Experiment2Feedback):
    """
    Enhanced Model Feedback with cross-arm influence analysis
    """
    
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for model experiments
        """
        hypothesis = exp.hypothesis
        current_result = exp.result
        
        logger.info("Generating Model feedback...")
        
        # Get SOTA
        sota_hypothesis, sota_experiment = trace.get_sota_hypothesis_and_experiment()
        
        # Process results
        sota_result = sota_experiment.result if sota_experiment else None
        combined_result = process_results(current_result, sota_result)
        metric_changes = calculate_metric_changes(current_result or {}, sota_result or {})
        
        # Get Factor Arm feedback for cross-influence
        factor_suggestions = self._get_factor_arm_suggestions(trace)
        
        # Get model code comparison
        current_code = exp.sub_workspace_list[0].file_dict.get("model.py", "No code available")
        sota_code = sota_experiment.sub_workspace_list[0].file_dict.get("model.py", "No SOTA code") if sota_experiment else "No SOTA code"
        
        # Generate system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )
        
        # Enhanced user prompt
        usr_prompt = f"""
# Model Experiment Feedback Request

## SOTA Hypothesis
{sota_hypothesis.hypothesis if sota_hypothesis else "No previous SOTA hypothesis"}

## Current Hypothesis
{hypothesis.hypothesis if hypothesis else "No hypothesis provided"}

## Results Comparison
{combined_result}

## Metric Changes (%)
"""
        for metric, change in metric_changes.items():
            usr_prompt += f"- {metric}: {change:+.2f}%\n"
        
        usr_prompt += f"""
## Factor Arm Suggestions
The following suggestions come from the Factor Arm based on its exploration:
{factor_suggestions if factor_suggestions else "No specific suggestions from Factor Arm yet."}

## Model Task
{exp.sub_tasks[0].get_task_information() if exp.sub_tasks else "No task info"}

## SOTA Model Code (Summary)
{sota_code[:500]}... [truncated]

## Current Model Code (Summary)
{current_code[:500]}... [truncated]

## Required Output
Please provide feedback in the following JSON format:
{{
    "Observations": "Detailed observations about model performance",
    "Feedback for Hypothesis": "Evaluation of whether the hypothesis was validated",
    "Code Analysis": "Analysis of code changes and their impact",
    "Impact on Factor Arm": "What types of factors would help this model",
    "New Hypothesis": "Suggested next hypothesis for model exploration",
    "Suggestions for Factor Arm": "Specific suggestions to pass to Factor Arm",
    "Reasoning": "Reasoning for the decision",
    "Decision": true/false
}}
"""
        
        # Call LLM (允许 Decision 为 bool 或 str，LLM 可能返回 true/false)
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Union[str, bool]],
        )
        
        # Parse response
        response_json = json.loads(response)
        
        # Extract fields
        observations = response_json.get("Observations", "No observations")
        hypothesis_eval = response_json.get("Feedback for Hypothesis", "No hypothesis feedback")
        code_analysis = response_json.get("Code Analysis", "No code analysis")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis")
        reasoning = response_json.get("Reasoning", "No reasoning")
        decision = convert2bool(response_json.get("Decision", "false"))
        
        # Record cross-arm suggestions
        impact_on_factor = response_json.get("Impact on Factor Arm", "")
        suggestions_for_factor = response_json.get("Suggestions for Factor Arm", "")
        
        if hasattr(trace, 'controller'):
            if hasattr(trace.controller, 'record_cross_arm_influence'):
                trace.controller.record_cross_arm_influence(
                    from_arm="model",
                    to_arm="factor",
                    suggestion=f"{impact_on_factor} {suggestions_for_factor}".strip()
                )
        
        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_eval,
            new_hypothesis=new_hypothesis,
            reason=reasoning,
            decision=decision,
        )
    
    def _get_factor_arm_suggestions(self, trace: Trace) -> str:
        """Get suggestions from Factor Arm"""
        if hasattr(trace, 'controller'):
            if hasattr(trace.controller, 'get_cross_arm_suggestions'):
                suggestions = trace.controller.get_cross_arm_suggestions(for_arm="model")
                if suggestions:
                    return "\n".join([f"- {s}" for s in suggestions])
        return ""


# Backward compatibility aliases
QlibFactorExperiment2Feedback = QlibFactorExperiment2FeedbackV2
QlibModelExperiment2Feedback = QlibModelExperiment2FeedbackV2
