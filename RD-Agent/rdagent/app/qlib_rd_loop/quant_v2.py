"""
Enhanced Quant (Factor & Model) workflow with QuantaAlpha integration

This is the main entry point for running the enhanced RD-Agent Quant loop with:
1. QuantaAlpha factor library integration
2. Multi-objective Bandit scheduling
3. Cross-arm influence tracking
4. Local execution (non-Docker)

Usage:
    # Run full loop
    python -m rdagent.app.qlib_rd_loop.quant_v2

    # Resume from checkpoint
    python -m rdagent.app.qlib_rd_loop.quant_v2 log/__session__/5/0_propose --step_n 1

    # Run with limited loops
    python -m rdagent.app.qlib_rd_loop.quant_v2 --loop_n 20
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Load .env file before any other imports to ensure environment variables are set
# This must be done before importing any RD-Agent modules that use LLMSettings
from dotenv import load_dotenv

# Find and load .env file
_current_file = Path(__file__).resolve()
_env_path = _current_file.parents[3] / ".env"  # rdagent/app/qlib_rd_loop/.env -> RD-Agent/.env
if _env_path.exists():
    load_dotenv(_env_path, override=True)
    print(f"[ENV] Loaded environment from {_env_path}")
else:
    # Fallback: try to load from current working directory
    load_dotenv(override=True)
    print(f"[ENV] Loaded environment from current directory")

import fire
import typer
from typing_extensions import Annotated

# Use enhanced configuration
from rdagent.app.qlib_rd_loop.conf_v2 import QUANT_PROP_SETTING_V2
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError, FactorEmptyError, ModelEmptyError
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger

# Import enhanced Quant trace
from rdagent.scenarios.qlib.proposal.quant_proposal_v2 import QlibQuantTraceV2


class QuantRDLoopV2(RDLoop):
    """
    Enhanced Quant RD Loop with:
    - Multi-objective Bandit scheduling
    - QuantaAlpha factor integration
    - Cross-arm influence tracking
    - Local execution
    """
    
    skip_loop_error = (
        FactorEmptyError,
        ModelEmptyError,
        CoderError,
    )
    skip_loop_error_stepname = "running"  # 当 coding 失败时跳到 running，由 model_runner 使用 fallback
    
    def __init__(self, PROP_SETTING):
        # Import scenario
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")
        
        # Log settings
        logger.log_object(PROP_SETTING.model_dump(), tag="QUANT_V2_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")
        
        # Initialize hypothesis generator (with enhanced Bandit)
        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.quant_hypothesis_gen)(scen)
        logger.log_object(self.hypothesis_gen, tag="quant hypothesis generator (V2)")
        
        # Initialize hypothesis to experiment converters
        self.factor_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.factor_hypothesis2experiment
        )()
        logger.log_object(self.factor_hypothesis2experiment, tag="factor hypothesis2experiment")
        
        self.model_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.model_hypothesis2experiment
        )()
        logger.log_object(self.model_hypothesis2experiment, tag="model hypothesis2experiment")
        
        # Initialize coders
        self.factor_coder: Developer = import_class(PROP_SETTING.factor_coder)(scen)
        logger.log_object(self.factor_coder, tag="factor coder")
        
        self.model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
        logger.log_object(self.model_coder, tag="model coder")
        
        # Initialize runners (local execution)
        self.factor_runner: Developer = import_class(PROP_SETTING.factor_runner)(scen)
        logger.log_object(self.factor_runner, tag="factor runner (local)")
        
        self.model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)
        logger.log_object(self.model_runner, tag="model runner (local)")
        
        # Initialize summarizers (with cross-arm feedback)
        self.factor_summarizer: Experiment2Feedback = import_class(PROP_SETTING.factor_summarizer)(scen)
        logger.log_object(self.factor_summarizer, tag="factor summarizer (V2)")
        
        self.model_summarizer: Experiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)
        logger.log_object(self.model_summarizer, tag="model summarizer (V2)")
        
        # Initialize enhanced trace
        self.trace = QlibQuantTraceV2(scen=scen)
        logger.log_object(self.trace.controller, tag="Bandit Controller (V2)")
        
        # Initialize parent
        super(RDLoop, self).__init__()
        
        logger.info("=" * 80)
        logger.info("QuantRDLoopV2 initialized successfully")
        logger.info(f"Settings: {PROP_SETTING.model_dump()}")
        logger.info("=" * 80)
    
    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        """
        Generate hypothesis and experiment with action selection
        """
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                # Generate hypothesis (Bandit decides action internally)
                hypo = self._propose()
                
                # Ensure action is set
                assert hasattr(hypo, 'action') and hypo.action in ["factor", "model"], \
                    f"Invalid action: {getattr(hypo, 'action', 'missing')}"
                
                # Generate experiment based on action
                if hypo.action == "factor":
                    exp = self.factor_hypothesis2experiment.convert(hypo, self.trace)
                else:
                    exp = self.model_hypothesis2experiment.convert(hypo, self.trace)
                
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                
                return {"propose": hypo, "exp_gen": exp}
            
            await asyncio.sleep(1)
    
    def coding(self, prev_out: dict[str, Any]):
        """
        Code generation based on action
        """
        action = prev_out["direct_exp_gen"]["propose"].action
        
        if action == "factor":
            exp = self.factor_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        elif action == "model":
            exp = self.model_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        else:
            raise ValueError(f"Unknown action: {action}")
        
        logger.log_object(exp, tag="coder result")
        return exp
    
    def running(self, prev_out: dict[str, Any]):
        """
        Execute experiment based on action
        """
        action = prev_out["direct_exp_gen"]["propose"].action
        
        if action == "factor":
            coding_out = prev_out.get("coding")
            if coding_out is None:
                raise FactorEmptyError("Factor coding failed (no output).")
            exp = self.factor_runner.develop(coding_out)
            if exp is None:
                logger.error("Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
        elif action == "model":
            # 当 coding 失败时使用 direct_exp_gen 的 exp，model_runner 会用 fallback
            coding_out = prev_out.get("coding")
            exp_input = coding_out if coding_out is not None else prev_out["direct_exp_gen"]["exp_gen"]
            exp = self.model_runner.develop(exp_input)
        else:
            raise ValueError(f"Unknown action: {action}")
        
        logger.log_object(exp, tag="runner result")
        
        # Log metrics
        if exp.result:
            logger.info(f"Experiment results:")
            for key, value in exp.result.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
        
        return exp
    
    def feedback(self, prev_out: dict[str, Any]):
        """
        Generate feedback based on action (with cross-arm influence)
        """
        # Check for exception
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
        else:
            action = prev_out["direct_exp_gen"]["propose"].action
            
            try:
                if action == "factor":
                    feedback = self.factor_summarizer.generate_feedback(prev_out["running"], self.trace)
                elif action == "model":
                    feedback = self.model_summarizer.generate_feedback(prev_out["running"], self.trace)
                else:
                    raise ValueError(f"Unknown action: {action}")
            except (RuntimeError, ConnectionError, TimeoutError) as api_err:
                # LLM API 失败时使用 fallback，避免阻断流程，仍能完成 record 并产出指标
                err_str = str(api_err).lower()
                if any(x in err_str for x in ("chat completion", "retries", "connection", "timeout", "api")):
                    exp = prev_out["running"]
                    res = exp.result if exp and hasattr(exp, "result") and exp.result else {}
                    obs = "; ".join(f"{k}={v}" for k, v in (res or {}).items() if isinstance(v, (int, float)))
                    logger.warning(f"Feedback API failed ({api_err}), using fallback. Observations: {obs[:200]}")
                    feedback = HypothesisFeedback(
                        observations=obs or str(api_err),
                        hypothesis_evaluation="",
                        new_hypothesis="",
                        reason="LLM API unavailable, using fallback",
                        decision=True,
                    )
                else:
                    raise
        
        logger.log_object(feedback, tag="feedback")
        
        # Log decision
        logger.info(f"Feedback decision: {'ACCEPT' if feedback.decision else 'REJECT'}")
        if feedback.new_hypothesis:
            logger.info(f"Suggested next hypothesis: {feedback.new_hypothesis[:100]}...")
        
        return feedback
    
    def record(self, prev_out: dict[str, Any]):
        """
        Record experiment and feedback to trace
        """
        feedback = prev_out["feedback"]
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen", {}).get("exp_gen")
        
        # Record to trace
        self.trace.sync_dag_parent_and_hist((exp, feedback), prev_out[self.LOOP_IDX_KEY])
        
        # Log trace statistics
        factor_count = sum(1 for e, _ in self.trace.hist if hasattr(e.hypothesis, 'action') and e.hypothesis.action == "factor")
        model_count = sum(1 for e, _ in self.trace.hist if hasattr(e.hypothesis, 'action') and e.hypothesis.action == "model")
        
        logger.info(f"Trace statistics: {factor_count} factor experiments, {model_count} model experiments")


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    checkout_path: Optional[str] = None,
):
    """
    Enhanced Auto R&D Evolving loop for Quant with QuantaAlpha integration
    
    Args:
        path: Path to resume from (e.g., log/__session__/5/0_propose)
        step_n: Step number to resume from (optional)
        loop_n: Maximum number of loops to run
        all_duration: Maximum total duration (e.g., "4h", "30m")
        checkout: Whether to checkout to a clean state
        checkout_path: Custom checkout path
    
    Examples:
        # Start fresh
        python -m rdagent.app.qlib_rd_loop.quant_v2
        
        # Run for 30 loops max
        python -m rdagent.app.qlib_rd_loop.quant_v2 --loop_n 30
        
        # Run for 4 hours max
        python -m rdagent.app.qlib_rd_loop.quant_v2 --all_duration "4h"
        
        # Resume from checkpoint
        python -m rdagent.app.qlib_rd_loop.quant_v2 log/__session__/5/0_propose --step_n 1
    """
    import os
    
    # Ensure we're using the rdagent conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'rdagent':
        print(f"Warning: Current conda environment is '{conda_env}', expected 'rdagent'")
        print("Please activate the rdagent environment:")
        print("  conda activate rdagent")
        print("")
    
    if checkout_path is not None:
        checkout = Path(checkout_path)
    
    # Initialize loop
    if path is None:
        quant_loop = QuantRDLoopV2(QUANT_PROP_SETTING_V2)
    else:
        quant_loop = QuantRDLoopV2.load(path, checkout=checkout)
    
    # Run loop
    try:
        asyncio.run(quant_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))
    except KeyboardInterrupt:
        print("\nLoop interrupted by user")
        print("To resume, run:")
        session_path = quant_loop.session_folder
        print(f"  python -m rdagent.app.qlib_rd_loop.quant_v2 {session_path}/__session__/{quant_loop.loop_idx}/0_propose")


if __name__ == "__main__":
    fire.Fire(main)
