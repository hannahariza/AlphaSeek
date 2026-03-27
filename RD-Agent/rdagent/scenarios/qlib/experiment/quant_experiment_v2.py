"""
Enhanced Quant Experiment for RD-Agent V2 (Local Execution)

This module provides a version of QlibQuantScenario that supports
local execution without Docker.
"""

from copy import deepcopy
from pathlib import Path

from rdagent.app.qlib_rd_loop.conf_v2 import QUANT_PROP_SETTING_V2

# Factor
from rdagent.components.coder.factor_coder.config import get_factor_env
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)

# Model
from rdagent.components.coder.model_coder.conf import get_model_env
from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class QlibFactorExperimentV2(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")


class QlibModelExperimentV2(ModelExperiment[ModelTask, QlibFBWorkspace, ModelFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "model_template")


class QlibQuantScenarioV2(Scenario):
    """
    Enhanced Quant Scenario with local execution support (non-Docker)
    """
    
    def __init__(self, use_docker: bool = False) -> None:
        super().__init__()
        # Use local execution by default (pass use_docker=False)
        self._source_data = deepcopy(get_data_folder_intro(use_docker=use_docker))

        self._rich_style_description = deepcopy(T(".prompts:qlib_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(
            T(".prompts:qlib_factor_experiment_setting").r(
                train_start=QUANT_PROP_SETTING_V2.train_start,
                train_end=QUANT_PROP_SETTING_V2.train_end,
                valid_start=QUANT_PROP_SETTING_V2.valid_start,
                valid_end=QUANT_PROP_SETTING_V2.valid_end,
                test_start=QUANT_PROP_SETTING_V2.test_start,
                test_end=QUANT_PROP_SETTING_V2.test_end,
            )
        )

    def background(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        quant_background = "The background of the scenario is as follows:\n" + T(".prompts:qlib_quant_background").r(
            runtime_environment=self.get_runtime_environment(),
        )
        factor_background = "The factor background is as follows:\n" + T(".prompts:qlib_factor_background").r(
            runtime_environment=self.get_runtime_environment(),
        )
        model_background = "The model background is as follows:\n" + T(".prompts:qlib_model_background").r(
            runtime_environment=self.get_runtime_environment(),
        )

        if tag is None:
            return quant_background + "\n" + factor_background + "\n" + model_background
        elif tag == "factor":
            return quant_background + "\n" + factor_background
        elif tag == "model":
            return quant_background + "\n" + model_background
        else:
            raise ValueError(f"tag {tag} is not supported")

    @property
    def source_data(self) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return T(".prompts:qlib_factor_output_format").r()

    @property
    def interface(self) -> str:
        return T(".prompts:qlib_factor_interface").r()

    @property
    def simulator(self) -> str:
        return T(".prompts:qlib_factor_simulator").r()

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        # Handle optional parameters that original Scenario may not support
        try:
            return super().get_scenario_all_desc(task, filtered_tag, simple_background)
        except TypeError:
            # Fallback for compatibility
            return f"""Background of the scenario:
{self.background}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your model:
{self.simulator}
"""

    def get_runtime_environment(self):
        factor_env = get_factor_env()
        model_env = get_model_env()
        factor_stdout = get_runtime_environment_by_env(env=factor_env)
        model_stdout = get_runtime_environment_by_env(env=model_env)
        return f"Factor Environment:\n{factor_stdout}\n\nModel Environment:\n{model_stdout}"
