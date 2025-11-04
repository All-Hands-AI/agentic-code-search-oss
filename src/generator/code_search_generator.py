import json
import asyncio
from socket import timeout
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid
from omegaconf import DictConfig
import traceback
import ray
import requests
from pathlib import Path
import os
import ast

import signal
from contextlib import contextmanager

from skyrl_train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    GeneratorOutput,
    GeneratorInput,
)
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    encode_messages_subset,
)
from openhands.tools.preset.default import get_default_agent

from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.sdk import (
    Agent,
    LLM,
    Event,
    Conversation,
    RemoteConversation,
    LLMConvertibleEvent,
    get_logger,
)

from src.prompts.prompt_builder import get_instruction
from src.utils.instance import clone_instance
from src.rewards.file_localization import file_localization_f1_reward
import logging

logger = get_logger(__name__)
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.ERROR)

file_path = os.path.dirname(__file__)

@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    litellm_base_url: dict,
    generator_cfg: DictConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: Union[TrajectoryID, Any],
    global_step: int,
    training_phase: Union[TrainingPhase, Any],
):

    instance_id = instance["instance_id"]
    repo_name = instance["repo"]
    commit_id = instance["base_commit"]
    workspace = Path("/tmp/testbed/")
    status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace)

    if training_phase == "eval":
        temperature = 0.6
    else:
        temperature = 1.0

    agent = None
    final_message = ""
    reward = 0
    error = None
    messages = []

    agent = get_default_agent(
        llm=LLM(
            service_id="agent",
            model=litellm_model_name,
            base_url=f"http://localhost:8080/v1/",
            api_key=os.getenv("API_KEY"),
            temperature=temperature,
        ),
        cli_mode=True,
    )

    conversation = Conversation(
        agent=agent,
        max_iteration_per_run=4,
        visualize=False,
        workspace=str(working_dir),
    )
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "templates", "file_localization.j2")
    input_message = get_instruction(instance, prompt_path, str(working_dir))
    conversation.send_message(input_message)
    print("Starting conversation...")
    try:
        logger.info("Conversation Starting")
        conversation.run()
    except Exception as e:
        logger.error(f"Error is sending conversation: {e}", exc_info=True)
    finally:
        conversation.close()
        logger.info("Conversation Finished")

        messages = list(map(lambda event: event.model_dump(), conversation.state.events))
        final_message = get_agent_final_response(conversation.state.events)

        path = Path(generator_cfg.traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.jsonl"
        path = path / filename

        print(f"Saving trajectory to {path}")
        with open(path, "w") as f:
            f.writelines(str(msg) + "\n" for msg in messages)

    try:
        reward = file_localization_f1_reward(final_message, instance)
    except Exception as e:
        reward = 0.0

    return (messages, reward, error)


class CodeSearchGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        # Call parent constructor first
        super().__init__(
            generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name
        )

        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_server_inference_engine_client_port", 8000
        )
        self.base_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        # self.litellm_model_name = "openai/" + self.model_name
        self.litellm_model_name = "hosted_vllm/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError(
                "OpenhandsGenerator doesn't support custom chat template"
            )

    async def code_search_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:
        # sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        # instance = env_extras["instance"]
        try:
            messages, reward, error = await init_and_run.remote(
                env_extras,
                self.litellm_model_name,
                # sweagent_config,
                self.base_url,
                self.generator_cfg,
                # env_extras["data_source"],
                "Swe-Gym",
                sampling_params,
                trajectory_id,
                batch_metadata.global_step,
                batch_metadata.training_phase,
            )
        except Exception as e:
            # TODO properly handle this
            reward = 0
            messages = [{"kind": "TokenEvent", "prompt_tokens_ids": [151643], "response_token_ids": [151643]}]

        print("=" * 100)
        print("Conversation finished. Got the following LLM messages:")
        for i, message in enumerate(messages):
            print(f"Message {i}: {str(message)[:200]}")

        messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]

        stop_reason = "complete"
        prompt_ids_list = []
        response_ids_list = []
        loss_mask = []
        initial_input_len = 0
        past_trajectory_len = 0
        for idx, message in enumerate(messages):
            current_prompt_ids = message["prompt_tokens_ids"]
            current_response_ids = message["response_token_ids"]

            prompt_ids_list.append(current_prompt_ids)
            response_ids_list.append(current_response_ids)

            if idx == 0:
                initial_input_ids = current_prompt_ids
                initial_input_len = len(initial_input_ids)
                # TODO properly handle max tokens
                # max_response_tokens = max_tokens + max_input_length - initial_input_len

            trajectory_ids = current_prompt_ids[initial_input_len:]
            past_response_len = len(response_ids_list[idx-1]) if idx > 0 else 0
            past_response_observation_ids = trajectory_ids[past_trajectory_len+past_response_len:]
            past_response_observation_len = len(past_response_observation_ids)
            loss_mask.extend([1] * past_response_len + [0] * past_response_observation_len)
            past_trajectory_len = len(trajectory_ids)

            # response_ids = trajectory_ids + current_response_ids
            # if len(response_ids) >= max_response_tokens:
            #     response_ids = response_ids[:max_response_tokens]
            #     loss_mask.extend([1] * len(current_response_ids))
            #     loss_mask = loss_mask[:max_response_tokens]
            #     stop_reason = "length"
            #     break
        
        response_ids = trajectory_ids + current_response_ids
        loss_mask.extend([1] * len(current_response_ids))

        return (response_ids, reward, stop_reason, loss_mask, initial_input_ids, None)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.backend, self.generator_cfg.sampling_params
        )

        tasks = []

        for i in range(len(prompts)):
            rollout = self.code_search_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            
            if True:
                tasks.append(rollout)
            else:
                tasks.extend(rollout)
                

        all_outputs = await asyncio.gather(*tasks)

        # Filter out the `None` entries, which means that trajectory generation failed
        responses = [output[0] for output in all_outputs if output[0] is not None]
        rewards = [output[1] for output in all_outputs if output[0] is not None]
        stop_reasons = [output[2] for output in all_outputs if output[0] is not None]
        loss_masks = [output[3] for output in all_outputs if output[0] is not None]
        prompt_token_ids = [
            output[4] for output in all_outputs if output[0] is not None
        ]
        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output
