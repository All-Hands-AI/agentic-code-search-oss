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
import logging

logger = get_logger(__name__)
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.ERROR)

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
    workspace = Path("testbed/")
    status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace)
    print("working_dir:", working_dir)
    working_dir = Path.cwd() / working_dir

    agent = None
    result = None
    patch = ""
    reward = 0
    error = None
    eval_error = None
    messages = []
    def conversation_callback(event: Event):
        if isinstance(event, LLMConvertibleEvent):
            messages.append(event.to_llm_message())

    agent = get_default_agent(
        # llm=llm,
        llm=LLM(
            service_id="agent",
            model=litellm_model_name,
            base_url=f"http://localhost:8080/v1/",
            api_key=os.getenv("API_KEY"),
        ),
        cli_mode=True,
    )

    conversation = Conversation(
        agent=agent,
        max_iteration_per_run=30,
        visualize=True,
        workspace=str(working_dir),
        callbacks=[conversation_callback],
    )
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "src", "prompts", "templates", "file_localization.j2")
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

    final_message = conversation.agent_final_response()

    # Do reward here

    processed_messages = [
        # system_messages,
        {"role": "user", "content": input_message},
    ]
    for idx, message in enumerate(messages):
        full_text = ""
        try:
            if message.role == "assistant":
                role = "assistant"
                if len(message.content) != 0:
                    full_text += message.content[0].text
                if len(message.tool_calls) != 0:
                    tool_name = message.tool_calls[0].name
                    tool_args = ast.literal_eval(message.tool_calls[0].arguments)
                    if tool_name == "finish":
                        full_text += tool_args["message"]
                        break
                    else:
                        full_text += "\n\n" + f"<function={tool_name}>"
                        for k, v in tool_args.items():
                            full_text += f"\n<parameter={k}>{v}</parameter>\n"
                        full_text += "</function>\n"
            elif message.role == "tool":
                role = "user"
                if len(message.content) != 0:
                    full_text += message.content[0].text
            else:
                continue
        except Exception as e:
            # print("message", message)
            logger.error(f"Error processing message {idx}: {e}", exc_info=True)
            continue

        processed_messages.append({"role": role, "content": full_text})

    print("Evaluation result:", reward)
    return (processed_messages, reward, error)


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
        instance = env_extras["instance"]
        messages, reward, error = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            # sweagent_config,
            self.base_url,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )
        # TODO Properly handle the right system prompt.
        
        input_prompt = messages[0]
        initial_input_ids = self.tokenizer.apply_chat_template(
            input_prompt, add_generation_prompt=False, tokenize=True
        )
        initial_prompt_length = len(initial_input_ids)

        response_ids: List[int] = []
        loss_mask: List[int] = []

        # print("=" * 100)
        # print("Conversation finished. Got the following LLM messages:")
        # for i, message in enumerate(processed_messages):
        #     print(f"Message {i}: {str(message)[:200]}")

        for message in messages:
            # Apply chat template and tokenize each message
            msg_encoding = encode_messages_subset([message], self.tokenizer)

            # Extend response_ids with the tokens
            response_ids.extend(msg_encoding)

            # Extend loss_mask: 0s for user, 1s for assistant
            if message["role"] in ["user", "tool"]:
                loss_mask.extend([0] * len(msg_encoding))
            else:  # assistant
                loss_mask.extend([1] * len(msg_encoding))
        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

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
