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
import shutil
from pydantic import SecretStr

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
from openhands.tools.preset.planning import get_planning_tools
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
import signal

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
    import os
    import shutil
    
    instance_id = instance["instance_id"]
    repo_name = instance["repo"]
    commit_id = instance["base_commit"]
    
    worker_id = os.getpid()
    workspace = Path(f"/data/user_data/sanidhyv/tmp/testbed_{worker_id}/")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Track what we created for cleanup
    created_workspace = False
    created_index = False
    index_path = None
    
    try:
        status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace)
        created_workspace = True
        print(f"[Worker {worker_id}] working_dir: {working_dir}")

        if training_phase == "eval":
            temperature = 0.6
        else:
            temperature = 1.0

        agent = None
        final_message = ""
        reward = 0
        reward_dict = {}
        error = None
        messages = []

        # Configure semantic search if enabled
        use_semantic_search = True
        mcp_config = None
        agent_context = None

        if use_semantic_search:
            from openhands.sdk.context.skills import Skill
            from openhands.sdk import AgentContext

            # Get base path
            base_path = Path(generator_cfg.get("base_path", "/home/sanidhyv/agentic-code-search-oss"))
            skill_path = base_path / ".openhands" / "skills" / "semantic-search.md"
            
            if not skill_path.exists():
                print(f"[Episode {instance_id}] Warning: Skill file not found, semantic search disabled")
                use_semantic_search = False
            else:
                skill = Skill.load(str(skill_path))
                wrapper_path = base_path / "run_mcp_server_training.sh"
                
                if not wrapper_path.exists():
                    print(f"[Episode {instance_id}] Warning: MCP wrapper not found")
                    use_semantic_search = False
                else:
                    import stat
                    wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC)
                    
                    # Track index location for cleanup
                    from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
                    repo_commit_hash = get_repo_commit_hash(repo_name, commit_id)
                    index_path = Path(f"/data/user_data/sanidhyv/tmp/embedding_cache/{repo_commit_hash}")
                    
                    mcp_config = {
                        "mcpServers": {
                            "semantic-code-search": {
                                "command": "bash",
                                "args": [str(wrapper_path)],
                                "env": {
                                    "WORKSPACE_PATH": str(working_dir),  # This should be the cloned repo, NOT "."
                                    "RAY_ADDRESS": os.environ.get("RAY_ADDRESS", "auto"),
                                    "PYTHONPATH": str(base_path),
                                }
                            }
                        }
                    }
                    
                    agent_context = AgentContext(skills=[skill])
                    print(f"[Episode {instance_id}] Semantic search enabled")
                    
                    # Mark that we'll create an index
                    if not index_path.exists():
                        created_index = True

        # Agent creation
        agent_kwargs = {
            "llm": LLM(
                service_id="agent",
                model=litellm_model_name,
                base_url=f"http://localhost:8080/v1/",
                api_key=SecretStr("dummy"),
                temperature=temperature,
                litellm_extra_body={
                    "return_token_ids": True,
                    "include_stop_str_in_output": True,
                }
            ),
            "tools": get_default_tools(enable_browser=False),
            "security_analyzer": None,
        }
        
        if use_semantic_search and mcp_config is not None and agent_context is not None:
            agent_kwargs["agent_context"] = agent_context
            agent_kwargs["mcp_config"] = mcp_config
        
        agent = Agent(**agent_kwargs)

        conversation = Conversation(
            agent=agent,
            max_iteration_per_run=8,
            visualizer=None,
            workspace=str(working_dir),
        )
        
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "templates", "file_localization.j2")
        input_message = get_instruction(instance, prompt_path, str(working_dir))
        
        # Truncate input if too long
        from transformers import AutoTokenizer
        MAX_INPUT_TOKENS = 12000
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                litellm_model_name,
                trust_remote_code=True,
                cache_dir="/data/user_data/sanidhyv/.cache/huggingface"
            )
            input_tokens = tokenizer.encode(input_message)
            
            if len(input_tokens) > MAX_INPUT_TOKENS:
                print(f"[Episode {instance_id}] Input too long ({len(input_tokens)} tokens), truncating")
                input_tokens = input_tokens[:500] + input_tokens[-(MAX_INPUT_TOKENS-500):]
                input_message = tokenizer.decode(input_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"[Episode {instance_id}] Could not check input length: {e}")
        
        conversation.send_message(input_message)
        print("Starting conversation...")
        
        try:
            class TimeoutError(Exception):
                pass

            def timeout_handler(signum, frame):
                raise TimeoutError("Operation timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60*10)  # 10 minute timeout

            logger.info("Conversation Starting")
            conversation.run()
        except TimeoutError as e:
            logger.error(f"Conversation timed out: {e}")
            error = f"Timeout: {str(e)}"
        except Exception as e:
            logger.error(f"Error in conversation: {e}", exc_info=True)
            error = str(e) + "\n" + traceback.format_exc()
        finally:
            signal.alarm(0)
            conversation.close()
            logger.info("Conversation Finished")

            messages = list(map(lambda event: event.model_dump(), conversation.state.events))
            
            # Truncate messages
            MAX_MESSAGES = 100
            if len(messages) > MAX_MESSAGES:
                original_len = len(messages)
                messages = messages[-MAX_MESSAGES:]
                print(f"[Episode {instance_id}] Truncated messages from {original_len} to {MAX_MESSAGES}")
            
            final_message = get_agent_final_response(conversation.state.events)

        try:
            reward_file = file_localization_f1_reward(final_message, instance)

            def multiturn_reward(messages):
                token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
                if len(token_messages) > 1:
                    return 1.0
                return 0.0

            reward_multiturn = multiturn_reward(messages)
            reward = reward_file + reward_multiturn
            reward_dict = {"file_localization_f1": reward_file, "multiturn_reward": reward_multiturn}
        except Exception as e:
            reward = 0.0
            reward_dict = {"file_localization_f1": 0.0, "multiturn_reward": 0.0}
        
        return (
            (messages, final_message),
            (reward, reward_dict),
            error,
        )
    
    finally:
        # Cleanup workspace and index
        print(f"[Worker {worker_id}] Cleaning up episode {instance_id}")
        
        # 1. Clean up workspace (repo clone)
        if created_workspace:
            try:
                shutil.rmtree(workspace, ignore_errors=True)
                print(f"[Worker {worker_id}] Removed workspace: {workspace}")
            except Exception as e:
                print(f"[Worker {worker_id}] Warning: Could not remove workspace: {e}")
        
        # 2. Clean up semantic search index (if we created it)
        if created_index and index_path and index_path.exists():
            try:
                shutil.rmtree(index_path, ignore_errors=True)
                print(f"[Worker {worker_id}] Removed index: {index_path}")
            except Exception as e:
                print(f"[Worker {worker_id}] Warning: Could not remove index: {e}")
                
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
        try:
            (messages, final_message), (reward, reward_dict), error = await init_and_run.remote(
                env_extras,
                self.litellm_model_name,
                self.base_url,
                self.generator_cfg,
                "swe-gym",
                sampling_params,
                trajectory_id,
                batch_metadata.global_step,
                batch_metadata.training_phase,
            )

            token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]

            stop_reason = "complete"
            prompt_ids_list = []
            response_ids_list = []
            trajectory_ids_list = []
            loss_mask = []
            initial_input_len = 0
            past_trajectory_len = 0
            
            for idx, message in enumerate(token_messages):
                current_prompt_ids = message["prompt_token_ids"]
                current_response_ids = message["response_token_ids"]

                prompt_ids_list.append(current_prompt_ids)
                response_ids_list.append(current_response_ids)
                trajectory_ids_list.append(current_prompt_ids + current_response_ids)

                if idx == 0:
                    initial_input_ids = current_prompt_ids
                    initial_input_len = len(initial_input_ids)
                    loss_mask = [1] * len(current_response_ids)
                    continue

                past_trajectory_len = len(trajectory_ids_list[idx-1])
                past_response_len = len(response_ids_list[idx-1])
                current_prompt_len = len(current_prompt_ids)
                current_response_len = len(current_response_ids)

                past_response_observation_ids = current_prompt_ids[past_trajectory_len:]
                past_response_observation_len = len(past_response_observation_ids)
                loss_mask.extend([0] * past_response_observation_len)
                loss_mask.extend([1] * current_response_len)
            
            response_ids = current_prompt_ids[initial_input_len:] + current_response_ids
            assert len(response_ids) == len(loss_mask), f"Response ids length {len(response_ids)} != loss mask length {len(loss_mask)}"

        except Exception as e:
            logger.error(f"Error in code_search_loop: {e}", exc_info=True)
            error = str(e) + "\n" + traceback.format_exc()
            reward = 0
            response_ids = [151643]
            stop_reason = "error"
            loss_mask = [1]
            initial_input_ids = [151643]
            final_message = ""
            reward_dict = {"file_localization_f1": 0.0, "multiturn_reward": 0.0}

        path = Path(self.generator_cfg.traj_dir) / f"step_{batch_metadata.global_step}" / batch_metadata.training_phase
        path.mkdir(parents=True, exist_ok=True)
        instance_id = env_extras["instance_id"]

        if error is not None:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.error"
            filename_path = path / filename
            print(f"Saving error to {filename_path}")
            with open(filename_path, "w") as f:
                f.write(error)
        else:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
            filename_path = path / filename

            result_dict = {
                "target": env_extras["target"],
                "final_message": final_message,
                "reward_dict": reward_dict,
                "messages": messages,
            }

            print(f"Saving trajectory to {filename_path}")
            with open(filename_path, "w") as f:
                json.dump(result_dict, f, indent=2)

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