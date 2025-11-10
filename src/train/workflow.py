
from platoon.train.areal_integration import ArealLLMClient
from platoon.agents.code_issue_localization.rollout import run_single_rollout_process
from areal.utils.data import concat_padded_tensors
from platoon.train.areal_integration import areal_llm_clients

import asyncio
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch

class CodeIssueLocalizationArealWorkflow:
    def __init__(self, config: dict):
        self.config = config
        
    async def arun_episode(self, engine, data: dict) -> dict | None:
        try:
            client = ArealLLMClient(model=self.config['model_name'], engine=engine)
            loop = asyncio.get_event_loop()
            
            config = deepcopy(self.config)
            config['llm_client'] = client # TODO: Need to think of a more explicit arg passing story.
            
            args = (data, config)
            # Use spawn context to avoid forking when AReaL's workflow thread/event loop is active
            with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as executor:
                results = await loop.run_in_executor(executor, run_single_rollout_process, args)
            #results = await asyncio.to_thread(run_single_rollout_process, args) # TODO: Support multithreading instead of multiprocessing.
            
            areal_completion_data_list = []
            for trajectory in results['trajectories'].values():
                areal_completion_data_list.append(trajectory['misc']['areal_completion_data'])
            return concat_padded_tensors(areal_completion_data_list) | {
                'task_reward': torch.tensor(list(results['trajectories'].values())[0]['reward']).unsqueeze(0),
                'num_steps': torch.tensor([float(len(trajectory['steps'])) for trajectory in results['trajectories'].values()])
            }
        except Exception as e:
            print(f"Error in areal workflow: {e}")
            return None
