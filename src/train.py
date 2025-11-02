import hydra
from omegaconf import DictConfig, OmegaConf
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
import ray

from src.generator.code_search_generator import CodeSearchGenerator


class CodeSearchPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        generator = CodeSearchGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=OmegaConf.create({"max_env_workers": 0}),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
        )
        return generator


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = CodeSearchPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
