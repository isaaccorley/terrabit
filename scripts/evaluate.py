"""Evaluation entrypoint."""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    main()
