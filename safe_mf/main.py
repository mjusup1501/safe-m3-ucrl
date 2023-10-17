import os
import sys
import warnings
import yaml
import torch
from datetime import datetime
from safe_mf.utils.parser import parser
from pathlib import Path
import torch

import wandb
from safe_mf.alg.safe_mf_marl import SafeMFMARL

warnings.filterwarnings("ignore")
CONFIG_EXCLUDE_KEYS = ["logging"]

parser.add_argument("--config", type=str)

def main():
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_id = Path(args.config).stem 
    if config["run_id"] is None:
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        run_id = f"{run_id}_{now}"
    else:
        suffix = config["run_id"]
        run_id = f"{run_id}_{suffix}"
    wandb_id = f"{run_id}_{config['exec_type']}"
    log_dir = Path(config["logdir"]) / run_id
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(
        project="safe-m3-ucrl",
        id=wandb_id,
        dir=log_dir,
        entity=config.get('entity', 'user'),
        save_code=False,
        config=config,
        config_exclude_keys=CONFIG_EXCLUDE_KEYS,
        mode=config.get("logging", "disabled"),
    )

    if config['device'] == 'cuda:0' and torch.cuda.is_available():
        device = config["device"]
    else:
        device = "cpu"

    alg = SafeMFMARL(
            env_name=config.get("env_name", "vehicle-repositioning-sequential"),
            **config["model"], 
            delta=1 / config["training"]["horizon"],
            device=device,
            log_dir=log_dir,
            dynamics_ckpt=config.get("dynamics_ckpt", None),
            policy_ckpt=config.get("policy_ckpt", None),
            exec_type=config.get("exec_type", "train"),
    )

    alg.run(**config["training"])

if __name__ == "__main__":
    main()
