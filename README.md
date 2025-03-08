# Safe Model-Based Multi-Agent Mean-Field Reinforcement Learning Repository

To install required packages, run `pip install -e .`

In `configs` directory you can find config files for vehicle repositioning and swarm environments.

To run an experiment:
1. It is necessary to modify *logdir* field in a config file
2. For vehicle repositioning you must provide the demand and origin-destination matrices and modify *input_data_path* in a config file. This step is not necessary for the swarm environment. 
2. Optionally modify other fields, including *entity* for wandb logging or use pre-trained checkpoints that you can find in `checkpoints` directory
3. Execute `python safe_mf/main.py --config path/to/config.yaml`

[![DOI](https://zenodo.org/badge/655903764.svg)](https://doi.org/10.5281/zenodo.10431635)
