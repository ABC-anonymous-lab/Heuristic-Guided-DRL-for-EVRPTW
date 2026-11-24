# Heuristic-Guided Deep Reinforcement Learning for Electric Vehicle Routing with Time Windows

Code for the paper ***Heuristic-Guided Deep Reinforcement Learning for Electric Vehicle Routing with Time Windows***. A fleet of electric vehicles leaves a depot, serves customers inside their time windows, recharges at charging stations when needed, and returns to the depot before the end of the operating interval. The objective is to serve as many customers as possible with as little travel distance as possible. The repository contains four deep reinforcement learning (DRL) methods, their HG variants (DRL guided by an OR-Tools Guided Local Search repair, used as a behavior-cloning teacher during training), and two baselines (OR-Tools and an LLM).

| Method (paper name) | Code | Trainer class |
|---|---|---|
| PN-LSTM | `algorithm/PN_LSTM/PN_LSTM.py` | `PNLSTMTrainer` |
| HG-PN-LSTM | `algorithm/PN_LSTM/HG_PN_LSTM.py` | `HGPNLSTMTrainer` |
| GAT-Transformer | `algorithm/GAT_Transformer/GAT_Transformer.py` | `GATTransformerTrainer` |
| HG-GAT-Transformer | `algorithm/GAT_Transformer/HG_GAT_Transformer.py` | `HGGATTransformerTrainer` |
| PPO | `algorithm/PPO/PPO.py` | `PPOTrainer` |
| HG-PPO | `algorithm/PPO/HG_PPO.py` | `HGPPOTrainer` |
| DDQN | `algorithm/DDQN/DDQN.py` | `DDQNTrainer` |
| HG-DDQN | `algorithm/DDQN/HG_DDQN.py` | `HGDDQNTrainer` |
| OR-Tools (baseline) | `OR_Tools_baseline/` | — |
| GPT-5.6-Luna (baseline) | `GPT_Luna_Baseline/` | — |

## Repository layout

```
main.py                         Entry point for DRL training + testing
algorithm/
  base_trainer.py               Shared training loop (validation, checkpoints, testing)
  PN_LSTM/  GAT_Transformer/  PPO/  DDQN/     One folder per method family (plain + HG)
  OR_Tools_GLS_guidance/        OR-Tools GLS repair used by the HG methods (subprocess pool)
models/                         Policy / Q networks (GAT encoder + pointer decoders, PN-LSTM)
env/
  EVRP.py                       EVRPTW environment (EVRPEnv)
  mask_function.py              Feasible-action mask
  reward_function.py            Reward
utils/                          Config → objects, dataloader, logger, checkpoints, plots
DRL_results/                    Configs and test results of every DRL experiment in the paper
example_dataset/                10 example instances per split and size
OR_Tools_baseline/              OR-Tools baseline (see below)
GPT_Luna_Baseline/              LLM baseline (see GPT_Luna_Baseline/README.md)
synthetic_dataset_generation/   Script used to generate the synthetic instances
environment.yml                 Conda environment
```

## Installation

```bash
conda env create -f environment.yml
conda activate HG_DRL_EVRPTW
```

The environment installs PyTorch with CUDA 12.8 wheels. A GPU is used when
available; otherwise training runs on the CPU.

## Data

The full datasets used in the paper are not included. `example_dataset/`
contains, for each problem size, the first 10 instances of the original
training, validation and test files, so that every config in this repository
runs end to end:

```
example_dataset/
  n50/   train/evrp_example_training_dataset.csv
         val/evrp_example_validation_dataset.csv
         test/evrp_example_test_dataset.csv
  n100/  ...
  n200/  ...
```

| Size | Customers | Charging stations | Nodes per instance |
|---|---|---|---|
| n50 | 50 | 1 | 52 |
| n100 | 100 | 2 | 103 |
| n200 | 200 | 4 | 205 |

Each row is one node; rows with the same `graph_id` form one instance.

| Column | Meaning |
|---|---|
| `graph_id`, `node_id` | Instance id, node index (node 0 is the depot) |
| `type` | 0 = depot, 1 = customer, 2 = charging station |
| `x`, `y` | Coordinates in the unit square |
| `x_km`, `y_km` | Coordinates in km (used for distances) |
| `time_slot` | Time window of the customer (1, 2, ... in `env.time_windows`; 0 for depot and stations) |
| `ready_time`, `due_date` | Time window in minutes |
| `ready_time_norm`, `due_date_norm` | Time window normalized to [0, 1] |
| `service_time` | Service time in minutes |
| `demand` | Only in the n50 / n100 files; not used (the problem has no load capacity) |

`synthetic_dataset_generation/gen_dataset.py` generates new instances in the
same format: raw customer locations from the R package `tspgen`
(`gen_raw_data.R`), then the depot (k-means center), charging stations (k-means),
scaling to km, time windows and the train/val/test split. The example config
reproduces the n200 setting of the paper (the n50 / n100 settings are described in it):

```bash
python synthetic_dataset_generation/gen_dataset.py synthetic_dataset_generation/dataset_config_example.yaml
```

## Running the DRL methods

Every experiment is described by a single YAML config. To train and test one
method:

```bash
python main.py DRL_results/ID08_n50_HG-PPO_w1.0/seed42/config.yaml
```

Several configs can be given at once, and `--seeds` overrides the seed in the
configs:

```bash
python main.py DRL_results/ID07_n50_PPO_w1.0/seed42/config.yaml DRL_results/ID08_n50_HG-PPO_w1.0/seed42/config.yaml
python main.py DRL_results/ID08_n50_HG-PPO_w1.0/seed42/config.yaml --seeds 21 42 84
```

Main config sections:

| Section | Content |
|---|---|
| `dataset` | Train / validation / test files; `graph_fraction` = fraction of instances used from each file |
| `train` | Trainer class (`trainer_class_path`) and its hyperparameters; HG methods also set the OR-Tools GLS repair (`use_gls_guided`, `repair_file_name`, `ortools_repair_time_limit`, `ortools_num_workers`) and the behavior-cloning loss (`bc_max_weight`, `bc_min_weight`, ...) |
| `model` | Network class (`class_path`), its arguments and the optimizer |
| `env` | Environment class, mask and reward functions, and the problem parameters (vehicles, battery, charging, speed, operating interval, time windows, `distance_weight` in the reward) |
| `logger` | Output directory and logging / plotting options |

During training the model is validated every `eval_interval` steps. Two best
checkpoints are kept, one by validation reward and one by validation distance,
each only among validations whose service rate reaches
`logger.min_served_threshold` (otherwise the best checkpoint without this
condition is used, and a warning is printed). After training, both are evaluated
on the test set. Outputs are written to `outputs/<size>/<run_group>/seed<seed>/`
(TensorBoard logs to `outputs/runs/`):

```
checkpoints/{reward,total_distance,fallback}/   Best checkpoints
results/                                        Train / validation / test metric CSVs, timing, code snapshot
result_analyzes/test_results/reward/test_summary.csv          Test result of the best-reward checkpoint (used in the paper)
result_analyzes/test_results/total_distance/test_summary.csv  Test result of the best-distance checkpoint
result_analyzes/{reward,total_distance}/graph_<id>/           Route, battery and timeline plots for sample test instances
```

## Paper results (`DRL_results/`)

Each DRL experiment of the paper has one folder, named after its ID in the
paper tables:

```
DRL_results/ID<id>_n<size>_<method>_w<distance_weight>/
  seed21/  config.yaml  test_summary.csv
  seed42/  config.yaml  test_summary.csv
  seed84/  config.yaml  test_summary.csv
```

- `config.yaml` is the exact configuration of that run. Only the data paths
  were changed (they now point to `example_dataset/`) and `graph_fraction` was
  set to 1.0, so the configs run as-is on the example data.
- `test_summary.csv` is the test result reported in the paper (best-reward
  checkpoint), computed on the original, unreleased test set.
- The paper reports, for each ID, the mean ± sample standard deviation over the
  three seeds of `served_count` (service rate, shown as %) and `total_distance`.

Results obtained with `example_dataset/` are not comparable to the paper
numbers, because the example data contains only 10 instances per split.

## Baselines

**OR-Tools.** The same state-expanded routing model as the HG repair, solved
with Guided Local Search under a time limit per instance (`ortools.time_limit_sec`).

```bash
python OR_Tools_baseline/run_evrp_ortools.py --config OR_Tools_baseline/configs/OR_Tools_example_config.yaml
```

The example config is for n50. For another size, copy the `env` section from
that size's config in `DRL_results/` and change `ortools.data_path`. Results
are written to `OR_Tools_baseline/results/<size>/<timestamp>/`. The search is
deterministic, but instances that have not converged when the time limit is
reached can give slightly different results between runs.

**GPT-5.6-Luna.** See [`GPT_Luna_Baseline/README.md`](GPT_Luna_Baseline/README.md)
for the prompts, an example request/response and how to run it (an OpenAI API
key is required).
