# LLM baseline (GPT-5.6-Luna)

This folder contains the LLM baseline reported in the paper as **GPT-5.6-Luna**.
For each test instance, the model receives the full EVRPTW instance as text and
returns routes in JSON. The routes are then simulated by `routes_to_result` in
`chatgpt_helper.py`, which uses the same vehicle, battery, charging and time
parameters as the RL environment (read from `config["env"]`), to compute service
rate, distance, battery and time-window metrics. `EVRPEnv` itself is only used to
draw the example route figures.

| File | Content |
|---|---|
| `chatgpt.py` | Experiment driver: instance selection, API calls, metric computation, example plots, resume |
| `chatgpt_helper.py` | Prompt construction (`build_prompts`), response → route conversion, metric aggregation |
| `prompt_example/` | One complete request/response pair from the paper run (see below) |
| `.env.example` | Template for the API key file |

## Prompts

`build_prompts` in `chatgpt_helper.py` builds two messages per instance:

- **System prompt**: a fixed template describing the vehicle, battery, charging
  and time rules and the objective priority, with the instance-size parameters
  filled in (number of vehicles, `battery_max`, charging threshold, operating
  interval).
- **User prompt**: the instance itself, meaning every node (depot, customers, charging
  stations) with coordinates, time window and service time.

`prompt_example/` is graph 1513 from the n = 50 paper run, copied unchanged:

| File | Content |
|---|---|
| `system_prompt.txt` | System prompt sent to the model |
| `user_prompt.txt` | User prompt (full instance) |
| `raw_response.txt` | Raw model output |
| `response.json` | Parsed routes |
| `metadata.json` | Model name and token usage |

The template is identical for all sizes; only the filled-in parameters differ:

| Size | Vehicles | `battery_max` | Charge to | Operating interval |
|---|---|---|---|---|
| n = 50 | 2 | 60 | 48 | [240, 1000] |
| n = 100 | 3 | 60 | 48 | [240, 1000] |
| n = 200 | 3 | 100 | 80 | [240, 1320] |

## Paper settings

Model `gpt-5.6-luna`, 100 test instances per size, seed 42. Reasoning effort per size:

| Size | Reasoning effort |
|---|---|
| n = 50 | low |
| n = 100 | medium |
| n = 200 | high |

Only the n = 50 config is provided as an example:
`GPT_Luna_Baseline/configs/GPT_Luna_example_config.yaml`. To run another size, copy
the `env` section from that size's RL config in `DRL_results/` (see the
parameter table above), add `station_allow_battery_ratio: 0.5` (the `EVRPEnv`
default, which the RL configs do not list but the prompt needs), and set
`dataset.test_csv` and `llm_baseline.reasoning_effort` accordingly.

The config points to `example_dataset/` (10 instances), so running it uses only
those 10 test instances; the paper used 100 instances sampled from the full test
set.

## Running

```bash
cp GPT_Luna_Baseline/.env.example GPT_Luna_Baseline/.env    # then put your OPENAI_API_KEY in .env

# from the repository root
python -m GPT_Luna_Baseline.chatgpt GPT_Luna_Baseline/configs/GPT_Luna_example_config.yaml --estimate_only   # token estimate, no API calls
python -m GPT_Luna_Baseline.chatgpt GPT_Luna_Baseline/configs/GPT_Luna_example_config.yaml                   # full run
```

`.env` is git-ignored. A variable already exported in the shell takes precedence
over the value in `.env`.
