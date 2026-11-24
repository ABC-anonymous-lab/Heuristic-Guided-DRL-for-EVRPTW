# GPT-5.6-Luna baseline: experiment driver.

# Selects the test instances, sends each one to the OpenAI API, evaluates the
# returned routes (routes_to_result in chatgpt_helper.py), saves per-graph metrics,
# a summary and, for a few example graphs, the prompts, response and route plots.
# A run can be resumed with --resume.

# Usage:
#     python -m GPT_Luna_Baseline.chatgpt GPT_Luna_Baseline/configs/GPT_Luna_example_config.yaml --estimate_only
#     python -m GPT_Luna_Baseline.chatgpt GPT_Luna_Baseline/configs/GPT_Luna_example_config.yaml

import os
import json
import argparse

import yaml
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from utils.init_model import build_env_builder
from utils.plot_functions import plot_all_vehicle_battery_and_distance, plot_all_vehicle_figures
from GPT_Luna_Baseline.chatgpt_helper import (
    build_prompts,
    estimate_tokens,
    routes_to_result,
    extract_metrics,
    summarize_metrics,
    save_example_summaries,
    save_prompt_example,
)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
FATAL_API_KEYWORDS = [
    "insufficient_quota", "quota exceeded", "exceeded your current quota",
    "billing", "credit", "credits", "payment required", "account balance",
    "usage limit", "hard limit",
]

# Structured-output schema of the model answer
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "routes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "vehicle": {"type": "integer", "minimum": 1},
                    "path": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
                },
                "required": ["vehicle", "path"],
                "additionalProperties": False,
            },
        },
        "stats": {
            "type": "object",
            "properties": {
                "customers_served": {"type": "integer", "minimum": 0},
                "total_distance": {"type": "number", "minimum": 0},
                "vehicles_used": {"type": "integer", "minimum": 0},
            },
            "required": ["customers_served", "total_distance", "vehicles_used"],
            "additionalProperties": False,
        },
        "explanation": {"type": "string"},
    },
    "required": ["routes", "stats", "explanation"],
    "additionalProperties": False,
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def resolve_dataset_path(config_path, dataset_path):
    dataset_path = os.path.expanduser(dataset_path)
    if os.path.exists(dataset_path):
        return os.path.abspath(dataset_path)
    candidate = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(config_path)), dataset_path))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")


def select_ids(all_ids, n_instances, seed, graph_ids=None, use_all=False):
    if graph_ids is not None:
        missing = [g for g in graph_ids if g not in set(all_ids)]
        if missing:
            raise ValueError(f"Missing graph IDs: {missing}")
        return list(graph_ids)
    if use_all:
        return list(all_ids)
    rng = np.random.default_rng(seed)
    n = min(int(n_instances), len(all_ids))
    return rng.choice(all_ids, size=n, replace=False).tolist()


def select_example_ids(sampled_ids, n_examples, seed):
    n = min(max(int(n_examples), 0), len(sampled_ids))
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    return rng.choice(sampled_ids, size=n, replace=False).tolist()


def call_openai(df: pd.DataFrame, model: str, reasoning_effort: str, config: dict):
    system_prompt, user_prompt, _ = build_prompts(df=df, config=config)

    client = OpenAI()   # reads OPENAI_API_KEY
    print(f"Sending request to OpenAI (model={model}, reasoning={reasoning_effort})...", flush=True)

    response = client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "evrp_solution",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            }
        },
        store=False,   # responses are saved locally
    )

    raw_response = response.output_text.strip()
    if not raw_response:
        raise RuntimeError("OpenAI returned an empty response.")
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenAI response could not be parsed as JSON.\nRaw response:\n{raw_response}") from e

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if response.usage is not None:
        usage["input_tokens"] = int(getattr(response.usage, "input_tokens", 0) or 0)
        usage["output_tokens"] = int(getattr(response.usage, "output_tokens", 0) or 0)
        usage["total_tokens"] = int(getattr(response.usage, "total_tokens", 0)
                                    or (usage["input_tokens"] + usage["output_tokens"]))

    print(f"OpenAI response received | input={usage['input_tokens']:,} | "
          f"output={usage['output_tokens']:,} | total={usage['total_tokens']:,}", flush=True)

    return {
        "result": result,
        "raw_response": raw_response,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "usage": usage,
    }


def solve_batch(
    config_path,
    n_instances=None,
    seed=None,
    save_prompt_examples=None,
    model=None,
    reasoning_effort=None,
    result_dir=None,
    graph_ids=None,
    use_all=False,
    estimate_only=False,
    resume=False,
):
    config = load_config(config_path)
    llm_cfg = config["llm_baseline"]
    test_path = resolve_dataset_path(config_path, config["dataset"]["test_csv"])

    n_instances = n_instances if n_instances is not None else llm_cfg["n_instances"]
    seed = seed if seed is not None else llm_cfg["seed"]
    save_prompt_examples = save_prompt_examples if save_prompt_examples is not None else llm_cfg["save_prompt_examples"]
    model = model or llm_cfg.get("model", DEFAULT_MODEL)
    reasoning_effort = reasoning_effort or llm_cfg["reasoning_effort"]
    result_dir = os.path.abspath(os.path.expanduser(result_dir or llm_cfg["result_dir"]))
    os.makedirs(result_dir, exist_ok=True)

    selection_path = os.path.join(result_dir, "experiment_selection.json")
    metrics_path = os.path.join(result_dir, "metrics_per_graph.csv")
    summary_path = os.path.join(result_dir, "summary.csv")
    token_estimates_path = os.path.join(result_dir, "token_estimates.csv")

    df_all = load_dataset(test_path)
    all_ids = df_all["graph_id"].drop_duplicates().tolist()

    if resume:
        if not os.path.exists(selection_path):
            raise FileNotFoundError(f"Cannot resume because {selection_path} does not exist.")
        with open(selection_path, "r", encoding="utf-8") as f:
            old_selection = json.load(f)
        sampled_ids = [int(x) for x in old_selection["selected_graph_ids"]]
        example_ids = [int(x) for x in old_selection["example_graph_ids"]]

        existing_metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
        if len(existing_metrics_df) > 0 and "graph_id" in existing_metrics_df.columns:
            existing_metrics_df["graph_id"] = existing_metrics_df["graph_id"].astype(int)
            completed_set = set(existing_metrics_df["graph_id"].tolist())
        else:
            completed_set = set()

        remaining_ids = [gid for gid in sampled_ids if gid not in completed_set]

        print("\n" + "=" * 65 + "\nRESUME MODE\n" + "=" * 65)
        print(f"Original instances : {len(sampled_ids)}")
        print(f"Already completed  : {len(completed_set)}")
        print(f"Remaining          : {len(remaining_ids)}")
        print(f"Original examples  : {example_ids}")
        if remaining_ids:
            print(f"Next graph         : {remaining_ids[0]}")
        print("=" * 65 + "\n")
    else:
        sampled_ids = [int(x) for x in select_ids(all_ids, n_instances, seed, graph_ids=graph_ids, use_all=use_all)]
        example_ids = [int(x) for x in select_example_ids(sampled_ids, save_prompt_examples, seed)]
        remaining_ids = list(sampled_ids)
        existing_metrics_df = pd.DataFrame()

        selection = {
            "config_path": os.path.abspath(config_path),
            "test_dataset": test_path,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "seed": seed,
            "selected_graph_ids": sampled_ids,
            "example_graph_ids": example_ids,
        }
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump(selection, f, indent=2)

    example_set = set(example_ids)

    # Token estimate
    if not resume or not os.path.exists(token_estimates_path):
        estimates = []
        for gid in sampled_ids:
            gdf = df_all[df_all["graph_id"] == gid].reset_index(drop=True)
            sp, up, _ = build_prompts(gdf, config)
            estimates.append({
                "graph_id": gid,
                "num_nodes": len(gdf),
                "num_customers": int((gdf["type"] == 1).sum()),
                "estimated_input_tokens": estimate_tokens(sp, up),
            })
        estimates_df = pd.DataFrame(estimates)
        estimates_df.to_csv(token_estimates_path, index=False)
    else:
        estimates_df = pd.read_csv(token_estimates_path)

    print(f"Model: {model}")
    print(f"Requested instances: {len(sampled_ids)}")
    print(f"Remaining instances: {len(remaining_ids)}")
    print(f"Examples: {example_ids}")
    if len(estimates_df):
        print(f"Estimated input tokens (full experiment): {int(estimates_df['estimated_input_tokens'].sum()):,}")

    if estimate_only:
        print("Estimate only; no API calls made.")
        return

    if len(remaining_ids) == 0:
        print("\nAll selected instances have already been completed.")
        metrics_df = existing_metrics_df.copy()
        if len(metrics_df):
            summary = summarize_metrics(metrics_df)
            summary.update({
                "num_requested": len(sampled_ids),
                "num_attempted": len(sampled_ids),
                "num_successful": len(metrics_df),
                "num_failed": 0,
                "num_not_attempted": 0,
                "failed_graph_ids": json.dumps([]),
                "completed_graph_ids": json.dumps(metrics_df["graph_id"].astype(int).tolist()),
                "stop_reason": "",
                "seed": seed,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "example_graph_ids": json.dumps(example_ids),
            })
            pd.DataFrame([summary]).to_csv(summary_path, index=False)
        return

    env_builder = build_env_builder(config) if example_ids else None

    if resume and len(existing_metrics_df) > 0:
        metrics_rows = existing_metrics_df.to_dict("records")
    else:
        metrics_rows = []

    failed_ids = []
    stop_reason = None

    # Main loop
    for idx, gid in enumerate(remaining_ids, 1):
        is_example = gid in example_set
        print(f"\n[{idx}/{len(remaining_ids)}] graph={gid} example={is_example}")
        gdf = df_all[df_all["graph_id"] == gid].reset_index(drop=True)

        try:
            api = call_openai(df=gdf, model=model, reasoning_effort=reasoning_effort, config=config)
            result = routes_to_result(routes=api["result"]["routes"], graph_df=gdf, config=config)
            est_tokens = estimate_tokens(api["system_prompt"], api["user_prompt"])

            row = extract_metrics(result, gid, est_tokens, api["usage"])
            row["is_example"] = is_example
            metrics_rows.append(row)

            pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

            if is_example:
                save_dir = os.path.join(result_dir, f"graph_{gid}")
                os.makedirs(save_dir, exist_ok=True)
                save_prompt_example(save_dir, gid, model, api["system_prompt"], api["user_prompt"],
                                    api["raw_response"], api["result"], api["usage"], est_tokens)
                save_example_summaries(result, save_dir)

                env = env_builder(gdf)
                plot_all_vehicle_figures(env=env, result=result,
                                         save_dir=os.path.join(save_dir, "route_figures"), depot_idx=0)
                plot_all_vehicle_battery_and_distance(result=result,
                                                      save_dir=os.path.join(save_dir, "battery_distance"), depot_idx=0)

            print(f"Done distance={row['total_distance']:.2f} served={row['served_count']:.4f} "
                  f"tokens={row['api_total_tokens']:,}")

        except KeyboardInterrupt:
            stop_reason = "KeyboardInterrupt"
            print("\nKeyboardInterrupt received.\nStopping batch and summarizing completed instances.")
            break

        except Exception as e:
            error_type, error_message = type(e).__name__, str(e)
            print(f"FAILED: {error_type}: {error_message}", flush=True)
            failed_ids.append(gid)

            error_text = f"{error_type} {error_message}".lower()
            if any(keyword in error_text for keyword in FATAL_API_KEYWORDS):
                stop_reason = f"{error_type}: {error_message}"
                print("\n" + "=" * 65)
                print("Fatal OpenAI quota/billing error detected.")
                print(f"Stopping with {len(metrics_rows)} total successful instance(s) saved.")
                print("Summary will use only successful instances.")
                print("=" * 65 + "\n")
                break

    # Final metrics
    metrics_df = pd.DataFrame(metrics_rows)
    if len(metrics_df):
        metrics_df["graph_id"] = metrics_df["graph_id"].astype(int)
        metrics_df = metrics_df.drop_duplicates(subset=["graph_id"], keep="last").reset_index(drop=True)
        graph_order = {int(gid): i for i, gid in enumerate(sampled_ids)}
        metrics_df["_order"] = metrics_df["graph_id"].map(graph_order)
        metrics_df = metrics_df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
        metrics_df.to_csv(metrics_path, index=False)
        summary = summarize_metrics(metrics_df)
        successful_set = set(metrics_df["graph_id"].astype(int).tolist())
    else:
        summary = {"num_graphs": 0}
        successful_set = set()

    failed_set = set(failed_ids)
    remaining_after_run = [gid for gid in sampled_ids if gid not in successful_set and gid not in failed_set]
    completed_graph_ids = [gid for gid in sampled_ids if gid in successful_set]

    summary.update({
        "num_requested": len(sampled_ids),
        "num_attempted": len(successful_set) + len(failed_set),
        "num_successful": len(successful_set),
        "num_failed": len(failed_set),
        "num_not_attempted": len(remaining_after_run),
        "failed_graph_ids": json.dumps(failed_ids),
        "completed_graph_ids": json.dumps(completed_graph_ids),
        "remaining_graph_ids": json.dumps(remaining_after_run),
        "stop_reason": stop_reason or "",
        "seed": seed,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "example_graph_ids": json.dumps(example_ids),
    })
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n" + "=" * 65 + "\nFinished\n" + "=" * 65)
    print(f"Requested     : {len(sampled_ids)}")
    print(f"Successful    : {len(successful_set)}")
    print(f"Failed        : {len(failed_set)}")
    print(f"Not completed : {len(remaining_after_run)}")
    if remaining_after_run:
        print(f"Next remaining: {remaining_after_run[0]}")
    if stop_reason:
        print(f"Stopped because: {stop_reason}")
    print(f"Results       : {result_dir}")
    print("=" * 65)

    return {
        "metrics_df": metrics_df,
        "summary": summary,
        "completed_graph_ids": completed_graph_ids,
        "remaining_graph_ids": remaining_after_run,
        "failed_graph_ids": failed_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="GPT-5.6-Luna baseline for the EVRPTW")
    parser.add_argument("config_path")
    parser.add_argument("--n_instances", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save_prompt_examples", type=int)
    parser.add_argument("--model")
    parser.add_argument("--reasoning_effort", choices=["none", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--result_dir")
    parser.add_argument("--graph_ids", type=int, nargs="+")
    parser.add_argument("--all", action="store_true", help="Use all test instances")
    parser.add_argument("--estimate_only", action="store_true", help="Only estimate input tokens; no API calls")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing experiment: reads the graph selection from "
                             "experiment_selection.json and skips graphs already in metrics_per_graph.csv.")
    args = parser.parse_args()

    solve_batch(
        config_path=args.config_path,
        n_instances=args.n_instances,
        seed=args.seed,
        save_prompt_examples=args.save_prompt_examples,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        result_dir=args.result_dir,
        graph_ids=args.graph_ids,
        use_all=args.all,
        estimate_only=args.estimate_only,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()