import json
import re
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence
from unittest.mock import patch

import torch
from loguru import logger
from transformers import PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn



def _safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _normalize_number_string(number_str: str) -> str:
    return number_str.replace(",", "").strip()


def extract_last_number(text: str) -> str | None:
    """Finds a number in the model's generated output."""
    number_pattern = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
    matches = number_pattern.findall(text)
    if not matches:
        return None
    return _normalize_number_string(matches[-1])


def parse_gsm8k_ground_truth(answer_text: str) -> str | None:
    """Finds the answer in a GSM8K label.
    GSM8K answers are typically in the format "#### 42".
    """
    gsm8k_answer_pattern = re.compile(
    r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)
    match = gsm8k_answer_pattern.search(answer_text)
    if match is not None:
        return _normalize_number_string(match.group(1))
    return extract_last_number(answer_text)


def evaluate_vllm(
    vllm_model: LLM,
    prompts: Sequence[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    ground_truths: Sequence[str],
    eval_sampling_params: SamplingParams,
    output_path: Path,
) -> dict[str, float]:
    """Evaluate a language model on a list of prompts, compute metrics, and
    serialize results to disk.

    Args:
        vllm_model: vLLM instance for generation.
        prompts: Input prompts.
        reward_fn: Scores each response against its ground truth.
        ground_truths: Ground-truth answers.
        eval_sampling_params: vLLM sampling configuration.
        output_path: Path to write per-example JSONL results.

    Returns:
        dict of aggregate metrics (reward, answer_reward, format_reward, exact_match).
    """
    if len(prompts) != len(ground_truths):
        raise ValueError("prompts and ground_truths must have equal length")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    format_rewards: list[float] = []
    answer_rewards: list[float] = []
    rewards: list[float] = []
    parsed_exact_matches: list[float] = []

    logger.info(f"Generating outputs for {len(prompts)} examples ...")
    outputs = vllm_model.generate(list(prompts), eval_sampling_params)

    with xopen(output_path, "w") as fout:
        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip() if output.outputs else ""
            ground_truth = ground_truths[idx]
            reward = reward_fn(generated_text, ground_truth)

            format_reward = float(reward.get("format_reward", 0.0))
            answer_reward = float(reward.get("answer_reward", 0.0))
            reward_value = float(reward.get("reward", 0.0))
            parsed_match = 1.0 if answer_reward > 0.0 else 0.0

            format_rewards.append(format_reward)
            answer_rewards.append(answer_reward)
            rewards.append(reward_value)
            parsed_exact_matches.append(parsed_match)

            parsed_pred = extract_last_number(generated_text)
            parsed_gt = parse_gsm8k_ground_truth(ground_truth)
            logger.debug(
                f"[{idx}] prompt={prompts[idx]!r}\n"
                f"     response={generated_text!r}\n"
                f"     ground_truth={ground_truth!r}\n"
                f"     reward={reward_value:.3f} | format={format_reward:.3f} | answer={answer_reward:.3f}"
            )

            record = {
                "example_index": idx,
                "prompt": prompts[idx],
                "model_output": generated_text,
                "parsed_prediction": parsed_pred,
                "parsed_ground_truth": parsed_gt,
                "metrics": {
                    "format_reward": format_reward,
                    "answer_reward": answer_reward,
                    "reward": reward_value,
                    "parsed_exact_match": parsed_match,
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "reward": _safe_mean(rewards),
        "answer_reward": _safe_mean(answer_rewards),
        "format_reward": _safe_mean(format_rewards),
        "exact_match": _safe_mean(parsed_exact_matches),
    }

    logger.info(
        f"Done. n={len(prompts)} | "
        + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )
    return metrics


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    """Initialize a vLLM LLM instance on a specific device.

    Uses monkeypatches from TRL to allow placing vLLM on a specific GPU
    (world_size_patch) and to skip a profiling check incompatible with our
    single-process setup (profiling_patch).

    Args:
        model_id: Path or HuggingFace identifier for the model.
        device: Device string, e.g. "cuda:1".
        seed: Random seed for reproducible sampling.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.

    Returns:
        Initialized LLM instance.
    """
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """Copy the current policy weights into a vLLM LLM instance.

    Called before each evaluation so vLLM generates with the latest trained
    weights. Adapted from TRL's GRPOTrainer.

    Args:
        policy: The PyTorch policy model being trained.
        llm: The vLLM LLM instance to update.
    """
    state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
