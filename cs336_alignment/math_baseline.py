import json
import re
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

from loguru import logger
from vllm import LLM, SamplingParams
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
) -> None:
    """
        Evaluate a language model on a list of prompts,
        compute evaluation metrics, and serialize results to disk.
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

    times = 5;

    with xopen(output_path, "w") as fout:
        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip() if output.outputs else ""
            ground_truth = ground_truths[idx]
            reward = reward_fn(generated_text, ground_truth)

            format_reward = float(reward.get("format_reward", 0.0))
            answer_reward = float(reward.get("answer_reward", 0.0))
            reward_value = float(reward.get("reward", 0.0))
            # Keep this metric aligned with the reward grader used for evaluation.
            parsed_match = 1.0 if answer_reward > 0.0 else 0.0

            format_rewards.append(format_reward)
            answer_rewards.append(answer_reward)
            rewards.append(reward_value)
            parsed_exact_matches.append(parsed_match)

            parsed_pred = extract_last_number(generated_text)
            parsed_gt = parse_gsm8k_ground_truth(ground_truth)
            if times > 0:
                logger.debug(
                    f"[{idx}] generated_text={generated_text} \n"+
                    f"ground_truth={ground_truth} \n"+
                    f"extracted_pred={parsed_pred}\nparsed_gt={parsed_gt} "+
                    f"correct={bool(parsed_match)}\nfmt={format_reward}"
                )
                times -= 1

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

    logger.info(
        f"Done. n={len(prompts)} | "
        f"reward={_safe_mean(rewards):.4f} | "
        f"answer={_safe_mean(answer_rewards):.4f} | "
        f"format={_safe_mean(format_rewards):.4f} | "
        f"exact_match={_safe_mean(parsed_exact_matches):.4f}"
    )

