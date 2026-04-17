import json
import re
import gc
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import typer
import torch.distributed as dist
from vllm import LLM, SamplingParams
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_MODEL_PATH = Path("./Qwen2.5-Math-1.5B")
DEFAULT_INPUT_PATH = Path("data/gsm8k/test.jsonl")
DEFAULT_PROMPT_TEMPLATE_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")
DEFAULT_OUTPUT_PATH = Path("outputs/qwen2_5_math_1_5b_gsm8k_r1_zero.jsonl")
DEFAULT_NUM_GPUS = 1
DEFAULT_BATCH_SIZE = 64
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 512

# Matches signed integers/decimals with optional thousands separators.
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
GSM8K_ANSWER_PATTERN = re.compile(
    r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)


def _safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _normalize_number_string(number_str: str) -> str:
    return number_str.replace(",", "").strip()


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return _normalize_number_string(matches[-1])


def parse_gsm8k_ground_truth(answer_text: str) -> str | None:
    match = GSM8K_ANSWER_PATTERN.search(answer_text)
    if match is not None:
        return _normalize_number_string(match.group(1))
    return extract_last_number(answer_text)


def load_gsm8k_jsonl(input_path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with xopen(input_path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {input_path}: {exc}"
                ) from exc

            if not isinstance(example, dict):
                raise ValueError(
                    f"Expected object at line {line_number} in {input_path}, got {type(example)}"
                )
            if "question" not in example or "answer" not in example:
                raise ValueError(
                    f"Missing required keys at line {line_number} in {input_path}; expected question and answer"
                )
            examples.append(example)
    return examples


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]] = r1_zero_reward_fn,
    prompts: Sequence[str] = (),
    ground_truths: Sequence[str] = (),
    eval_sampling_params: SamplingParams | None = None,
    *,
    examples: Sequence[dict[str, Any]] | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name_or_path: str = str(DEFAULT_MODEL_PATH),
) -> dict[str, float]:
    """Run batched vLLM evaluation, write per-example JSONL, and return summary metrics."""
    if eval_sampling_params is None:
        eval_sampling_params = SamplingParams(
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    if examples is None:
        examples = [{} for _ in prompts]

    if not (len(prompts) == len(ground_truths) == len(examples)):
        raise ValueError("prompts, ground_truths, and examples must have equal length")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    format_rewards: list[float] = []
    answer_rewards: list[float] = []
    rewards: list[float] = []
    parsed_exact_matches: list[float] = []
    parse_successes: list[float] = []

    with xopen(output_path, "w") as fout:
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            prompt_batch = list(prompts[start:end])
            output_batch = vllm_model.generate(prompt_batch, eval_sampling_params)

            for offset, output in enumerate(output_batch):
                idx = start + offset
                generated_text = output.outputs[0].text.strip() if output.outputs else ""
                ground_truth = ground_truths[idx]
                reward = reward_fn(generated_text, ground_truth)

                parsed_prediction = extract_last_number(generated_text)
                parsed_ground_truth = parse_gsm8k_ground_truth(ground_truth)
                parse_success = (
                    parsed_prediction is not None and parsed_ground_truth is not None
                )

                format_reward = float(reward.get("format_reward", 0.0))
                answer_reward = float(reward.get("answer_reward", 0.0))
                reward_value = float(reward.get("reward", 0.0))
                # Keep this metric aligned with the reward grader used for evaluation.
                parsed_match = 1.0 if answer_reward > 0.0 else 0.0

                format_rewards.append(format_reward)
                answer_rewards.append(answer_reward)
                rewards.append(reward_value)
                parsed_exact_matches.append(parsed_match)
                parse_successes.append(1.0 if parse_success else 0.0)

                record = {
                    **examples[idx],
                    "example_index": idx,
                    "model_name_or_path": model_name_or_path,
                    "prompt": prompts[idx],
                    "model_output": generated_text,
                    "parsed_prediction": parsed_prediction,
                    "parsed_ground_truth": parsed_ground_truth,
                    "metrics": {
                        "format_reward": format_reward,
                        "answer_reward": answer_reward,
                        "reward": reward_value,
                        "parse_success": 1.0 if parse_success else 0.0,
                        "parsed_exact_match": parsed_match,
                    },
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "num_examples": float(len(prompts)),
        "mean_format_reward": _safe_mean(format_rewards),
        "mean_answer_reward": _safe_mean(answer_rewards),
        "mean_reward": _safe_mean(rewards),
        "parse_success_rate": _safe_mean(parse_successes),
        "parsed_exact_match": _safe_mean(parsed_exact_matches),
    }


def cleanup_vllm(llm: LLM | None) -> None:
    if llm is None:
        return

    try:
        model_executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
        if model_executor is not None and hasattr(model_executor, "shutdown"):
            model_executor.shutdown()
    except Exception as exc:
        typer.echo(f"Warning: model executor shutdown failed: {exc}", err=True)

    del llm
    gc.collect()

    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:
        typer.echo(f"Warning: distributed process group cleanup failed: {exc}", err=True)


@app.command()
def main(
    model_name_or_path: Path = typer.Option(
        DEFAULT_MODEL_PATH,
        help="Local path or HF identifier for the model.",
    ),
    input_path: Path = typer.Option(
        DEFAULT_INPUT_PATH,
        help="Path to GSM8K JSONL evaluation data.",
    ),
    prompt_template_path: Path = typer.Option(
        DEFAULT_PROMPT_TEMPLATE_PATH,
        help="Path to r1_zero prompt template file.",
    ),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        help="Path to write per-example JSONL outputs.",
    ),
    summary_path: Path | None = typer.Option(
        None,
        help="Optional path to write aggregate metrics JSON. Defaults to output_path with .summary.json suffix.",
    ),
    num_gpus: int = typer.Option(DEFAULT_NUM_GPUS, min=1, help="Tensor parallel GPU count for vLLM."),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, min=1, help="Number of prompts per generation batch."),
    max_examples: int | None = typer.Option(
        None, min=1, help="Optional cap on number of evaluation examples for quick runs."
    ),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE, help="Sampling temperature."),
    top_p: float = typer.Option(DEFAULT_TOP_P, help="Nucleus sampling top-p."),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, min=1, help="Maximum generated tokens per example."),
    fast_grading: bool = typer.Option(
        True,
        help="Use faster reward grading mode. Disable for more thorough grading.",
    ),
    trust_remote_code: bool = typer.Option(
        True,
        help="Whether to trust remote code while loading model in vLLM.",
    ),
) -> None:
    if not input_path.exists():
        raise typer.BadParameter(f"Input dataset does not exist: {input_path}")
    if not prompt_template_path.exists():
        raise typer.BadParameter(f"Prompt template does not exist: {prompt_template_path}")

    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    if "{question}" not in prompt_template:
        raise typer.BadParameter(
            f"Prompt template must contain '{{question}}': {prompt_template_path}"
        )

    examples = load_gsm8k_jsonl(input_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    prompts = [
        prompt_template.format(question=example["question"])
        for example in examples
    ]
    ground_truths = [str(example["answer"]) for example in examples]

    typer.echo(f"Loaded {len(examples)} examples from {input_path}")
    typer.echo(f"Loading model from {model_name_or_path} ...")
    llm: LLM | None = None
    try:
        llm = LLM(
            model=str(model_name_or_path),
            tensor_parallel_size=num_gpus,
            trust_remote_code=trust_remote_code,
        )

        eval_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        reward_fn = lambda response, ground_truth: r1_zero_reward_fn(
            response,
            ground_truth,
            fast=fast_grading,
        )

        summary = evaluate_vllm(
            vllm_model=llm,
            reward_fn=reward_fn,
            prompts=prompts,
            ground_truths=ground_truths,
            eval_sampling_params=eval_sampling_params,
            examples=examples,
            output_path=output_path,
            batch_size=batch_size,
            model_name_or_path=str(model_name_or_path),
        )

        if summary_path is None:
            summary_path = output_path.with_suffix(".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary_payload: dict[str, Any] = {
            **summary,
            "model_name_or_path": str(model_name_or_path),
            "input_path": str(input_path),
            "prompt_template_path": str(prompt_template_path),
            "output_path": str(output_path),
            "num_gpus": num_gpus,
            "batch_size": batch_size,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "fast_grading": fast_grading,
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        typer.echo(f"Wrote per-example outputs to {output_path}")
        typer.echo(f"Wrote summary metrics to {summary_path}")
        typer.echo(
            "Summary: "
            f"n={int(summary['num_examples'])}, "
            f"reward={summary['mean_reward']:.4f}, "
            f"answer_reward={summary['mean_answer_reward']:.4f}, "
            f"format_reward={summary['mean_format_reward']:.4f}, "
            f"parsed_em={summary['parsed_exact_match']:.4f}"
        )
    finally:
        cleanup_vllm(llm)


if __name__ == "__main__":
    app()