import json
import gc
from pathlib import Path
from typing import Any, Callable, Sequence

from config.logging import set_up_logger
from cs336_alignment.math_baseline import evaluate_vllm, extract_last_number, parse_gsm8k_ground_truth, _safe_mean
from loguru import logger
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
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 512



def load_gsm8k_jsonl(input_path: Path) -> list[dict[str, Any]]:
    """Load GSM8K examples from a JSONL file.

    Args:
        input_path: Path to the JSONL file. Supports compressed files via xopen.

    Returns:
        List of dicts, each containing at minimum ``question`` and ``answer`` keys.

    Raises:
        ValueError: If a line is not valid JSON, not a dict, or missing required keys.
    """
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
) -> None:

    set_up_logger("DEBUG")

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

    logger.info(f"Loaded {len(examples)} examples from {input_path}")
    logger.info(f"Loading model from {model_name_or_path} ...")

    llm = LLM(model=str(model_name_or_path))

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

    evaluate_vllm(
        llm,
        prompts,
        reward_fn,
        ground_truths,
        eval_sampling_params,
        output_path,
    )


if __name__ == "__main__":
    app()