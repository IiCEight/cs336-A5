"""SFT training script for Qwen2.5-Math-1.5B on the MATH dataset.

Implements Algorithm 1 (Supervised Finetuning) from the assignment PDF.
Requires 2 GPUs: one for the policy model (training) and one for vLLM (eval).

Usage:
    uv run python train_sft.py --help
    uv run python train_sft.py --n-sft-steps 1000 --eval-every 100
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import typer
import wandb
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
from xopen import xopen

from config.logging import set_up_logger
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.log_probs import get_response_log_probs
from cs336_alignment.math_baseline import (
    evaluate_vllm,
    init_vllm,
    load_policy_into_vllm_instance,
)
from cs336_alignment.tokenize import tokenize_prompt_and_output
from cs336_alignment.train_utils import sft_microbatch_train_step

app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_MODEL_PATH = Path("./Qwen2.5-Math-1.5B")
DEFAULT_TRAIN_DATA = Path("data/gsm8k/train.jsonl")
DEFAULT_VAL_DATA = Path("data/gsm8k/test.jsonl")
DEFAULT_PROMPT_TEMPLATE = Path("cs336_alignment/prompts/r1_zero.prompt")
DEFAULT_OUTPUT_DIR = Path("/data/outputs/sft")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples = []
    with xopen(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _load_val_examples(
    val_data_path: Path,
    prompt_template: str,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Load up to n validation examples with prompt and ground_truth keys."""
    examples = _load_jsonl(val_data_path)
    rng = random.Random(seed)
    if len(examples) > n:
        examples = rng.sample(examples, n)

    result = []
    for ex in examples:
        if "prompt" in ex:
            prompt = ex["prompt"]
            gt = ex.get("response", "")
        else:
            prompt = prompt_template.format(question=ex["question"])
            gt = str(ex["answer"])
        result.append({"prompt": prompt, "ground_truth": gt})
    return result


@app.command()
def main(
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, help="Policy model path."),
    train_data_path: Path = typer.Option(DEFAULT_TRAIN_DATA, help="Training JSONL (prompt/response)."),
    val_data_path: Path = typer.Option(DEFAULT_VAL_DATA, help="Validation JSONL for periodic eval."),
    prompt_template_path: Path = typer.Option(DEFAULT_PROMPT_TEMPLATE, help="Prompt template for val eval."),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory to save final model."),
    n_sft_steps: int = typer.Option(1000, help="Total optimizer steps."),
    batch_size: int = typer.Option(4, help="Examples per microbatch."),
    gradient_accumulation_steps: int = typer.Option(4, help="Microbatches per optimizer step."),
    learning_rate: float = typer.Option(2e-5, help="AdamW learning rate."),
    eval_every: int = typer.Option(100, help="Run eval every N optimizer steps."),
    n_eval_prompts: int = typer.Option(50, help="Number of val prompts for eval."),
    max_eval_tokens: int = typer.Option(1024, help="Max tokens for vLLM eval generation."),
    eval_temperature: float = typer.Option(0.0, help="Sampling temperature for eval."),
    eval_output_dir: Path = typer.Option(Path("outputs/eval"), help="Directory for eval JSONL outputs."),
    policy_device: str = typer.Option("cuda:0", help="Device for policy model."),
    vllm_device: str = typer.Option("cuda:1", help="Device for vLLM instance."),
    gpu_memory_utilization: float = typer.Option(0.85, help="vLLM GPU memory utilization."),
    seed: int = typer.Option(42, help="Random seed."),
    wandb_project: str = typer.Option("cs336-sft", help="W&B project name."),
    wandb_run_name: str = typer.Option("", help="W&B run name (empty = auto)."),
) -> None:
    set_up_logger("DEBUG")
    torch.manual_seed(seed)

    # --- W&B setup ---
    wandb.init(
        project=wandb_project,
        name=wandb_run_name or None,
        config={
            "model_path": str(model_path),
            "n_sft_steps": n_sft_steps,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
        },
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # --- Load policy model + tokenizer ---
    logger.info(f"Loading policy model from {model_path} on {policy_device} ...")
    tokenizer = AutoTokenizer.from_pretrained(Path(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        Path(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(policy_device)
    model.train()

    # --- Load training data ---
    logger.info(f"Loading training data from {train_data_path} ...")
    prompt_template = prompt_template_path.read_text(encoding="utf-8") if prompt_template_path.exists() else "{question}"
    train_examples = _load_jsonl(train_data_path)
    logger.info(f"Loaded {len(train_examples)} training examples.")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Init vLLM for eval ---
    logger.info(f"Initializing vLLM on {vllm_device} ...")
    vllm_model = init_vllm(
        model_id=str(model_path),
        device=vllm_device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    eval_sampling_params = SamplingParams(
        temperature=eval_temperature,
        max_tokens=max_eval_tokens,
        min_tokens=4,
    )

    # --- Load validation examples ---
    val_examples = _load_val_examples(val_data_path, prompt_template, n_eval_prompts, seed)
    logger.info(f"Loaded {len(val_examples)} validation examples.")

    # --- Training loop ---
    logger.info(f"Starting SFT training for {n_sft_steps} steps ...")
    rng = random.Random(seed)
    train_step = 0

    while train_step < n_sft_steps:
        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            # Sample a random microbatch of question-response pairs
            batch_examples = rng.choices(train_examples, k=batch_size)
            if "prompt" in batch_examples[0]:
                prompt_strs = [ex["prompt"] for ex in batch_examples]
                response_strs = [ex["response"] for ex in batch_examples]
            else:
                prompt_strs = [prompt_template.format(question=ex["question"]) for ex in batch_examples]
                response_strs = [ex["answer"] for ex in batch_examples]

            # Tokenize prompt+response, get response_mask for loss masking
            tokenized = tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)
            input_ids = tokenized["input_ids"].to(policy_device)
            labels = tokenized["labels"].to(policy_device)
            response_mask = tokenized["response_mask"].to(policy_device).float()

            log_probs_dict = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            policy_log_probs = log_probs_dict["log_probs"]  # (batch, seq_len)

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            total_loss += loss.item()

        # Gradient clipping (clip value 1.0 as recommended in PDF)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_step += 1

        logger.debug(f"Step {train_step}/{n_sft_steps} | loss={total_loss:.4f}")
        wandb.log({"train/loss": total_loss, "train_step": train_step})

        # Periodic eval
        if train_step % eval_every == 0:
            logger.info(f"Running eval at step {train_step} ...")
            model.eval()
            load_policy_into_vllm_instance(model, vllm_model)

            eval_output_path = eval_output_dir / f"step_{train_step:06d}.jsonl"
            metrics = evaluate_vllm(
                vllm_model=vllm_model,
                prompts=[e["prompt"] for e in val_examples],
                reward_fn=r1_zero_reward_fn,
                ground_truths=[e["ground_truth"] for e in val_examples],
                eval_sampling_params=eval_sampling_params,
                output_path=eval_output_path,
            )
            wandb.log({f"eval/{k}": v for k, v in metrics.items()} | {"eval_step": train_step})
            model.train()

    # --- Save checkpoint ---
    logger.info(f"Training complete. Saving model to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    logger.info("Done.")
    wandb.finish()


if __name__ == "__main__":
    app()
