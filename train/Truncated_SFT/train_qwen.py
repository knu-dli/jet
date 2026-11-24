"""Truncated SFT (Qwen3): LoRA fine-tuning on the truncated trajectories without the rationale, with the default
TRL loss.
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

from JET.train_qwen import LORA_TARGET_MODULES, prepare_output
from utils import (
    add_adapter_args,
    add_common_args,
    attach_adapter,
    build_sft_config,
    drop_long_rows,
    load_tokenizer,
    parse_tool_arguments,
    save_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an adapter on the golden-step trajectories with the plain SFT loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    add_adapter_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = prepare_output(args, __file__)

    tokenizer = load_tokenizer(args.tokenizer or args.model, args.chat_template)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model, peft_config = attach_adapter(model, args, LORA_TARGET_MODULES)
    dataset = load_dataset(args.dataset_dir, split="train").map(parse_tool_arguments)
    trainer = SFTTrainer(
        args=build_sft_config(args, loss_type="chunked_nll"),
        model=model,
        processing_class=tokenizer,
        train_dataset=drop_long_rows(dataset, tokenizer, args.max_length),
        peft_config=peft_config,
    )
    if trainer.is_world_process_zero():
        save_args(args, output_dir / "train_args.json")
    trainer.train()


if __name__ == "__main__":
    main()
