"""JET (Qwen3): LoRA fine-tuning on the truncated trajectories with the halting weight λ (--halting-weight)."""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from JET.halting import HaltingTrainer
from utils import (
    add_adapter_args,
    add_common_args,
    attach_adapter,
    build_sft_config,
    drop_long_rows,
    load_tokenizer,
    parse_tool_arguments,
    resolve_output_dir,
    save_args,
    set_local_device,
    tee_output,
)

# 어댑터를 붙일 선형층: 어텐션과 MLP (SFT/train_qwen.py 와 같다)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an adapter on the stop-labeled trajectories, from the base model or the SFT adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    add_adapter_args(parser)
    parser.add_argument(
        "--halting-weight",
        type=float,
        default=5.0,
        help="loss weight of the halting turn before <finish> (lambda); 5 in the paper",
    )
    return parser.parse_args()


def prepare_output(args: argparse.Namespace, script: str) -> Path:
    """출력 폴더를 만들고 진행 로그·wandb 기록을 그 안에 두게 한다."""
    resolve_output_dir(args, Path(script).resolve().parent)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    # 진행 상황(손실, 진행 막대)은 첫 프로세스만 남긴다
    if int(os.environ.get("RANK", "0")) == 0:
        tee_output(output_dir / "train_stdout.log")
    # wandb 의 로컬 기록도 출력 디렉터리 밑에 둔다 (기본은 실행 위치의 wandb/)
    os.environ.setdefault("WANDB_DIR", str(output_dir))
    set_local_device()
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = prepare_output(args, __file__)

    tokenizer = load_tokenizer(args.tokenizer or args.model, args.chat_template)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model, peft_config = attach_adapter(model, args, LORA_TARGET_MODULES)
    dataset = load_dataset(args.dataset_dir, split="train").map(parse_tool_arguments)
    trainer = HaltingTrainer(
        halting_weight=args.halting_weight,
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
