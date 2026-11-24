"""JET (Ministral 3): train_qwen.py with the Ministral model, tokenizer, and chat template."""

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText

from JET.halting import HaltingTrainer
from JET.train_qwen import parse_args, prepare_output
from utils import (
    attach_adapter,
    build_sft_config,
    drop_long_rows,
    load_tokenizer,
    parse_tool_arguments,
    save_args,
)

# 언어 모델의 선형층만 (vision tower 와 projector 에도 같은 이름의 층이 있다)
LORA_TARGET_MODULES = r"^(?!.*(vision|projector)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def main() -> None:
    args = parse_args()
    output_dir = prepare_output(args, __file__)

    tokenizer = load_tokenizer(args.tokenizer or args.model, args.chat_template)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
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
    # Trainer 가 누적 배치의 토큰 수를 손실에 넘기게 한다 (모듈 docstring)
    trainer.model_accepts_loss_kwargs = True
    if trainer.is_world_process_zero():
        save_args(args, output_dir / "train_args.json")
    trainer.train()


if __name__ == "__main__":
    main()
