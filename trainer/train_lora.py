import os
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# =========================
# Editable parameters
# =========================
MODEL_ID_OR_PATH = "model/qwen"  # TODO: replace with your exact model ID/path
OUTPUT_DIR = "out/qwen3-4b-lora-medical"

# Data
DATA_PATH = "data/lora_medical.jsonl"
MAX_SEQ_LENGTH = 1024

# Training
NUM_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10
SAVE_STEPS = 200
SEED = 42

# Precision
USE_BF16 = True
USE_FP16 = False

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def resolve_data_path() -> str:
    if DATA_PATH:
        return DATA_PATH
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", "lora_identity.jsonl")


def main() -> None:
    data_path = resolve_data_path()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_OR_PATH, use_fast=True)


    # Load custom chat template with {% generation %} support for assistant_only_loss
    chat_template_path = "./template/qwen_chat_template.jinja"
    with open(chat_template_path, "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()

    dataset = load_dataset("json", data_files=data_path, split="train")

    # Rename 'conversations' to 'messages' (TRL's standard field name for conversational datasets)
    def rename_to_messages(example: Dict) -> Dict:
        return {"messages": example["conversations"]}

    dataset = dataset.map(rename_to_messages, remove_columns=dataset.column_names)

    dtype = torch.bfloat16 if USE_BF16 else torch.float16 if USE_FP16 else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_OR_PATH,
        torch_dtype=dtype,
        device_map="auto",
    )

    print("bos_token_id:", tokenizer.bos_token_id)
    print("eos_token_id:", tokenizer.eos_token_id)
    print("pad_token_id:", tokenizer.pad_token_id)
    # Align to tokenizer (recommended for TRL training)
    model.config.bos_token_id = tokenizer.bos_token_id          # None
    model.config.eos_token_id = tokenizer.eos_token_id          # 151645
    model.config.pad_token_id = tokenizer.pad_token_id          # 151643

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id


    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    # Use SFTConfig with assistant_only_loss for proper loss masking
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        bf16=USE_BF16 and torch.cuda.is_available(),
        fp16=USE_FP16 and torch.cuda.is_available(),
        seed=SEED,
        report_to=[],
        max_length=MAX_SEQ_LENGTH,
        assistant_only_loss=True,  # Only compute loss on assistant responses
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )



    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()