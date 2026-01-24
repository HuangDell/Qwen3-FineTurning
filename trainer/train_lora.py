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
NUM_EPOCHS = 2  
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4  
LOGGING_STEPS = 10
SAVE_STEPS = 200
EVAL_STEPS = 200  
SEED = 42
VAL_SPLIT_RATIO = 0.1  # 10% for validation

# Precision
USE_BF16 = True
USE_FP16 = False

# LoRA
LORA_R = 8  # Reduced from 16 to reduce overfitting risk
LORA_ALPHA = 16  # Adjusted proportionally
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

    # Split into train and validation sets
    print(f"Total samples: {len(dataset)}")
    dataset_split = dataset.train_test_split(test_size=VAL_SPLIT_RATIO, seed=SEED)
    train_dataset = dataset_split['train']
    val_dataset = dataset_split['test']
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Rename 'conversations' to 'messages' (TRL's standard field name for conversational datasets)
    def rename_to_messages(example: Dict) -> Dict:
        return {"messages": example["conversations"]}

    train_dataset = train_dataset.map(rename_to_messages, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(rename_to_messages, remove_columns=val_dataset.column_names)

    dtype = torch.bfloat16 if USE_BF16 else torch.float16 if USE_FP16 else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_OR_PATH,
        torch_dtype=dtype,
        device_map="auto",
    )


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
        report_to=["tensorboard"],
        max_length=MAX_SEQ_LENGTH,
        assistant_only_loss=True,  # Only compute loss on assistant responses
        logging_dir=f"{OUTPUT_DIR}/logs",
        
        # Validation and checkpointing
        eval_strategy="steps",  # Evaluate at regular intervals
        eval_steps=EVAL_STEPS,  # Evaluate every N steps
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Use validation loss to determine best model
        greater_is_better=False,  # Lower loss is better
        save_total_limit=2,  # Keep only the 2 best checkpoints
        save_strategy="steps",  # Save at regular intervals
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add validation dataset
        processing_class=tokenizer,
        peft_config=lora_config,
    )



    print("\n" + "="*70)
    print("Starting training with validation monitoring...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"LoRA rank: {LORA_R}")
    print(f"Save limit: 2 checkpoints")
    print(f"TensorBoard logs: {OUTPUT_DIR}/logs")
    print("="*70 + "\n")
    
    trainer.train()
    
    print("\n" + "="*70)
    print("Training completed! Best model loaded automatically.")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"View training curves: tensorboard --logdir={OUTPUT_DIR}/logs")
    print("="*70 + "\n")
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Also save the chat template for consistency
    import shutil
    chat_template_save_path = os.path.join(OUTPUT_DIR, "chat_template.jinja")
    shutil.copy(chat_template_path, chat_template_save_path)
    print(f"Chat template saved to: {chat_template_save_path}")


if __name__ == "__main__":
    main()