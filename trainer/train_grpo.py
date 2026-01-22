import os
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# =========================
# Editable parameters
# =========================
MODEL_PATH = "model/qwen"  # Policy model path
REWARD_MODEL_PATH = "model/internlm"  # Reward model path
OUTPUT_DIR = "out/qwen3-4b-grpo"

# Data
DATA_PATH = "data/rlaif-mini.jsonl"
MAX_SEQ_LENGTH = 1024

# Training
NUM_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-6
LOGGING_STEPS = 10
SAVE_STEPS = 200
SEED = 42

# GRPO-specific parameters
NUM_GENERATIONS = 4  # Number of completions per prompt
MAX_NEW_TOKENS = 256  # Max tokens to generate per completion
TEMPERATURE = 0.7

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


# Global reward model and tokenizer (loaded once)
_reward_model = None
_reward_tokenizer = None


def load_reward_model():
    """Load the InternLM reward model and tokenizer."""
    global _reward_model, _reward_tokenizer
    
    if _reward_model is None:
        print(f"Loading reward model from: {REWARD_MODEL_PATH}")
        _reward_model = AutoModel.from_pretrained(
            REWARD_MODEL_PATH,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        _reward_tokenizer = AutoTokenizer.from_pretrained(
            REWARD_MODEL_PATH,
            trust_remote_code=True
        )
        print("Reward model loaded successfully")
    
    return _reward_model, _reward_tokenizer


def reward_func(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    Compute rewards for completions using InternLM reward model.
    
    Args:
        completions: List of generated completions
        prompts: List of corresponding prompts
        **kwargs: Additional arguments (unused)
    
    Returns:
        List of reward scores
    """
    reward_model, reward_tokenizer = load_reward_model()
    
    # Construct chat format for each prompt-completion pair
    chats = []
    for prompt, completion in zip(prompts, completions):
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        chats.append(chat)
    
    # Get scores in batch
    with torch.no_grad():
        scores = reward_model.get_scores(reward_tokenizer, chats)
    
    return scores


def prepare_dataset(data_path: str):
    """
    Load and prepare dataset for GRPO training.
    
    Extracts user prompts from conversations format.
    """
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    def extract_prompt(example: Dict) -> Dict:
        """Extract user prompt from conversations."""
        conversations = example["conversations"]
        # Get the first user message as the prompt
        prompt = ""
        for msg in conversations:
            if msg["role"] == "user":
                prompt = msg["content"]
                break
        return {"prompt": prompt}
    
    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    
    # Filter out empty prompts
    dataset = dataset.filter(lambda x: len(x["prompt"].strip()) > 0)
    
    print(f"Loaded {len(dataset)} prompts from {data_path}")
    return dataset


def main() -> None:
    """Main training function."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    
    # Load custom chat template
    chat_template_path = "./template/qwen_chat_template.jinja"
    if os.path.exists(chat_template_path):
        with open(chat_template_path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()
    
    # Ensure pad token is set
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare dataset
    dataset = prepare_dataset(DATA_PATH)
    
    # Load policy model
    dtype = torch.bfloat16 if USE_BF16 else torch.float16 if USE_FP16 else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Align token IDs
    # print("bos_token_id:", tokenizer.bos_token_id)
    # print("eos_token_id:", tokenizer.eos_token_id)
    # print("pad_token_id:", tokenizer.pad_token_id)
    
    # model.config.bos_token_id = tokenizer.bos_token_id
    # model.config.eos_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id
    
    # if getattr(model, "generation_config", None) is not None:
    #     model.generation_config.bos_token_id = tokenizer.bos_token_id
    #     model.generation_config.eos_token_id = tokenizer.eos_token_id
    #     model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Pre-load reward model
    load_reward_model()
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    
    # GRPO training configuration
    training_args = GRPOConfig(
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
        # GRPO-specific parameters
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        # Generation parameters
        max_prompt_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
        beta=0.001,
    )
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        peft_config=lora_config,
    )
    
    print("="*60)
    print("Starting GRPO Training")
    print("="*60)
    print(f"Policy Model: {MODEL_PATH}")
    print(f"Reward Model: {REWARD_MODEL_PATH}")
    print(f"Dataset: {DATA_PATH} ({len(dataset)} samples)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Generations per prompt: {NUM_GENERATIONS}")
    print("="*60)
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
